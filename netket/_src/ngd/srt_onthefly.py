from collections.abc import Callable
from functools import partial

from einops import rearrange

import jax
import jax.numpy as jnp
from jax.tree_util import Partial, tree_map

from jax.sharding import NamedSharding, PartitionSpec as P

import disparax as dpx

from netket import jax as nkjax
from netket import config
from netket.jax._jacobian.default_mode import JacobianMode
from netket.utils import timing
from netket.utils.types import Array

from netket.jax import _ntk as nt


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "log_psi",
        "solver_fn",
        "chunk_size",
        "mode",
        "_bs",
    ),
)
def srt_onthefly(
    log_psi,
    local_energies,
    parameters,
    model_state,
    samples,
    *,
    diag_shift: float | Array,
    solver_fn: Callable[[Array, Array], Array],
    mode: JacobianMode,
    proj_reg: float | Array | None = None,
    momentum: float | Array | None = None,
    old_updates: Array | None = None,
    chunk_size: int | None = None,
    weights: Array | None = None,
    _bs: int  | None = None
):
    if weights is not None:
        raise NotImplementedError(
            "Weighted samples are not yet supported in srt_onthefly."
        )

    N_mc = local_energies.size

    # Split all parameters into real and imaginary parts separately
    parameters_real, rss = nkjax.tree_to_real(parameters)

    # complex: (Nmc) -> (Nmc,2) - splitting real and imaginary output like 2 classes
    # real:    (Nmc) -> (Nmc,)  - no splitting
    def _apply_fn(model_state, parameters_real, samples):
        variables = {"params": rss(parameters_real), **model_state}
        log_amp = log_psi(variables, samples)

        if mode == "complex":
            re, im = log_amp.real, log_amp.imag
            return jnp.concatenate(
                (re[:, None], im[:, None]), axis=-1
            )  # shape [N_mc,2]
        else:
            return log_amp.real  # shape [N_mc, ]

    def jvp_f_chunk(model_state, parameters, vector, samples):
        r"""
        Creates the jvp of the function `_apply_fn` with respect to the parameters.
        This jvp is then evaluated in chunks of `chunk_size` samples.
        """
        f = lambda params: _apply_fn(model_state, params, samples)
        _, acc = jax.jvp(f, (parameters,), (vector,))
        return acc

    # compute rhs of the linear system
    local_energies = local_energies.flatten()
    de = local_energies - jnp.mean(local_energies)

    # At the moment the final vjp is centered by centering the auxiliary vector a.
    # This is the same as centering the jacobian but may have larger variance.
    dv = 2.0 * de / jnp.sqrt(N_mc)  # shape [N_mc,]
    if mode == "complex":
        dv = jnp.stack([jnp.real(dv), jnp.imag(dv)], axis=-1)  # shape [N_mc,2]
    else:
        dv = jnp.real(dv)  # shape [N_mc,]

    if momentum is not None:
        if old_updates is None:
            old_updates = tree_map(jnp.zeros_like, parameters_real)
        else:
            acc = nkjax.apply_chunked(
                jvp_f_chunk, in_axes=(None, None, None, 0), chunk_size=chunk_size
            )(model_state, parameters_real, old_updates, samples)

            avg = jnp.mean(acc, axis=0)
            acc = (acc - avg) / jnp.sqrt(N_mc)
            dv -= momentum * acc

    if mode == "complex":
        dv = jax.lax.collapse(dv, 0, 2)  # shape [2*N_mc,] or [N_mc, ] if not complex

    # Collect all samples on all MPI ranks, those label the columns of the T matrix
    all_samples = samples
    if config.netket_experimental_sharding:
        samples = jax.lax.with_sharding_constraint(
            samples, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
        )
        all_samples = jax.lax.with_sharding_constraint(
            samples, NamedSharding(jax.sharding.get_abstract_mesh(), P())
        )

    _jacobian_contraction = nt.empirical_ntk_by_jacobian(
        f=Partial(_apply_fn, model_state),
        trace_axes=(),
        vmap_axes=0,
    )

    def jacobian_contraction(samples, all_samples, parameters_real):
        if config.netket_experimental_sharding:
            parameters_real = jax.lax.pvary(parameters_real, "S")
        if chunk_size is None:
            # STRUCTURED_DERIVATIVES returns a complex array, but the imaginary part is zero
            # shape [N_mc/p.size, N_mc, 2, 2]
            return _jacobian_contraction(samples, all_samples, parameters_real).real
        else:
            _all_samples, _ = nkjax.chunk(all_samples, chunk_size=chunk_size)
            ntk_local = jax.lax.map(
                lambda batch_lattice: _jacobian_contraction(
                    samples, batch_lattice, parameters_real
                ).real,
                _all_samples,
            )
            if mode == "complex":
                return rearrange(ntk_local, "nbatches i j z w -> i (nbatches j) z w")
            else:
                return rearrange(ntk_local, "nbatches i j -> i (nbatches j)")

    if _bs is None:
        # If we are sharding, use shard_map manually
        if config.netket_experimental_sharding:
            mesh = jax.sharding.get_abstract_mesh()
            # SAMPLES, ALL_SAMPLES PARAMETERS_REAL
            in_specs = (P("S", None), P(), P())
            out_specs = P("S", None)

            # By default, I'm not sure whether the jacobian_contraction of NeuralTangents
            # Is correctly automatically sharded across devices. So we force it to be
            # sharded with shard map to be sure

            jacobian_contraction_sh = jax.shard_map(
                jacobian_contraction,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
            )

        # This disables the nkjax.sharding_decorator in here, which might appear
        # in the apply function inside.
        with nkjax.sharding._increase_SHARD_MAP_STACK_LEVEL():
            ntk_local = jacobian_contraction_sh(samples, all_samples, parameters_real).real

        # shape [N_mc, N_mc, 2, 2] or [N_mc, N_mc]
        if config.netket_experimental_sharding:
            # this sharding constraint should be useless, but let's keep it for safety.
            ntk = jax.lax.with_sharding_constraint(
                ntk_local, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
            )
        else:
            ntk = ntk_local
        if mode == "complex":
            # shape [2*N_mc, 2*N_mc] checked with direct calculation of J^T J
            ntk = rearrange(ntk, "i j z w -> (i z) (j w)")

        # Center the NTK by avoiding the construction of a big dense matrix to lower memory pressure.
        # Equivalent to the 'old'  delta = jnp.eye(N_mc) - 1 / N_mc
        # ntk = (delta_conc @ (ntk @ delta_conc)) / N_mc
        if mode == "complex":
            ntk = ntk.reshape(N_mc, 2, N_mc, 2)
            col_means = ntk.mean(axis=0, keepdims=True)  # all-reduce
            row_means = ntk.mean(axis=2, keepdims=True)  # all-reduce
            global_mean = col_means.mean(axis=2, keepdims=True)  # local, reuse mean_0
            ntk = ntk - col_means - row_means + global_mean
            ntk = ntk.reshape(2 * N_mc, 2 * N_mc)
        else:
            row_means = ntk.mean(axis=1, keepdims=True)  # local: mean over columns
            col_means = ntk.mean(axis=0, keepdims=True)  # all-reduce: mean over rows
            global_mean = col_means.mean()  # local: col_means is already replicated
            ntk = ntk - col_means - row_means + global_mean

        ntk = ntk / N_mc

        # Create identity matrix with same sharding as ntk: P("S", None)
        if config.netket_experimental_sharding:
            local_size = ntk.shape[0]
            identity = jnp.eye(local_size)
            identity = jax.lax.with_sharding_constraint(
                identity, NamedSharding(jax.sharding.get_abstract_mesh(), P("S", None))
            )
        else:
            identity = jnp.eye(ntk.shape[0])

        # add diag shift
        ntk_shifted = ntk + diag_shift * identity

        # add projection regularization
        if proj_reg is not None:
            ntk_shifted = ntk_shifted + proj_reg / N_mc

        # some solvers return a tuple, some others do not.
        aus_vector = solver_fn(ntk_shifted, dv)
    else:
        def _compute_ntk(p, x, y):
            return jacobian_contraction(x, y, p).real

        bs = _bs
        m, rest = divmod(N_mc, bs)
        if rest > 0:
            raise ValueError('block size needs to divide number of samples')

        f = Partial(_compute_ntk, parameters_real)
        ntk = dpx.compute_blocks(f, all_samples, bs, col_major=True, axis_name='S')

        if mode == "complex":
            raise NotImplementedError
            # return rearrange(ntk, "a i j z w -> a (i z) (j w)")


        if mode == "complex":
            raise NotImplementedError
            # ntk = ntk.reshape(N_mc, 2, N_mc, 2)
            # col_means = ntk.mean(axis=0, keepdims=True)  # all-reduce
            # row_means = ntk.mean(axis=2, keepdims=True)  # all-reduce
            # global_mean = col_means.mean(axis=2, keepdims=True)  # local, reuse mean_0
            # ntk = ntk - col_means - row_means + global_mean
            # ntk = ntk.reshape(2 * N_mc, 2 * N_mc)
        else:
            col_means = dpx.sum_herm(ntk, m, col_major=True, axis=0, axis_name='S') * (1./N_mc)
            def _op(global_mean, ntk, col_means, row_means, mask):
                return ntk + mask * (- col_means[:, None] - row_means[None, :] + global_mean)
            ntk = dpx.block_diagonal_trafo(ntk, col_means, col_major=True, op=Partial(_op, col_means.mean()), axis_name='S')

        ntk = ntk / N_mc

        # add diag shift
        ntk_shifted = dpx.add_diagonal(ntk, jnp.ones((m, bs))*diag_shift, m, col_major=True, axis_name='S')

        # add projection regularization
        if proj_reg is not None:
            raise NotImplementedError
            # ntk_shifted = ntk_shifted + proj_reg / N_mc

        aus_vector = dpx.solve(ntk_shifted, dv.reshape(-1, bs), m, col_major=True, axis_name='S').ravel()

    if isinstance(aus_vector, tuple):
        aus_vector, info = aus_vector
        if info is None:
            info = {}
    else:
        info = {}

    if config.netket_experimental_sharding:
        aus_vector = jax.lax.with_sharding_constraint(
            aus_vector,
            NamedSharding(jax.sharding.get_abstract_mesh(), P("S")),
        )

    aus_vector = jnp.squeeze(aus_vector)
    if mode == "complex":
        aus_vector = aus_vector.reshape((N_mc, 2))

    # Center the vector, equivalent to centering the Jacobian
    # This is equivalent to: aus_vector = delta_conc @ aus_vector
    aus_vector = (aus_vector - jnp.mean(aus_vector, axis=0, keepdims=True)) / jnp.sqrt(
        N_mc
    )
    # shape [N_mc // p.size,2]
    if config.netket_experimental_sharding:
        aus_vector = jax.lax.with_sharding_constraint(
            aus_vector,
            NamedSharding(
                jax.sharding.get_abstract_mesh(),
                P("S", *(None,) * (aus_vector.ndim - 1)),
            ),
        )

    # _, vjp_fun = jax.vjp(f, parameters_real)
    vjp_fun = nkjax.vjp_chunked(
        Partial(_apply_fn, model_state),
        parameters_real,
        samples,
        chunk_size=chunk_size,
        chunk_argnums=1,
        nondiff_argnums=1,
    )

    (updates,) = vjp_fun(aus_vector)  # pytree [N_params,]

    if momentum is not None:
        updates = tree_map(lambda x, y: x + momentum * y, updates, old_updates)
        old_updates = updates

    return rss(updates), old_updates, info

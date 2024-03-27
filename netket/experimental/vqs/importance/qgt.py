from netket.optimizer.qgt.qgt_onthefly import QGTOnTheFlyT, mat_vec_chunked_factory, mat_vec_factory
import netket.jax as nkjax
from sequa.utils import logsumexp_mpi
from netket.utils import mpi
import jax
import jax.numpy as jnp
from functools import partial
from jax.tree_util import Partial


def QGTOnTheFlyImportance(vstate=None, *, chunk_size=None, **kwargs) -> "QGTOnTheFlyT":
    if vstate is None:
        return partial(QGTOnTheFlyImportance, chunk_size=chunk_size, **kwargs)

    if kwargs.pop("diag_scale", None) is not None:
        raise NotImplementedError("\n`diag_scale` argument is not yet supported by QGTOnTheFly." "Please use `QGTJacobianPyTree` or `QGTJacobianDense`.\n\n" "You are also encouraged to nag the developers to support " "this feature.\n\n")

    print("using QGTOnTheFlyImportance")
    vs = vstate
    samples_q = vs.samples_q  # samples from q
    # Ns = samples_q.shape[0] * samples_q.shape[1] * mpi.n_nodes  # in total
    samples_q = samples_q.reshape(-1, samples_q.shape[-1])
    Ns = samples_q.shape[0] * mpi.n_nodes  # in total
    lpqinv = vs.log_w_fun(samples_q)  # log p/q

    denom, _ = logsumexp_mpi(lpqinv)
    denom = denom - jnp.log(Ns)  # <p/q>
    p_q_normalized = jnp.exp(lpqinv - denom)

    samples = samples_q
    pdf = p_q_normalized / Ns

    # when using the pdf the old qgt code
    # does not expect a batch dim, so we flatten here
    samples = samples.reshape(-1, samples.shape[-1])
    pdf = pdf.ravel()

    if chunk_size is None and hasattr(vstate, "chunk_size"):
        chunk_size = vstate.chunk_size

    n_samples = samples.shape[0]  # per rank

    if chunk_size is None or chunk_size >= n_samples:
        mv_factory = mat_vec_factory
        chunking = False
    else:
        samples, _ = nkjax.chunk(samples, chunk_size)
        if pdf is not None:
            pdf, _ = nkjax.chunk(pdf, chunk_size)
        mv_factory = mat_vec_chunked_factory
        chunking = True

    mat_vec = mv_factory(
        forward_fn=vstate._apply_fun,
        params=vstate.parameters,
        model_state=vstate.model_state,
        samples=samples,
        pdf=pdf,
    )
    return QGTOnTheFlyT(
        _mat_vec=mat_vec,
        _params=vstate.parameters,
        _chunking=chunking,
        **kwargs,
    )

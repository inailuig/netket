# Copyright 2023 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from jax.tree_util import register_pytree_node_class
from netket.operator import DiscreteJaxOperator
from ._fermion_operator_2nd_base import FermionOperator2ndBase
from ._fermion_operator_2nd_utils import _is_diag_term


@partial(jax.vmap, in_axes=(0, None, None))
def _reverse_split_cast_term_part(term, site_dtype, dagger_dtype):
    # splits sites and daggers out of terms, casts to desired dtype
    sites, daggers = jnp.array(term)[::-1].reshape([-1, 2]).T
    return sites.astype(site_dtype), daggers.astype(dagger_dtype)


def prepare_terms_list(
    operators,
    constant=None,
    site_dtype=np.uint32,
    dagger_dtype=np.int8,
    weight_dtype=jnp.float64,
    cutoff=0,
):
    # group the terms together with respect to the number of sites they act on
    terms_dicts = {}
    for t, w in operators.items():
        l = len(t)
        d = terms_dicts.get(l, {})
        d[t] = w
        terms_dicts[l] = d
    res = []
    for d in terms_dicts.values():
        w = jnp.array(list(d.values()), dtype=weight_dtype)
        t = np.array(list(d.keys()), dtype=int)
        res.append((w, *_reverse_split_cast_term_part(t, site_dtype, dagger_dtype)))
    if constant is not None and np.abs(constant) > cutoff:
        res.append(
            (
                jnp.array(constant, dtype=weight_dtype).reshape((1,)),
                *_reverse_split_cast_term_part(
                    np.zeros((1, 0), dtype=int), site_dtype, dagger_dtype
                ),
            )
        )
    return res


# TODO implement a version with indexing instead of masks
# TODO implement a version which uses a fori loop to loop over sites
@partial(jax.jit)
def apply_term(x, w, sites, daggers):
    # sites and daggers need to have reversed order (hightest first!)

    # sites can be an unsigned int
    # daggers and x need to be signed, preferably of the same type

    if not jnp.issubdtype(x.dtype, jnp.signedinteger):
        if jnp.issubdtype(x.dtype, jnp.floating):
            pass  # allow float for the time being
        else:
            raise ValueError(
                f"x has incompatible type. expect a signed integer but got {x.dtype}"
            )
    if not jnp.issubdtype(daggers.dtype, jnp.signedinteger):
        raise ValueError(
            f"daggers has incompatible type. expect a signed integer but got {daggers.dtype}"
        )
    if not jnp.issubdtype(sites.dtype, jnp.integer):
        raise ValueError(
            f"sites has incompatible type. expect a integer but got {sites.dtype}"
        )

    # for daggers it's crucial its a signed int, we need it to go negative
    # (we might be able to get away using underflow if we are careful not to cast,
    # but let's not rely on it)

    if len(sites) == 0:  # constant diagonal term
        return x, jnp.full(x.shape[:-1], w), jnp.full(x.shape[:-1], True)

    # TODO precomute all those masks, and pass them instead of the index?

    n_orbitals = x.shape[-1]
    # mask for the sites we are acting on
    # ensure it's the same dtype as daggers
    # e.g. [0,0,1,0]
    ara = jnp.arange(n_orbitals, dtype=sites.dtype)
    masks_site = ara[None] == sites[:, None]

    # mask for all sites up to (not including) each site to flip (will be needed for the jordan-wigner)
    # e.g. [1,1,0,0]
    masks_all_up_to_site = ara < sites[:, None]

    # here we do jordan wigner
    # 1. we start with the raising / lowering operators, as given by daggers
    # raising creates a particle, so we have to add +1
    # and lowering destroys one, so we have to add -1

    # map [0,1] -> [-1, +1] ( by taking 2x-1)
    # be careful about type as daggers_pm can and will be negative
    daggers_pm = daggers - (1 - daggers)
    # compute the action of each creation/annihilation operator in the term
    add_flip = masks_site * daggers_pm[:, None]

    # add a first row of 0, for starting the cumsum
    add_flip_padded = jnp.concatenate(
        [jnp.zeros_like(add_flip[..., :1, :]), add_flip], axis=-2
    )
    # now compute the cumulative action, i.e. what was applied to the state when operator i sees it
    # the first operator gets the initial state which is why we just padded with 0 meaning do nothing
    add_flip_cum = jnp.cumsum(add_flip_padded, axis=-2)

    # now apply the actions we just computed
    # this gives us the state when operator i sees it, after all up to i have been applied
    x_at_i = x[..., None, :] + add_flip_cum[None]

    # the last one (remember, we padded) is the final state:
    x_final_ = x_at_i[..., -1, :]
    # remove final state, now its the state when operator i sees it
    x_at_i = x_at_i[..., :-1, :]

    # we might have just tried to create a fermion in an already
    # occupied orbital, or destroyed one in an empty orbital
    # the resulting matrix element will be zero
    # to get a valid state here we clip, although we could also just set it to the original state
    x_final = jnp.clip(x_final_, 0, 1)

    # compute where we created a fermion in an already occupied orbital or destroyed one in an empty orbital
    # both will result in mel 0
    # version with indexing
    # xi = x_at_i[..., jnp.arange(len(sites), dtype=np.uint32), sites]
    # not_illegal = (xi != daggers[None]).all(axis=-1)
    # version with masks
    illegal = ((x_at_i == daggers[:, None]) * masks_site).any(axis=(-2, -1))
    not_illegal = ~illegal

    # 2. now do the Z gates for the jordan-wigner
    # (for an operator on site i apply Z on all up to site i)

    # jordan-wigner sign
    sgn = 1 - 2 * jnp.remainder(
        jnp.einsum("...aij,ij -> ...a", x_at_i, masks_all_up_to_site), 2
    )

    # compute the final matrix element
    # here we cast, for the case when x is float64 but weights are float32
    # and jax would promote the result to float64
    w_final = w * (not_illegal * sgn).astype(w.dtype)
    # we assume w is not 0, otherwise we would have already removed it
    return x_final, w_final, not_illegal


@partial(jax.vmap, in_axes=(None, 0, 0, 0), out_axes=(-2, -1, -1))
def apply_terms(x, w, sites, daggers):
    return apply_term(x, w, sites, daggers)


@partial(jax.jit, static_argnums=(0, 1))
def get_conn_padded_jax(max_conn_size, dtype, tl_diag, tl_offdiag, x):
    # dtype arg is only needed for the empty case when there are no terms

    if len(tl_diag) == 0 and len(tl_offdiag) == 0:
        xp = x[..., None, :][..., :0, :]
        mels = jnp.zeros(xp.shape[:-1], dtype=dtype)
        n_conn = np.zeros(mels.shape, dtype=int)
        return xp, mels, n_conn

    if len(tl_diag) > 0:
        weight_dtype = tl_diag[-1][0].dtype
        assert weight_dtype == dtype
    if len(tl_offdiag) > 0:
        weight_dtype = tl_offdiag[-1][0].dtype
        assert weight_dtype == dtype

    xp_list = []
    mels_list = []
    nonzero_mask_list = []

    # all terms in the diagonal have the same final state,
    # we sum the mels
    xp_diag_ = x[..., None, :]
    mel_diag_ = jnp.zeros(xp_diag_.shape[:-1], dtype=weight_dtype)
    nonzero_mask_ = jnp.ones(mel_diag_.shape, dtype=jnp.bool_)
    if len(tl_diag) == 0:
        xp_diag_ = xp_diag_[..., :0, :]
        mel_diag_ = mel_diag_[..., :0]
        nonzero_mask_ = nonzero_mask_[..., :0]
    else:
        # iterate over the different length terms (0, 2, 4, ...)
        for w, sites, daggers in tl_diag:
            # we trash xp, dce will make sure we don't even compute it
            _, mels_, _ = apply_terms(x, w, sites, daggers)
            mel_diag_ = mel_diag_ + mels_.sum(axis=-1, keepdims=True)
    # TODO here we could check if the diagonal is < cutoff and set nonzero_mask_ to False

    xp_list.append(xp_diag_)
    mels_list.append(mel_diag_)
    nonzero_mask_list.append(nonzero_mask_)
    # iterate over the different length terms (0, 2, 4, ...)
    for w, sites, daggers in tl_offdiag:
        xp_, mels_, nonzero_mask_ = apply_terms(x, w, sites, daggers)
        xp_list.append(xp_)
        mels_list.append(mels_)
        nonzero_mask_list.append(nonzero_mask_)

    # pad with 0 and old state
    xp_list.append(x[..., None, :])
    mels_list.append(jnp.zeros((x.shape[:-1] + (1,)), dtype=weight_dtype))

    xp_padded = jnp.concatenate(xp_list, axis=-2)
    mels_padded = jnp.concatenate(mels_list, axis=-1)
    nonzero_mask = jnp.concatenate(nonzero_mask_list, axis=-1)

    # move the nonzeros to the beginning

    n_nonzero = nonzero_mask.sum(axis=-1)
    _nonzero_fn = partial(jnp.where, size=max_conn_size, fill_value=-1)
    (i_nonzero,) = jnp.vectorize(_nonzero_fn, signature="(i)->(j)")(nonzero_mask)
    xp_u = jnp.take_along_axis(xp_padded, i_nonzero[..., None], axis=-2)
    mels_u = jnp.take_along_axis(mels_padded, i_nonzero, axis=-1)

    # TODO here would be the place to remove / merge repeated mels
    #
    # you should check that n_nonzero <= max_conn_size outside of jit,
    # and increase max_conn_size if it's not
    return xp_u, mels_u, n_nonzero


@partial(jax.jit, static_argnums=0)
def n_conn_jax(dtype, tl_diag, tl_offdiag, x):
    max_conn_size = 0
    # let dce take care of not computing xp
    _, _, n_conn = get_conn_padded_jax(max_conn_size, dtype, tl_diag, tl_offdiag, x)
    return n_conn


@register_pytree_node_class
class FermionOperator2ndJax(FermionOperator2ndBase, DiscreteJaxOperator):
    r"""
    A fermionic operator in :math:`2^{nd}` quantization.
    Jax implementation.
    """

    def _setup(self, force: bool = False):
        """Analyze the operator strings and precompute arrays for get_conn inference"""
        if force or not self._initialized:
            # TODO ideally we would set dagger_dtype to the same as x
            # however, unfortunately, the dtype of the states in netket
            # is stored in the sampler and not in hilbert, so we don't know it at this stage
            diag_operators = {
                k: v for k, v in self._operators.items() if _is_diag_term(k)
            }
            offdiag_operators = {
                k: v for k, v in self._operators.items() if not _is_diag_term(k)
            }

            self._terms_list_diag = prepare_terms_list(
                diag_operators,
                self._constant,
                site_dtype=np.uint32,
                dagger_dtype=np.int8,
                weight_dtype=self._dtype,
            )
            self._terms_list_offdiag = prepare_terms_list(
                offdiag_operators,
                site_dtype=np.uint32,
                dagger_dtype=np.int8,
                weight_dtype=self._dtype,
            )

            # TODO the following could be reduced further
            self._max_conn_size = int(len(self._terms_list_diag) > 0) + len(
                offdiag_operators
            )
            self._initialized = True

    def tree_flatten(self):
        self._setup()
        data = (
            self._terms_list_diag,
            self._terms_list_offdiag,
        )
        metadata = {
            "hilbert": self.hilbert,
            "operators": self._operators,
            "constant": self._constant,
            "dtype": self.dtype,
            "max_conn_size": self._max_conn_size,
        }
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        hi = metadata["hilbert"]
        constant = metadata["constant"]
        dtype = metadata["dtype"]
        op = cls(hi, [], [], constant=constant, dtype=dtype)
        op._operators = metadata["operators"]
        op._max_conn_size = metadata["max_conn_size"]
        (op._terms_list_diag, op._terms_list_offdiag) = data
        op._initialized = True
        return op

    def to_numba_operator(self) -> "FermionOperator2nd":  # noqa: F821
        """
        Returns the standard numba version of this operator, which is an
        instance of :class:`netket.experimental.operator.FermionOperator2nd`.
        """
        from ._fermion_operator_2nd_numba import FermionOperator2nd

        return self.copy(cls=FermionOperator2nd)

    def _get_conn_padded(self, x):
        self._setup()
        xp, mels, n_conn = get_conn_padded_jax(
            self._max_conn_size,
            self._dtype,
            self._terms_list_diag,
            self._terms_list_offdiag,
            x,
        )
        # TODO if we are outside jit (i don't know how to detect it)
        # we coule check here that _max_conn_size was not too small
        #
        # success = jax.jit(jnp.max)(n_conn) <= self._max_conn_size # jit for gda
        # if not success:
        #     raise ValueError("more connected elements than _max_conn_size")
        #
        # alternatively we could return success
        return xp, mels, n_conn

    def get_conn_padded(self, x):
        xp, mels, _ = self._get_conn_padded(x)
        return xp, mels

    def n_conn(self, x):
        return n_conn_jax(
            self._dtype,
            self._terms_list_diag,
            self._terms_list_offdiag,
            x,
        )

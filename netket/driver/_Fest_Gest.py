import jax
import netket as nk
from functools import partial

from ._batch_utils import *

from netket import jax as nkjax


def tree_mul(a, t):
    # a: scalar
    # t: tree
    return jax.tree_map(lambda x: a * x, t)


def tree_conj(t):
    return jax.tree_map(
        lambda x: jax.lax.conj(x) if jax.numpy.iscomplexobj(x) else x, t
    )


# F1: psi/phi evaluated at samplesphi
# F2: phi/psi evaluated at samplespsi
# G12: O*psi/phi evaluated at samplesphi
# G22: O evaluated at samplespsi

# H12: O*phi/psi evaluated at samplespsi


def _sq(x):
    return jax.numpy.vdot(x, x).real


def _Fest_and_Gest_logwf(
    theta, logphi, logpsi, samplesphi, samplespsi, nonzero_wavefun, phi_samplesphi=None
):
    # logphi(s)
    # logpsi(theta, s)
    if phi_samplesphi is None:
        logphi_samplesphi = logphi(samplesphi)
    logpsi_samplesphi, vjp_fun_G12 = jax.vjp(logpsi, theta, samplesphi)
    psi_over_phi_samplesphi = jax.numpy.exp(logpsi_samplesphi - logphi_samplesphi)
    F1 = psi_over_phi_samplesphi.mean()

    v = psi_over_phi_samplesphi / samplesphi.shape[0]
    G12 = tree_conj(vjp_fun_G12(v)[0])

    logphi_samplespsi = logphi(samplespsi)
    logpsi_samplespsi, vjp_fun_G22 = jax.vjp(logpsi, theta, samplespsi)
    phi_over_psi_samplespsi = jax.numpy.exp(logphi_samplespsi - logpsi_samplespsi)
    F2 = phi_over_psi_samplespsi.mean()

    v = (
        jax.numpy.ones(samplespsi.shape[0], dtype=logpsi_samplespsi.dtype)
        / samplespsi.shape[0]
    )
    G22 = tree_conj(vjp_fun_G22(v)[0])

    v2 = phi_over_psi_samplespsi / samplespsi.shape[0]
    H12 = tree_conj(vjp_fun_G22(v2.conjugate())[0])

    # F1_sq = _sq(psi_over_phi_samplesphi)/psi_over_phi_samplesphi.shape[0]
    # F2_sq = _sq(phi_over_psi_samplespsi)/phi_over_psi_samplespsi.shape[0]

    res = (F1, F2, G12, G22)

    if nonzero_wavefun:
        res = (F1, F2, H12, G22)

    return res


def _Fest_and_Gest(
    theta, phi, psi, samplesphi, samplespsi, nonzero_wavefun, phi_samplesphi=None
):
    # phi(s)
    # psi(theta, s)
    if phi_samplesphi is None:
        phi_samplesphi = phi(samplesphi)
    psi_samplesphi = psi(theta, samplesphi)

    _, vjp_fun_G12 = jax.vjp(psi, theta, samplesphi)

    psi_over_phi_samplesphi = psi_samplesphi / phi_samplesphi
    F1 = psi_over_phi_samplesphi.mean()

    v = 1.0 / phi_samplesphi / samplesphi.shape[0]
    G12 = tree_conj(vjp_fun_G12(v)[0])

    phi_samplespsi = phi(samplespsi)
    psi_samplespsi = psi(theta, samplespsi)

    _, vjp_fun_G22 = jax.vjp(psi, theta, samplespsi)
    phi_over_psi_samplespsi = phi_samplespsi / psi_samplespsi
    F2 = phi_over_psi_samplespsi.mean()

    v = 1.0 / psi_samplespsi / samplespsi.shape[0]
    G22 = tree_conj(vjp_fun_G22(v)[0])

    v2 = phi_over_psi_samplespsi / samplespsi.shape[0]
    H12 = tree_conj(vjp_fun_G22(v2.conjugate())[0])

    res = (F1, F2, G12, G22)

    if nonzero_wavefun:
        res = (F1, F2, H12, G22)

    return res


# def _assemble_FG(F1, F2, G12, G22):
def _assemble_FG(F1, F2, GH12, G22, nonzero_wavefun):

    H12 = G12 = GH12
    # return F and grad F
    F = F1 * F2
    G1 = tree_mul(F2.conjugate(), G12)
    G2 = tree_mul(F, G22)

    if nonzero_wavefun:
        G1 = tree_mul(F1, H12)
    G = jax.tree_map(lambda x, y: 2 * (x - y), G1, G2)
    return F, G


# def _assemble_FGlog(F1, F2, G12, G22):
def _assemble_FGlog(F1, F2, GH12, G22, nonzero_wavefun):
    # return F and grad log F

    H12 = G12 = GH12

    F = F1 * F2
    Glog1 = tree_mul(F2.conjugate() / (F1 * F2), G12)
    Glog2 = G22

    if nonzero_wavefun:
        Glog1 = tree_mul(1.0 / F2, H12)

    Glog = jax.tree_map(lambda x, y: 2 * (x - y), Glog1, Glog2)

    return F, Glog


@partial(jax.jit, static_argnums=(0, 1, 8, 9, 10, 11))
def _Fest_and_Gest_batched(
    jax_forward_phi,
    jax_forward_psi,
    state_phi,
    state_psi,
    params_phi,
    params_psi,
    samples_phi,
    samples_psi,
    gradlog,
    nonzero_wavefun,
    _Fest_and_Gest_fun,
    state_fun,
    phi_samplesphi,
):
    # samples are assumed to be batched, have shape (nbatches, batchsize, ndims...)

    def phi(s):
        return state_fun(jax_forward_phi, state_phi)(params_phi, s)

    def psi(theta, s):
        return state_fun(jax_forward_psi, state_psi)(theta, s)

    def f_(s1, s2, ph_sp=None):
        return _Fest_and_Gest_fun(params_psi, phi, psi, s1, s2, nonzero_wavefun, ph_sp)

    # cant use scan like this since we dont want to store lots of gradients
    # def g_(_, x):
    #    return _, f_(*x)
    # _, FG12s = jax.lax.scan(g_, None, (samples_phi, samples_psi))

    # here we assume phi and psi have the same number of batches

    def g_(i, val):
        if phi_samplesphi is None:
            return jax.tree_map(jax.lax.add, val, f_(samples_phi[i], samples_psi[i]))
        else:
            return jax.tree_map(
                jax.lax.add, val, f_(samples_phi[i], samples_psi[i], phi_samplesphi[i])
            )

    if phi_samplesphi is None:
        FG12_ = f_(samples_phi[0], samples_psi[0])
        FG12_ = jax.lax.fori_loop(1, samples_phi.shape[0], g_, FG12_)
    else:
        FG12_ = f_(samples_phi[0], samples_psi[0], phi_samplesphi[0])
        FG12_ = jax.lax.fori_loop(1, samples_phi.shape[0], g_, FG12_)
    FG12_ = jax.tree_map(
        partial(jax.numpy.multiply, x2=1.0 / samples_phi.shape[0]), FG12_
    )
    FG12s = jax.tree_map(partial(jax.numpy.expand_dims, axis=0), FG12_)

    # since the batches are all the same size we can first take the mean of each batch and then the mean of all batches
    # mean over batches and nodes
    FG12 = jax.tree_map(partial(nk.stats.mean, axis=0), FG12s)  # MPI allreduce

    if gradlog:
        return _assemble_FGlog(*FG12, nonzero_wavefun)
    else:
        return _assemble_FG(*FG12, nonzero_wavefun)


def _get_apply_fun(varstate):
    return nkjax.HashablePartial(varstate._apply_fun, varstate.model)


def apply_with_state_fun(apply_fun, model_state):
    return lambda w, σ: apply_fun({"params": w, **model_state}, σ)


def Fest_and_Gest_nk_vs_batched(
    vs_target,
    vs,
    gradlog=False,
    nonzero_wavefun=False,
    apply_fun_is_logwf=True,
    estbatchsize=None,
    phi_samplesphi=None,
):
    # pick estbatchsize so that it does not consume too much memory

    jax_forward_phi = vs_target._apply_fun
    params_phi = vs_target.parameters
    samples_phi = rebatch(vs_target.samples, estbatchsize)
    state_phi = vs_target.model_state

    jax_forward_psi = vs._apply_fun
    params_psi = vs.parameters
    samples_psi = rebatch(vs.samples, estbatchsize)
    state_psi = vs.model_state

    if phi_samplesphi is not None:
        phi_samplesphi = rebatch(phi_samplesphi, estbatchsize)

    if apply_fun_is_logwf:
        est_fun = _Fest_and_Gest_logwf
    else:
        est_fun = _Fest_and_Gest

    return _Fest_and_Gest_batched(
        jax_forward_phi,
        jax_forward_psi,
        state_phi,
        state_psi,
        params_phi,
        params_psi,
        samples_phi,
        samples_psi,
        gradlog,
        nonzero_wavefun,
        est_fun,
        apply_with_state_fun,
        phi_samplesphi,
    )

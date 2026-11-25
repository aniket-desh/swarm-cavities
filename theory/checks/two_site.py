"""Two-site cavity four-point functions in discrete time (claim3.md eqs. (5)-(12)).

One updating population (fraction rho_b of the N agents; the rest are frozen and drop out), couplings with
variance g^2/N and pair correlation gamma. Time indices run over n = T + 1 rounds.

Inputs are single-site quantities, resolved by site "type" k (weights w_k summing to 1):
    R_k[t, t']  thermal-averaged response of a type-k site, d<s(t)>/dh(t'), zero unless t' < t
    v_k[t, t']  thermal covariance of a type-k site, <ds(t) ds(t')>
Outputs: four-point functions as arrays [t, t', u, u'], all for a pair of distinct updating sites (0, 1), normalized by
the number of updating agents N_b = rho_b N (written N below; for rho_b = 1 the two coincide):
    SS  = N <Sigma_01(t,t') Sigma_01(u,u')>     Sigma_01(t,t') = <ds_0(t) ds_1(t')>
    XX  = N <chi_01(t,t') chi_01(u,u')>         chi_01(t,t')   = d<s_0(t)>/dh_1(t')
    XXt = N <chi_01(t,t') chi_10(u,u')>
    SX  = N <Sigma_01(t,t') chi_01(u,u')>
    SXt = N <Sigma_01(t,t') chi_10(u,u')>
"""
import numpy as np


def four_point(R, v, w, g, gamma, rho_b=1.0):
    R, v, w = np.asarray(R), np.asarray(v), np.asarray(w)
    n = R.shape[-1]
    G2 = rho_b * g**2            # variance of a coupling into the updating population, times N_b
    tau = rho_b * gamma * g**2   # covariance of a reciprocal pair, times N_b
    # site-averaged second moments, M[a, b, c, d] = <A[a, b] B[c, d]>
    MRR = np.einsum("k,kab,kcd->abcd", w, R, R)
    Mvv = np.einsum("k,kab,kcd->abcd", w, v, v)
    MRv = np.einsum("k,kab,kcd->abcd", w, R, v)
    I2 = np.einsum("pq,rs->pqrs", np.eye(n), np.eye(n))
    SS = np.zeros((n,) * 4); XX = SS.copy(); XXt = SS.copy(); SX = SS.copy(); SXt = SS.copy()
    E = lambda spec, *ops: np.einsum(spec, *ops, optimize=True)
    for it in range(4 * n + 4):      # R is strictly causal, so the iteration terminates; loop until unchanged
        # second moments of the effective couplings K_01 = J_01 + sum_jk J_0j chi_jk J_k1 and of the common input c_01
        KK = G2 * I2 + G2**2 * (MRR + XX)                              # N <K_01[p,q] K_01[p',q']>          eq. (7)
        KKt = tau * I2 + tau**2 * (MRR + XXt)                          # N <K_01[p,q] K_10[p',q']>          eq. (8)
        cc = G2**2 * (Mvv + SS)                                        # N <c_01[p,q] c_01[p',q']>          eq. (9)
        cK = G2 * tau * (MRv.transpose(2, 3, 0, 1) + SX)               # N <c_01[p,q] K_01[p',q']>          eq. (10)
        cKt = G2 * tau * (MRv.transpose(2, 3, 0, 1) + SXt)             # N <c_01[p,q] K_10[p',q']>          eq. (11)
        # chi_01 = R_0 K_01 R_1 ;  chi_10 = R_1 K_10 R_0
        XX_new = E("tpux,qrys,pqxy->truw".replace("w", "s"), MRR, MRR, KK)
        XXt_new = E("tpys,qrux,pqxy->trus", MRR, MRR, KKt)
        # Sigma_01 = T1 + T2 + T3,  T1 = R_0 c R_1^T,  T2 = R_0 K_01 v_1,  T3 = v_0 K_10^T R_1^T   (eq. (6))
        # indices: (t, r) = (t, t'), (u, s) = (u, u'); p,q and x,y are the internal times of the two factors
        T1T1 = E("tpux,rqsy,pqxy->trus", MRR, MRR, cc)
        T2T2 = E("tpux,qrys,pqxy->trus", MRR, Mvv, KK)
        T3T3 = E("tquy,rpsx,pqxy->trus", Mvv, MRR, KK)
        T1T2 = E("tpux,rqys,pqxy->trus", MRR, MRv, cK)
        T1T3 = E("tpuy,rqsx,pqxy->trus", MRv, MRR, cKt)
        T2T3 = E("tpuy,sxqr,pqxy->trus", MRv, MRv, KKt)
        sym = lambda A: A + A.transpose(2, 3, 0, 1)
        SS_new = T1T1 + T2T2 + T3T3 + sym(T1T2) + sym(T1T3) + sym(T2T3)
        # N <Sigma_01(t,r) chi_01(u,s)>, chi_01(u,s) = R_0[u,x] K_01[x,y] R_1[y,s]
        SX_new = (E("tpux,rqys,pqxy->trus", MRR, MRR, cK)
                  + E("tpux,ysqr,pqxy->trus", MRR, MRv, KK)
                  + E("uxtq,rpys,xypq->trus", MRv, MRR, KKt))
        # N <Sigma_01(t,r) chi_10(u,s)>, chi_10(u,s) = R_1[u,x] K_10[x,y] R_0[y,s]
        SXt_new = (E("tpys,rqux,pqxy->trus", MRR, MRR, cKt)
                   + E("tpys,uxqr,pqxy->trus", MRR, MRv, KKt)
                   + E("ystq,rpux,pqxy->trus", MRv, MRR, KK))
        delta = max(np.abs(a - b).max() for a, b in [(SS_new, SS), (XX_new, XX), (XXt_new, XXt), (SX_new, SX), (SXt_new, SXt)])
        SS, XX, XXt, SX, SXt = SS_new, XX_new, XXt_new, SX_new, SXt_new
        if delta < 1e-13:
            break
    return dict(SS=SS, XX=XX, XXt=XXt, SX=SX, SXt=SXt, iterations=it + 1)


def pr_over_n(v, w, SS):
    """PR(t) / N_b = vbar(t)^2 / (<v_i(t)^2> + psi_b(t)),  psi_b(t) = N_b <Sigma_ij(t,t)^2>   (claim3.md eq. (13)).

    four_point(..., rho_b) returns functions normalized by the number N_b of updating agents, psi_b = rho_b psi.
    """
    vd = np.einsum("ktt->kt", np.asarray(v))
    psi = np.einsum("tttt->t", SS)
    with np.errstate(invalid="ignore"):
        return (w @ vd) ** 2 / (w @ vd**2 + psi), psi


def linear_single_site(a, g, gamma, sigma, n, rho_b=1.0):
    """Single-site DMFT of x(t+1) = a [J x](t) + sigma xi(t), x(0) = 0, for site types with gains a_k, noise sigma_k.

    Returns R_k[t,t'] (dressed response to a field kick) and v_k[t,t'] = <x(t) x(t')>, both (K, n, n).
    Rbar(t+1, t') = <a_k (delta_{t t'} + Gam sum_u Rbar(t, u) R_k(u, t'))>,  Gam = rho_b gamma g^2.
    """
    a, sigma = np.atleast_1d(a).astype(float), np.atleast_1d(sigma).astype(float)
    K = len(a); w = np.full(K, 1.0 / K)
    Gam = rho_b * gamma * g**2
    R = np.zeros((K, n, n)); Rbar = np.zeros((n, n))
    for t in range(n - 1):
        for k in range(K):
            R[k, t + 1] = a[k] * ((np.arange(n) == t) + Gam * Rbar[t] @ R[k])
        Rbar[t + 1] = w @ R[:, t + 1]
    # x_k = R_k (eta + (sigma_k / a_k) xi):  v_k = R_k (rho_b g^2 Cbar + (sigma_k/a_k)^2 1) R_k^T, solved causally
    v = np.zeros((K, n, n)); Cbar = np.zeros((n, n))
    for t in range(1, n):
        for k in range(K):
            src = rho_b * g**2 * Cbar + (sigma[k] / a[k]) ** 2 * np.eye(n)
            row = R[k, t] @ src @ R[k, : t + 1].T
            v[k, t, : t + 1] = row; v[k, : t + 1, t] = row
        Cbar[t, : t + 1] = w @ v[:, t, : t + 1]; Cbar[: t + 1, t] = Cbar[t, : t + 1]
    return R, v, w

"""Participation-ratio estimators from R replicas of an N-agent state (claim3.md section 6).

X has shape (R, N): row r is the state vector s(t) of replica r (same couplings and private fields, fresh noise).
    pr_naive      (Tr S)^2 / Tr S^2 with the sample covariance S; biased low, 1/PR_naive ~ 1/PR + 1/(R-1)
    pr_wishart    exact debiasing of Tr S^2 and (Tr S)^2 if the data are Gaussian        (eq. (17))
    pr_ustat      distribution-free unbiased estimators of Tr Sigma^2 and (Tr Sigma)^2   (eq. (18))
"""
import numpy as np


def pr_naive(X):
    Xc = X - X.mean(0)
    G = Xc @ Xc.T / (len(X) - 1)            # Gram matrix has the same nonzero spectrum as S
    return np.trace(G) ** 2 / np.sum(G * G)


def pr_wishart(X):
    n = len(X) - 1
    Xc = X - X.mean(0)
    G = Xc @ Xc.T / n
    a, b = np.trace(G) ** 2, np.sum(G * G)  # E a = t1^2 + (2/n) t2 ;  E b = (1 + 1/n) t2 + t1^2 / n
    t2 = n * (n * b - a) / ((n + 2) * (n - 1))
    t1sq = a - 2 * t2 / n
    return t1sq / t2


def _sum_distinct(A):
    """sum over distinct (i, j, k, l) of A_ij A_kl for a symmetric matrix A with zero diagonal."""
    return A.sum() ** 2 - 4 * np.sum(A.sum(0) ** 2) + 2 * np.sum(A * A)


def trace_moments_ustat(X):
    """Unbiased estimates of (Tr Sigma)^2 and Tr Sigma^2 from R >= 4 replicas, for any distribution.

    With d_ij = x_i - x_j:  E |d_12|^2 |d_34|^2 = 4 (Tr Sigma)^2  and  E (d_12 . d_34)^2 = 4 Tr Sigma^2.
    Averaged over all ordered quadruples of distinct replicas using the Gram matrix K = X X^T.
    """
    R = len(X)
    K = X @ X.T
    kd = np.diag(K)
    D = kd[:, None] + kd[None, :] - 2 * K                      # |x_i - x_j|^2, zero diagonal
    n4 = R * (R - 1) * (R - 2) * (R - 3)
    t1sq = _sum_distinct(D) / (4 * n4)
    # (d_ij . d_kl)^2 = (K_ik - K_il - K_jk + K_jl)^2 summed over distinct quadruples, written with K0 = K off-diagonal
    K0 = K - np.diag(kd)
    S1 = np.sum(K0 * K0)
    S2 = np.sum(K0.sum(0) ** 2) - S1                           # sum over distinct (i,j,k) of K_ij K_jk
    S3 = _sum_distinct(K0)
    t2 = S1 / (R * (R - 1)) - 2 * S2 / (R * (R - 1) * (R - 2)) + S3 / n4
    return t1sq, t2


def pr_ustat(X):
    t1sq, t2 = trace_moments_ustat(X)
    return t1sq / t2

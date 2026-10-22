from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.cluster import KMeans


class DiagonalGaussianHMM:
    """Small diagonal Gaussian HMM with an explicit causal filter.

    Baum-Welch is fit only on the training fold. `filter` is a forward-only
    recursion; it never computes smoothed state probabilities.
    """

    def __init__(self, n_states: int = 3, max_iter: int = 100, tol: float = 1e-4,
                 variance_floor: float = 1e-4, random_state: int = 1729):
        self.n_states = n_states
        self.max_iter = max_iter
        self.tol = tol
        self.variance_floor = variance_floor
        self.random_state = random_state

    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        delta = x[:, None, :] - self.means_[None, :, :]
        return -0.5 * np.sum(
            np.log(2.0 * np.pi * self.variances_)[None, :, :] + delta * delta / self.variances_[None, :, :],
            axis=2,
        )

    def _forward_backward(self, log_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        n, k = log_b.shape
        log_a = np.log(np.maximum(self.transmat_, 1e-300))
        alpha = np.empty((n, k), dtype=np.float64)
        alpha[0] = np.log(np.maximum(self.startprob_, 1e-300)) + log_b[0]
        for t in range(1, n):
            alpha[t] = log_b[t] + logsumexp(alpha[t - 1, :, None] + log_a, axis=0)
        loglik = float(logsumexp(alpha[-1]))

        beta = np.zeros((n, k), dtype=np.float64)
        for t in range(n - 2, -1, -1):
            beta[t] = logsumexp(log_a + log_b[t + 1][None, :] + beta[t + 1][None, :], axis=1)

        log_gamma = alpha + beta - loglik
        gamma = np.exp(log_gamma)
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)
        if n > 1:
            log_xi = (
                alpha[:-1, :, None]
                + log_a[None, :, :]
                + log_b[1:, None, :]
                + beta[1:, None, :]
            )
            log_xi -= logsumexp(log_xi, axis=(1, 2), keepdims=True)
            xi_sum = np.exp(log_xi).sum(axis=0)
        else:
            xi_sum = np.zeros((k, k), dtype=np.float64)
        return gamma, xi_sum, loglik

    def fit(self, x: np.ndarray) -> "DiagonalGaussianHMM":
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or len(x) < self.n_states * 5 or not np.isfinite(x).all():
            raise ValueError("HMM training matrix must be finite 2D data with enough rows.")
        k = self.n_states
        labels = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit_predict(x)
        global_var = np.maximum(x.var(axis=0), self.variance_floor)
        means = np.empty((k, x.shape[1]), dtype=np.float64)
        variances = np.empty_like(means)
        for state in range(k):
            group = x[labels == state]
            means[state] = group.mean(axis=0) if len(group) else x[state]
            variances[state] = np.maximum(group.var(axis=0) if len(group) else global_var, self.variance_floor)
        counts = np.ones((k, k), dtype=np.float64)
        np.fill_diagonal(counts, 10.0)
        for left, right in zip(labels[:-1], labels[1:]):
            counts[left, right] += 1.0
        self.startprob_ = np.full(k, 1.0 / k, dtype=np.float64)
        self.transmat_ = counts / counts.sum(axis=1, keepdims=True)
        self.means_ = means
        self.variances_ = variances
        previous = -np.inf

        for _ in range(self.max_iter):
            log_b = self._log_emission(x)
            gamma, xi_sum, loglik = self._forward_backward(log_b)
            weights = np.maximum(gamma.sum(axis=0), 1e-12)
            self.startprob_ = np.maximum(gamma[0], 1e-8)
            self.startprob_ /= self.startprob_.sum()
            row_counts = xi_sum + 1e-6
            self.transmat_ = row_counts / row_counts.sum(axis=1, keepdims=True)
            self.means_ = (gamma.T @ x) / weights[:, None]
            for state in range(k):
                residual = x - self.means_[state]
                self.variances_[state] = np.maximum(
                    (gamma[:, state, None] * residual * residual).sum(axis=0) / weights[state],
                    self.variance_floor,
                )
            if np.isfinite(previous) and abs(loglik - previous) <= self.tol * (1.0 + abs(previous)):
                break
            previous = loglik
        self.log_likelihood_ = float(self._forward_backward(self._log_emission(x))[2])
        self.n_iter_ = _ + 1
        return self

    def filter(self, x: np.ndarray, initial_prior: np.ndarray | None = None) -> np.ndarray:
        """Return P(state_t | observations_1:t), one row per observation."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.means_.shape[1] or not np.isfinite(x).all():
            raise ValueError("HMM filter input has wrong shape or non-finite values.")
        if len(x) == 0:
            return np.empty((0, self.n_states), dtype=np.float64)
        prior = self.startprob_ if initial_prior is None else np.asarray(initial_prior, dtype=np.float64)
        prior = np.maximum(prior, 1e-300)
        prior /= prior.sum()
        log_a = np.log(np.maximum(self.transmat_, 1e-300))
        log_b = self._log_emission(x)
        posterior = np.empty_like(log_b)
        log_alpha = np.log(prior) + log_b[0]
        posterior[0] = np.exp(log_alpha - logsumexp(log_alpha))
        for t in range(1, len(x)):
            log_alpha = log_b[t] + logsumexp(log_alpha[:, None] + log_a, axis=0)
            posterior[t] = np.exp(log_alpha - logsumexp(log_alpha))
        return posterior

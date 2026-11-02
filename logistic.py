import numpy as np
from scipy.special import expit


class LogisticRegressionScratch:
    """L2-regularised logistic regression trained by gradient descent.

    Objective: mean log-loss + (lam / 2) * ||w||^2   (intercept not penalised).
    With lam = 1 / (C * n) this matches scikit-learn's LogisticRegression(C=C).
    """

    def __init__(self, lam=1e-3, max_iter=50_000, tol=1e-9):
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def _loss(self, X, y, w, b):
        z = X @ w + b
        return np.mean(np.logaddexp(0, z) - y * z) + 0.5 * self.lam * (w @ w)

    def _grad(self, X, y, w, b):
        r = expit(X @ w + b) - y
        return X.T @ r / len(y) + self.lam * w, r.mean()

    def fit(self, X, y):
        n, d = X.shape
        Z = np.c_[X, np.ones(n)]
        L = 0.25 * np.linalg.norm(Z, 2) ** 2 / n + self.lam   # gradient Lipschitz bound
        step = 1.0 / L
        w, b = np.zeros(d), 0.0
        self.loss_history_ = []
        for _ in range(self.max_iter):
            gw, gb = self._grad(X, y, w, b)
            w = w - step * gw
            b = b - step * gb
            self.loss_history_.append(self._loss(X, y, w, b))
            if np.sqrt(gw @ gw + gb ** 2) < self.tol:
                break
        self.coef_, self.intercept_ = w, b
        return self

    def predict_proba(self, X):
        return expit(X @ self.coef_ + self.intercept_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)
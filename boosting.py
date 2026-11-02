import numpy as np
from scipy.special import expit

from tree import DecisionTreeScratch


def _leaf_groups(node, X, idx):
    """Yield (leaf_node, row_indices) for the rows of X that fall in each leaf."""
    if node.left is None:
        yield node, idx
        return
    go_left = X[idx, node.feature] <= node.threshold
    yield from _leaf_groups(node.left, X, idx[go_left])
    yield from _leaf_groups(node.right, X, idx[~go_left])


def _tree_predict(root, X):
    out = np.empty(len(X))
    for node, idx in _leaf_groups(root, X, np.arange(len(X))):
        out[idx] = node.value
    return out


class GradientBoostingScratch:
    """Gradient boosting with small regression trees.

    task = 'regression'     : squared error, leaf value = mean residual.
    task = 'classification' : binary logistic loss, leaf value = one Newton step,
                              sum(r) / sum(p(1-p)), as in scikit-learn.
    """

    def __init__(self, task="regression", n_estimators=100, learning_rate=0.1,
                 max_depth=3, min_samples_leaf=1):
        self.task = task
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = len(y)
        if self.task == "classification":
            self.classes_ = np.unique(y)
            if len(self.classes_) != 2:
                raise ValueError("binary classification only")
            y = (y == self.classes_[1]).astype(float)
            p0 = y.mean()
            self.init_ = float(np.log(p0 / (1 - p0)))
        else:
            y = y.astype(float)
            self.init_ = float(y.mean())

        F = np.full(n, self.init_)
        self.trees_, self.train_loss_ = [], []
        for _ in range(self.n_estimators):
            if self.task == "classification":
                p = expit(F)
                r = y - p
                h = p * (1 - p)
            else:
                r = y - F
            tree = DecisionTreeScratch("regression", max_depth=self.max_depth,
                                       min_samples_leaf=self.min_samples_leaf).fit(X, r)
            for node, idx in _leaf_groups(tree.root_, X, np.arange(n)):
                if self.task == "classification":
                    den = h[idx].sum()
                    node.value = 0.0 if abs(den) < 1e-150 else r[idx].sum() / den
                F[idx] += self.learning_rate * node.value
            self.trees_.append(tree)
            if self.task == "classification":
                self.train_loss_.append(np.mean(np.logaddexp(0, F) - y * F))
            else:
                self.train_loss_.append(np.mean((y - F) ** 2))
        return self

    def decision_function(self, X, n_estimators=None):
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_)
        for tree in self.trees_[:n_estimators]:
            F += self.learning_rate * _tree_predict(tree.root_, X)
        return F

    def staged_decision_function(self, X):
        X = np.asarray(X, dtype=float)
        F = np.full(len(X), self.init_)
        for tree in self.trees_:
            F = F + self.learning_rate * _tree_predict(tree.root_, X)
            yield F.copy()

    def predict_proba(self, X):
        return expit(self.decision_function(X))

    def predict(self, X):
        if self.task == "classification":
            return self.classes_[(self.predict_proba(X) > 0.5).astype(int)]
        return self.decision_function(X)
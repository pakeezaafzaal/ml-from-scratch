import numpy as np


class _Node:
    def __init__(self, value, n):
        self.value, self.n = value, n
        self.feature = self.threshold = self.left = self.right = None


class DecisionTreeScratch:
    """CART tree. task = 'classification' (Gini) or 'regression' (squared error).

    A split maximises S = sum(c_L^2)/n_L + sum(c_R^2)/n_R, which is equivalent to
    minimising the weighted child impurity. Thresholds are midpoints between
    consecutive distinct feature values.
    """

    def __init__(self, task="classification", max_depth=None,
                 min_samples_leaf=1, min_samples_split=2):
        self.task = task
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == "classification":
            self.classes_, y_idx = np.unique(y, return_inverse=True)
            T = np.eye(len(self.classes_))[y_idx]
        else:
            T = y.astype(float).reshape(-1, 1)
        self.root_ = self._build(X, T, depth=0)
        return self

    def _is_pure(self, T):
        if self.task == "classification":
            return T.sum(axis=0).max() == len(T)
        return T[:, 0].max() == T[:, 0].min()

    def _best_split(self, X, T):
        n = len(T)
        msl = self.min_samples_leaf
        total = T.sum(axis=0)
        n_left = np.arange(1, n)
        n_right = n - n_left
        size_ok = (n_left >= msl) & (n_right >= msl)
        if not size_ok.any():
            return None, None, None
        best_j, best_thr, best_score = None, None, -np.inf
        for j in range(X.shape[1]):
            order = np.argsort(X[:, j], kind="stable")
            xs = X[order, j]
            left_sums = np.cumsum(T[order], axis=0)[:-1]
            ok = size_ok & (xs[1:] > xs[:-1])
            if not ok.any():
                continue
            score = (left_sums ** 2).sum(axis=1) / n_left \
                + ((total - left_sums) ** 2).sum(axis=1) / n_right
            score = np.where(ok, score, -np.inf)
            i = int(np.argmax(score))
            if score[i] > best_score:
                best_j, best_thr, best_score = j, 0.5 * (xs[i] + xs[i + 1]), score[i]
        return best_j, best_thr, best_score

    def _build(self, X, T, depth):
        node = _Node(T.mean(axis=0) if self.task == "classification" else T[:, 0].mean(),
                     len(T))
        if ((self.max_depth is not None and depth >= self.max_depth)
                or len(T) < self.min_samples_split or self._is_pure(T)):
            return node
        j, thr, score = self._best_split(X, T)
        if j is None:
            return node
        parent_score = (T.sum(axis=0) ** 2).sum() / len(T)
        if score - parent_score <= 1e-12:      # no impurity decrease
            return node
        mask = X[:, j] <= thr
        node.feature, node.threshold = j, thr
        node.left = self._build(X[mask], T[mask], depth + 1)
        node.right = self._build(X[~mask], T[~mask], depth + 1)
        return node

    def _predict_one(self, x):
        node = self.root_
        while node.left is not None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict_proba(self, X):
        return np.array([self._predict_one(x) for x in np.asarray(X, dtype=float)])

    def predict(self, X):
        out = self.predict_proba(X)
        return self.classes_[out.argmax(axis=1)] if self.task == "classification" else out

    def depth(self):
        def d(node):
            return 0 if node.left is None else 1 + max(d(node.left), d(node.right))
        return d(self.root_)

    def n_leaves(self):
        def c(node):
            return 1 if node.left is None else c(node.left) + c(node.right)
        return c(self.root_)
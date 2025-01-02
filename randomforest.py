import numpy as np
from joblib import Parallel, delayed, cpu_count
from sklearn.metrics import mean_squared_error

def bootstrap_sample(X, y):
    """
    Create a bootstrap sample of X and y.
    Returns:
        X_bootstrap, y_bootstrap, oob_mask
    where oob_mask is a boolean array indicating out-of-bag samples.
    """
    if X is None or y is None:
        raise ValueError("X and y cannot be None.")

    n_samples = X.shape[0]
    if y.shape[0] != n_samples:
        raise ValueError("X and y must have the same number of samples.")

    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    oob_mask = np.ones(n_samples, dtype=bool)
    oob_mask[indices] = False

    return X[indices], y[indices], oob_mask


class DecisionTree:
    """
    A Decision Tree extended with multiple "forest_mode" options, including:
      - "classic": Standard variance-minimizing splits (Breiman).
      - "pure_random": Random thresholds for numeric; random subsets for categorical.
      - "totally_random": Random feature + random threshold, ignoring impurity.
      - "centered": Split at the midpoint of the chosen feature's current range.
      - "uniform": Split at a random point in the chosen feature's current range.
      - "purf": Same as 'uniform' for p=1, but labeled distinctly if needed.
    """

    def __init__(self,
                 max_depth=10,
                 min_var=1e-7,
                 subset_features=False,
                 forest_mode="classic",
                 cat_cols=None):
        """
        :param max_depth: Maximum depth of the tree.
        :param min_var: Minimum variance threshold to stop splitting.
        :param subset_features: Whether to use a random subset of features at each split.
        :param forest_mode: One of {'classic', 'pure_random', 'totally_random',
                                    'centered', 'uniform', 'purf'}.
        :param cat_cols: A set (or list) of feature indices considered categorical.
        """
        self.max_depth = max_depth
        self.min_var = min_var
        self.subset_features = subset_features

        # "classic", "pure_random", "totally_random", "centered", "uniform", "purf"
        self.forest_mode = forest_mode.lower()

        self.cat_cols = set(cat_cols) if cat_cols else set()

        # Tree structure
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None  # For numeric splits
        self.categories_left = None  # For categorical splits
        self.is_categorical_split = False

        # Leaf value
        self.value = None

        # For MDI (Mean Decrease in Impurity) tracking
        self.impurity_reduction = {}

    def fit(self, X, y, depth=0):
        """
        Recursively grow the tree.
        """
        if X.shape[0] == 0:
            # No samples => leaf with NaN
            self.value = np.nan
            return

        # Leaf criteria
        if len(set(y)) <= 1 or np.var(y) < self.min_var or depth >= self.max_depth:
            self.value = np.mean(y) if len(y) > 0 else np.nan
            return

        # Feature subset selection
        if self.subset_features:
            n_features = X.shape[1]
            # Like Breiman's approach: pick sqrt(n_features) at random
            features = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        else:
            features = np.arange(X.shape[1])

        # Pick the splitting strategy
        if self.forest_mode == "classic":
            (feature, threshold, best_gain, is_cat_split, cats_left) = self._find_best_split(X, y, features)
        elif self.forest_mode == "pure_random":
            (feature, threshold, best_gain, is_cat_split, cats_left) = self._split_pure_random(X, y, features)
        elif self.forest_mode == "totally_random":
            (feature, threshold, best_gain, is_cat_split, cats_left) = self._split_totally_random(X, y, features)
        elif self.forest_mode == "centered":
            (feature, threshold, best_gain, is_cat_split, cats_left) = self._split_centered(X, y, features)
        elif self.forest_mode == "uniform":
            (feature, threshold, best_gain, is_cat_split, cats_left) = self._split_uniform(X, y, features)
        elif self.forest_mode == "purf":
            # For p=1, this is effectively the same as 'uniform',
            # but you can change the logic if your definition of PURF differs
            (feature, threshold, best_gain, is_cat_split, cats_left) = self._split_uniform(X, y, features)
        else:
            raise ValueError(f"Unknown forest_mode: {self.forest_mode}")

        # If no valid split found => leaf
        if feature is None:
            self.value = np.mean(y)
            return

        # Track impurity reduction for MDI
        self.impurity_reduction[feature] = self.impurity_reduction.get(feature, 0) + best_gain

        # Store split info
        self.feature = feature
        self.threshold = threshold
        self.is_categorical_split = is_cat_split
        self.categories_left = cats_left

        # Split the data
        if is_cat_split:
            left_mask = np.isin(X[:, feature], list(cats_left))
            right_mask = ~left_mask
        else:
            left_mask = X[:, feature] < threshold
            right_mask = ~left_mask

        # Grow children
        self.left = DecisionTree(max_depth=self.max_depth,
                                 min_var=self.min_var,
                                 subset_features=self.subset_features,
                                 forest_mode=self.forest_mode,
                                 cat_cols=self.cat_cols)
        self.left.fit(X[left_mask], y[left_mask], depth + 1)

        self.right = DecisionTree(max_depth=self.max_depth,
                                  min_var=self.min_var,
                                  subset_features=self.subset_features,
                                  forest_mode=self.forest_mode,
                                  cat_cols=self.cat_cols)
        self.right.fit(X[right_mask], y[right_mask], depth + 1)

    def _find_best_split(self, X, y, features):
        """
        'classic' approach: 
        Find the best split by checking all possible midpoints (numeric)
        or partitions (categorical) that minimize variance (MSE).
        """
        best_feat = None
        best_thr = None
        best_gain = 0.0
        is_cat_split = False
        cats_left = None

        n = len(y)
        parent_mse = np.var(y) * n

        for feat in features:
            if feat in self.cat_cols:
                # Attempt all possible non-trivial subsets for categorical
                cats = np.unique(X[:, feat])
                if len(cats) == 1:
                    continue
                from itertools import combinations
                all_subsets = []
                for r in range(1, len(cats)):
                    for combo in combinations(cats, r):
                        all_subsets.append(set(combo))

                for subset in all_subsets:
                    left_mask = np.isin(X[:, feat], list(subset))
                    right_mask = ~left_mask
                    y_left, y_right = y[left_mask], y[right_mask]
                    if len(y_left) == 0 or len(y_right) == 0:
                        continue
                    gain = parent_mse - (np.var(y_left)*len(y_left) + np.var(y_right)*len(y_right))
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = feat
                        best_thr = None
                        is_cat_split = True
                        cats_left = subset
            else:
                # Numeric
                sorted_indices = np.argsort(X[:, feat])
                X_sorted = X[sorted_indices, feat]
                y_sorted = y[sorted_indices]
                unique_vals = np.unique(X_sorted)
                if len(unique_vals) == 1:
                    continue
                midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2
                for thr in midpoints:
                    left_mask = X_sorted < thr
                    right_mask = ~left_mask
                    y_left, y_right = y_sorted[left_mask], y_sorted[right_mask]
                    if len(y_left) == 0 or len(y_right) == 0:
                        continue
                    gain = parent_mse - (np.var(y_left)*len(y_left) + np.var(y_right)*len(y_right))
                    if gain > best_gain:
                        best_gain = gain
                        best_feat = feat
                        best_thr = thr
                        is_cat_split = False
                        cats_left = None

        return best_feat, best_thr, best_gain, is_cat_split, cats_left

    def _split_pure_random(self, X, y, features):
        """
        'pure_random' approach:
          - Pick a random feature from 'features'.
          - If numeric, pick threshold ~ U(min, max).
          - If categorical, pick random subset of categories.
        """
        if len(features) == 0:
            return None, None, 0, False, None

        feat = np.random.choice(features)
        gain = 0.0
        is_cat_split = (feat in self.cat_cols)
        cats_left = None

        if is_cat_split:
            all_cats = np.unique(X[:, feat])
            if len(all_cats) <= 1:
                return None, None, 0, False, None
            subset_size = np.random.randint(1, len(all_cats))
            cats_left = set(np.random.choice(all_cats, subset_size, replace=False))
            threshold = None
        else:
            col_vals = X[:, feat]
            min_val, max_val = np.min(col_vals), np.max(col_vals)
            if min_val == max_val:
                return None, None, 0, False, None
            threshold = np.random.uniform(min_val, max_val)

        return feat, threshold, gain, is_cat_split, cats_left

    def _split_totally_random(self, X, y, features):
        """
        'totally_random' approach:
          - Pick ANY feature from the entire set of columns (not just 'features').
          - If numeric, pick threshold ~ U(min, max).
          - If categorical, pick random subset of categories.
        """
        all_feats = np.arange(X.shape[1])
        if len(all_feats) == 0:
            return None, None, 0, False, None

        feat = np.random.choice(all_feats)
        gain = 0.0
        is_cat_split = (feat in self.cat_cols)
        cats_left = None

        if is_cat_split:
            all_cats = np.unique(X[:, feat])
            if len(all_cats) <= 1:
                return None, None, 0, False, None
            subset_size = np.random.randint(1, len(all_cats))
            cats_left = set(np.random.choice(all_cats, subset_size, replace=False))
            threshold = None
        else:
            col_vals = X[:, feat]
            min_val, max_val = np.min(col_vals), np.max(col_vals)
            if min_val == max_val:
                return None, None, 0, False, None
            threshold = np.random.uniform(min_val, max_val)

        return feat, threshold, gain, is_cat_split, cats_left

    def _split_centered(self, X, y, features):
        """
        'centered' approach:
          - Choose exactly one feature from 'features' at random.
          - Threshold = (min_val + max_val) / 2 in that feature's current range.
          - Ignore improvement gain in selection (gain=0).
        """
        if len(features) == 0:
            return None, None, 0, False, None

        feat = np.random.choice(features)

        # If the chosen feature is categorical, no well-defined 'center'
        # You could skip or degrade to random subset. For simplicity, skip:
        if feat in self.cat_cols:
            return None, None, 0, False, None

        col_vals = X[:, feat]
        min_val, max_val = np.min(col_vals), np.max(col_vals)
        if min_val == max_val:
            return None, None, 0, False, None

        threshold = 0.5 * (min_val + max_val)
        gain = 0.0
        return feat, threshold, gain, False, None

    def _split_uniform(self, X, y, features):
        """
        'uniform' approach:
          - Choose exactly one feature from 'features' at random.
          - Threshold ~ Uniform( min_val, max_val ).
          - If feature is categorical, skip or degrade to None (for simplicity).
        """
        if len(features) == 0:
            return None, None, 0, False, None

        feat = np.random.choice(features)

        if feat in self.cat_cols:
            # Not defined how to pick a "uniform" threshold for categories,
            # so skip or degrade to None.
            return None, None, 0, False, None

        col_vals = X[:, feat]
        min_val, max_val = np.min(col_vals), np.max(col_vals)
        if min_val == max_val:
            return None, None, 0, False, None

        threshold = np.random.uniform(min_val, max_val)
        gain = 0.0
        return feat, threshold, gain, False, None

    def predict(self, X):
        return np.array([self._predict_row(row) for row in X])

    def _predict_row(self, row):
        if self.value is not None:
            return self.value

        if self.is_categorical_split:
            # If row[self.feature] in the left categories, go left
            return (self.left._predict_row(row)
                    if row[self.feature] in self.categories_left
                    else self.right._predict_row(row))
        else:
            # Numeric split
            if row[self.feature] < self.threshold:
                return self.left._predict_row(row)
            else:
                return self.right._predict_row(row)


class CustomRandomForestRegressor:
    """
    Extended Random Forest supporting:
      - 'classic'
      - 'pure_random'
      - 'totally_random'
      - 'centered'
      - 'uniform'
      - 'purf'
    """

    def __init__(self,
                 n_trees=100,
                 max_depth=10,
                 subset_features=False,
                 forest_mode="classic",
                 cat_cols=None):
        """
        :param n_trees: Number of trees in the forest.
        :param max_depth: Maximum depth of each tree.
        :param subset_features: If True, random subset of features is used per split.
        :param forest_mode: One of {'classic','pure_random','totally_random',
                                    'centered','uniform','purf'}.
        :param cat_cols: Indices of categorical features.
        """
        print(f"Initializing RandomForestRegressor with forest_mode={forest_mode}")
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.subset_features = subset_features
        self.forest_mode = forest_mode.lower()
        self.cat_cols = set(cat_cols) if cat_cols else set()

        self.trees = []
        self.oob_masks = []
        self.X_ = None
        self.y_ = None

    def fit(self, X, y, n_jobs=1):
        X, y = self._validate_data(X, y)

        self.X_ = X
        self.y_ = y

        def fit_one_tree(_):
            X_boot, y_boot, oob_mask = bootstrap_sample(X, y)
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_var=1e-7,
                subset_features=self.subset_features,
                forest_mode=self.forest_mode,
                cat_cols=self.cat_cols
            )
            tree.fit(X_boot, y_boot, depth=0)
            return tree, oob_mask

        if n_jobs == 1:
            for _ in range(self.n_trees):
                tree, oob_mask = fit_one_tree(_)
                self.trees.append(tree)
                self.oob_masks.append(oob_mask)
        else:
            n_jobs_eff = cpu_count() if n_jobs < 0 else n_jobs
            results = Parallel(n_jobs=n_jobs_eff)(
                delayed(fit_one_tree)(i) for i in range(self.n_trees)
            )
            self.trees = [r[0] for r in results]
            self.oob_masks = [r[1] for r in results]

    def _validate_data(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        return X, y

    def predict(self, X):
        X = np.array(X)
        if X.ndim != 2:
            raise ValueError("X must be 2D for prediction.")
        if len(self.trees) == 0:
            raise ValueError("Random Forest has not been fitted yet.")

        preds = [tree.predict(X) for tree in self.trees]
        return np.mean(preds, axis=0)

    def feature_importances_mdi(self):
        """
        MDI (Mean Decrease in Impurity)
        """
        total_importance = {}
        for tree in self.trees:
            for feat, imp_val in tree.impurity_reduction.items():
                total_importance[feat] = total_importance.get(feat, 0) + imp_val

        denom = sum(total_importance.values()) if total_importance else 1e-12
        for f in total_importance:
            total_importance[f] /= denom
        return total_importance

    def feature_importances_mda(self, n_repeats=5, random_state=None):
        """
        MDA (Mean Decrease in Accuracy) using OOB samples.
        """
        if random_state is not None:
            np.random.seed(random_state)

        if len(self.oob_masks) == 0:
            return {}

        n_features = self.X_.shape[1]
        importances = np.zeros(n_features)
        n_trees_with_oob = 0

        for idx, tree in enumerate(self.trees):
            oob_mask = self.oob_masks[idx]
            if not np.any(oob_mask):
                continue

            X_oob = self.X_[oob_mask]
            y_oob = self.y_[oob_mask]
            baseline_pred = tree.predict(X_oob)
            baseline_error = mean_squared_error(y_oob, baseline_pred)
            n_trees_with_oob += 1

            for feat in range(n_features):
                perm_errors = []
                X_oob_copy = np.copy(X_oob)
                for _ in range(n_repeats):
                    np.random.shuffle(X_oob_copy[:, feat])
                    perm_pred = tree.predict(X_oob_copy)
                    perm_errors.append(mean_squared_error(y_oob, perm_pred))

                mean_perm_error = np.mean(perm_errors)
                importances[feat] += (mean_perm_error - baseline_error)

        if n_trees_with_oob == 0:
            return {feat: 0.0 for feat in range(n_features)}

        importances /= n_trees_with_oob
        return {feat: val for feat, val in enumerate(importances)}


from joblib import Parallel, delayed
from TreeModelsFromScratch.DecisionTree import DecisionTree
import numpy as np
import pandas as pd
from warnings import warn, catch_warnings, simplefilter
from sklearn.metrics import mean_squared_error, accuracy_score
import numbers
from shap.explainers._tree import SingleTree
from shap import TreeExplainer
from TreeModelsFromScratch.SmoothShap import verify_shap_model, smooth_shap, conf_int_ratio_two_var, conf_int_cohens_d, conf_int_ratio_mse_ratio

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, min_samples_leaf=1, n_feature="sqrt",
                 bootstrap=True, oob=True, oob_SHAP=False, criterion="gini", treetype="classification", HShrinkage=False,
                 HS_lambda=0, HS_smSHAP=False, HS_nodewise_shrink_type=None, cohen_reg_param=2, alpha=0.05,
                 cohen_statistic="f", k=None, random_state=None, testHS=False, depth_dof=False):
        """A random forest model for classification or regression tasks."""

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_features = n_feature
        self.bootstrap = bootstrap
        self.oob = oob
        self.oob_SHAP = oob_SHAP
        self.criterion = criterion
        self.k = k
        self.HShrinkage = HShrinkage
        self.HS_lambda = HS_lambda
        self.HS_smSHAP = HS_smSHAP
        self.HS_nodewise_shrink_type = HS_nodewise_shrink_type
        self.cohen_reg_param = cohen_reg_param
        self.alpha = alpha
        self.cohen_statistic = cohen_statistic
        self.treetype = treetype
        self.random_state = random_state
        self.random_state_ = self._check_random_state(random_state)
        self.trees = []
        self.feature_names = None
        self.smSHAP_HS_applied = False
        self.nodewise_HS_applied = False
        self.testHS = testHS
        self.depth_dof = depth_dof

    def _check_random_state(self, seed):
        if isinstance(seed, numbers.Integral) or seed == None:
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed

    def _build_tree(self, X, y, seed):
        """Builds a single tree with the given seed and training data."""
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            n_features=self.n_features,
            criterion=self.criterion,
            treetype=self.treetype,
            feature_names=self.feature_names,
            HShrinkage=self.HShrinkage,
            HS_lambda=self.HS_lambda,
            k=self.k,
            random_state=seed,
            depth_dof=self.depth_dof
        )

        X_inbag, y_inbag, idxs_inbag = self._bootstrap_samples(X, y, self.bootstrap, seed)
        tree.fit(X_inbag, y_inbag)
        return tree, idxs_inbag, X_inbag, y_inbag

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.oob_preds_tree_id = [[] for _ in range(X.shape[0])] if self.oob else None
        shap_scores_inbag = np.full([X.shape[0], X.shape[1], self.n_trees], np.nan) if self.oob_SHAP else None
        shap_scores_oob = np.full([X.shape[0], X.shape[1], self.n_trees], np.nan) if self.oob_SHAP else None

        seed_list = self.random_state_.randint(np.iinfo(np.int32).max, size=self.n_trees)

        results = Parallel(n_jobs=-1)(
            delayed(self._build_tree)(X, y, seed)
            for seed in seed_list
        )
        
        self.trees, idxs_inbag_list, X_inbag_list, y_inbag_list = zip(*results)

        feature_importance_trees = np.zeros((self.n_trees, X.shape[1]))
        for i, (tree, idxs_inbag, X_inbag, y_inbag) in enumerate(zip(self.trees, idxs_inbag_list, X_inbag_list, y_inbag_list)):
            feature_importance_trees[i, :] = tree.feature_importances_

            if self.oob:
                n_samples = X.shape[0]
                tree.oob_preds = np.full(n_samples, np.nan)

                X_oob, y_oob, idxs_oob = self._oob_samples(X, y, idxs_inbag)
                tree.oob_preds[idxs_oob] = tree.predict(X_oob)

                for j in idxs_oob:
                    self.oob_preds_tree_id[j].append(i)

                if self.HS_nodewise_shrink_type != None:
                    self.apply_nodewise_HS(tree, X_inbag, y_inbag, X_oob, y_oob, shrinkage_type=self.HS_nodewise_shrink_type, HS_lambda=self.HS_lambda, cohen_reg_param=self.cohen_reg_param, alpha=self.alpha, cohen_statistic=self.cohen_statistic, testHS=self.testHS)

                if self.oob_SHAP:
                    shap_scores_inbag_tree = np.full([X.shape[0], X.shape[1]], np.nan)
                    shap_scores_oob_tree = np.full([X.shape[0], X.shape[1]], np.nan)

                    export_tree = tree.export_tree_for_SHAP()
                    explainer_tree = TreeExplainer(export_tree)
                    verify_shap_model(tree, explainer_tree, X_inbag)

                    shap_tree_inbag = explainer_tree.shap_values(X_inbag)
                    shap_tree_oob = explainer_tree.shap_values(X_oob)

                    np.put_along_axis(shap_scores_inbag_tree, idxs_inbag.reshape(idxs_inbag.shape[0], 1), shap_tree_inbag, axis=0)
                    np.put_along_axis(shap_scores_oob_tree, idxs_oob.reshape(idxs_oob.shape[0], 1), shap_tree_oob, axis=0)

                    shap_scores_inbag[:, :, i] = shap_scores_inbag_tree.copy()
                    shap_scores_oob[:, :, i] = shap_scores_oob_tree.copy()

        self.feature_importances_ = np.mean(feature_importance_trees, axis=0)
        if self.oob:
            with catch_warnings():
                simplefilter("ignore", category=RuntimeWarning)
                self.oob_preds_forest = np.nanmean([tree.oob_preds for tree in self.trees], axis=0)
            y_test_oob = y.copy()

            if np.isnan(self.oob_preds_forest).any():
                nan_indxs = np.argwhere(np.isnan(self.oob_preds_forest))
                message = """{} out of {} samples do not have OOB scores. This probably means too few trees were used to compute any reliable OOB estimates. These samples were dropped before computing the oob_score""".format(len(nan_indxs), len(y))
                warn(message)
                mask = np.ones(self.oob_preds_forest.shape[0], dtype=bool)
                mask[nan_indxs] = False
                self.oob_preds_forest = self.oob_preds_forest[mask]
                y_test_oob = y[mask]

            if self.treetype == "classification":
                self.oob_preds_forest = self.oob_preds_forest
                self.oob_score = accuracy_score(y_test_oob, self.oob_preds_forest.round(0))
            elif self.treetype == "regression":
                self.oob_score = mean_squared_error(y_test_oob, self.oob_preds_forest, squared=False)

            if self.HS_nodewise_shrink_type != None:
                self.nodewise_HS_applied = True

            if self.oob_SHAP:
                self.inbag_SHAP_values = np.nanmean(shap_scores_inbag, axis=2)
                self.oob_SHAP_values = np.nanmean(shap_scores_oob, axis=2)

                if self.HS_smSHAP:
                    self.apply_smSHAP_HS(HS_lambda=self.HS_lambda)

    def _bootstrap_samples(self, X, y, bootstrap, random_state):
        if isinstance(random_state, numbers.Integral):
            random_state = np.random.RandomState(random_state)
        if bootstrap:
            n_samples = X.shape[0]
            idxs_inbag = random_state.choice(n_samples, n_samples, replace=True)
            return X[idxs_inbag], y[idxs_inbag], idxs_inbag
        else:
            return X, y, np.arange(X.shape[0])

    def _oob_samples(self, X, y, idxs_inbag):
        mask = np.ones(X.shape[0], dtype=bool)
        mask[idx

s_inbag] = False
        X_oob = X[mask]
        y_oob = y[mask]
        idxs_oob = mask.nonzero()[0]
        return X_oob, y_oob, idxs_oob

    def predict_proba(self, X):
        if self.treetype != "classification":
            message = "This function is only available for classification tasks. This model is of type {}".format(self.treetype)
            warn(message)
            return

        if isinstance(X, pd.DataFrame):
            X = X.values

        predictions = np.array([tree.predict_proba(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([np.mean(pred, axis=0) for pred in tree_preds])

        return predictions

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.treetype == "regression":
            predictions = np.array([tree.predict(X) for tree in self.trees])
            tree_preds = np.swapaxes(predictions, 0, 1)
            predictions = np.mean(tree_preds, axis=1)
            return predictions

        elif self.treetype == "classification":
            predictions = np.argmax(self.predict_proba(X), axis=1)
            return predictions

    def export_forest_for_SHAP(self):
        tree_dicts = []
        for tree in self.trees:
            _, tree_dict = tree.export_tree_for_SHAP(return_tree_dict=True)
            tree_dicts.append(tree_dict)

        if self.treetype == "regression":
            model = [
                SingleTree(t, scaling=1.0 / len(tree_dicts))
                for t in tree_dicts
            ]
        elif self.treetype == "classification":
            model = [
                SingleTree(t, scaling=1.0 / len(tree_dicts), normalize=True)
                for t in tree_dicts
            ]
        return model

    def apply_smSHAP_HS(self, HS_lambda=0):
        if (self.trees[0].HS_applied == True) | (self.smSHAP_HS_applied == True):
            message = "For the given model (selective) hierarchical shrinkage was already applied during fit! Please use an estimator with HSShrinkage=False & HS_smSHAP=False"
            warn(message)
            return

        smSHAP_vals, _, smSHAP_coefs = smooth_shap(self.inbag_SHAP_values, self.oob_SHAP_values)
        self.smSHAP_coefs = smSHAP_coefs
        self.smSHAP_vals = smSHAP_vals

        for tree in self.trees:
            tree.HS_lambda = HS_lambda
            tree._apply_hierarchical_srinkage(HS_lambda=HS_lambda, smSHAP_coefs=smSHAP_coefs)
            tree._create_node_dict()

        self.smSHAP_HS_applied = True

    def apply_nodewise_HS(self, tree, X_inbag, y_inbag, X_oob, y_oob, shrinkage_type="MSE_ratio", HS_lambda=0, cohen_reg_param=2, alpha=0.05, cohen_statistic="f", testHS=False):
        if (tree.HS_applied == True) | (self.nodewise_HS_applied == True):
            message = "For the given model (selective) hierarchical shrinkage was already applied during fit! Please use an estimator with HSShrinkage=False & HS_nodewise=False"
            warn(message)
            return

        _, reest_node_vals_inbag, nan_rows_inbag, y_inbag_p_node = tree._reestimate_node_values(X_inbag, y_inbag)
        _, reest_node_vals_oob, nan_rows_oob, y_oob_p_node = tree._reestimate_node_values(X_oob, y_oob)

        conf_int_nodes = []
        m_nodes = []

        for i in range(tree.n_nodes):
            if shrinkage_type == "MSE_ratio":
                conf_int, m = conf_int_ratio_mse_ratio(y_inbag_p_node[i, :][~np.isnan(y_inbag_p_node[i, :])],
                                                       y_oob_p_node[i, :][~np.isnan(y_oob_p_node[i, :])],
                                                       tree.node_list[i].value,
                                                       node_dict_inbag=reest_node_vals_inbag[i],
                                                       node_dict_oob=reest_node_vals_oob[i],
                                                       alpha=alpha, type=tree.treetype)
                conf_int_nodes.append(conf_int)
                m_nodes.append(m)
            elif shrinkage_type == "effect_size":
                conf_int, m = conf_int_cohens_d(y_inbag_p_node[i, :][~np.isnan(y_inbag_p_node[i, :])],
                                                y_oob_p_node[i, :][~np.isnan(y_oob_p_node[i, :])],
                                                reg_param=cohen_reg_param, alpha=alpha, cohen_statistic=cohen_statistic)
                conf_int_nodes.append(conf_int)
                m_nodes.append(m)

        tree._apply_hierarchical_srinkage(HS_lambda=HS_lambda, m_nodes=m_nodes, testHS=testHS)
        tree._create_node_dict()

        tree.nodewise_HS_dict = {"conf_intervals": conf_int_nodes,
                                 "m_values": m_nodes,
                                 "shrinkage_type": shrinkage_type,
                                 "alpha": alpha,
                                 "reest_node_vals_inbag": reest_node_vals_inbag,
                                 "nan_rows_inbag": nan_rows_inbag,
                                 "reest_node_vals_oob": reest_node_vals_oob,
                                 "nan_rows_oob": nan_rows_oob}

        if shrinkage_type == "effect_size":
            tree.nodewise_HS_dict["cohen_reg_param"] = cohen_reg_param
            tree.nodewise_HS_dict["cohen_statistic"] = cohen_statistic
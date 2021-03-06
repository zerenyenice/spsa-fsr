
# SPFSR: Feature Selection and Ranking via SPSA
# Code developed by V. Aksakalli & Z. D. Yenice
# GPL-3.0, 2019
# Please refer to below for more information:
# https://arxiv.org/abs/1804.05589

import numpy as np
from spfsr_log import SpfsrLog
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.utils import shuffle


class SpfsrKernel:

    def __init__(self, params):
        """
        algorithm parameters initialization
        """
        self._iter_max = params['iter_max']
        self._stall_limit = params['stall_limit']
        self._num_grad_avg = params['num_grad_avg']
        self._num_gain_smoothing = params['num_gain_smoothing']
        self._stratified_cv = params['stratified_cv']
        self._num_cv_reps_grad = params['cv_reps_grad']
        self._num_cv_reps_eval = params['cv_reps_eval']
        self._perf_eval_method = params['perf_eval_method']
        self._num_cv_folds = params['cv_folds']
        self._num_cores = params['n_jobs']
        self._num_features_selected = params['num_features']
        self._measure = params['scoring_metric']
        self._gain_type = params['gain_type']
        self._opt_sign = 1 if params['maximize_score'] else -1
        #####
        self._mon_gain_big_a = params.get('mon_gain_big_a') if params.get('mon_gain_big_a') else 100
        self._mon_gain_a = params.get('mon_gain_a') if params.get('mon_gain_a') else 0.75
        self._mon_gain_alpha = params.get('mon_gain_alpha') if params.get('mon_gain_alpha') else 0.6
        #####
        self._print_freq = 5  # how often do we want to print iteration results
        self._stall_tolerance = 10e-7
        self._perturb_amount = 0.05
        self._gain_min = 0.01
        self._gain_max = 1.0
        self._imp_min = 0.0
        self._imp_max = 1.0
        #####
        self._input_x = None
        self._output_y = None
        self._best_value = self._opt_sign * np.inf
        self._best_std = np.inf
        self._stall_counter = 1
        self._same_count_max = self._iter_max
        self._run_time = -1
        self._curr_imp = None
        self._p = None
        self._features_to_keep = None
        self._selected_features = list()
        self._selected_features_prev = list()
        self._features_to_keep_idx = None
        self._best_features = list()
        self._best_imps = list()
        self._best_iter = -1
        self._gain = -1
        self._raw_gain_seq = list()
        self._curr_imp_prev = None
        self._wrapper = None
        self._imp = None
        self._ghat = None
        self._cv_feat_eval = None  # default goes to training error usage
        self._cv_grad_avg = None  # default goes to training error usage
        self._iter_results = self.prepare_results_dict()
        self._features_names = None

    def set_inputs(self, x, y, wrapper):
        SpfsrLog.logger.debug(f"x shape:{x.shape}")
        SpfsrLog.logger.debug(f"y shape:{y.shape}")
        self._input_x = x
        self._output_y = y
        self._wrapper = wrapper
        self._features_names = list(range(x.shape[1]))
        SpfsrLog.logger.debug(f"feature names: {self._features_names}")

    def shuffle_data(self):
        if any([self._input_x is None, self._output_y is None]):
            raise ValueError('There is no data inside shuffle_data()')
        else:
            self._input_x, self._output_y = shuffle(self._input_x, self._output_y)

    @staticmethod
    def prepare_results_dict():
        iter_results = dict()
        iter_results['values'] = list()
        iter_results['stds'] = list()
        iter_results['gains'] = list()
        iter_results['gains_raw'] = list()
        iter_results['importances'] = list()
        iter_results['feature_names'] = list()
        SpfsrLog.logger.debug('empty iter_results dictionary created')
        return iter_results

    def init_parameters(self):
        # SpfsrLog.logger.info(f'SPFSR gain type: {self._gain_type}')
        SpfsrLog.logger.info('Setting initial parameters...')
        self._p = self._input_x.shape[1]
        SpfsrLog.logger.debug(f'number of features : {self._p}')
        self._curr_imp = np.repeat(0.5, self._p)
        SpfsrLog.logger.debug(f'curr imp : {self._curr_imp}')
        self._ghat = np.repeat(0.0, self._p)
        SpfsrLog.logger.debug(f'ghat : {self._ghat}')
        self._curr_imp_prev = self._curr_imp
        SpfsrLog.logger.debug(f'curr imp prev : {self._ghat}')

    def get_selected_features(self, imp):
        """
        given the importance array, determine which features to select (as indices)
        :param imp: importance array
        :return: indices of selected features
        """
        SpfsrLog.logger.debug("func get_selected_features:")
        selected_features = imp.copy()  # init_parameters
        if self._features_to_keep_idx is not None:
            selected_features[self._features_to_keep_idx] = 1.0  # keep these for sure by setting their imp to 1

        if self._num_features_selected == 0:
            num_features_to_select = np.sum(selected_features >= 0.5)
            if num_features_to_select == 0:
                num_features_to_select = 1   # select at least one!
        else:
            num_features_to_select = np.minimum(
                len(selected_features),
                (
                    (0 if self._features_to_keep_idx is None else len(self._features_to_keep_idx)) +
                    self._num_features_selected
                )
            )
        SpfsrLog.logger.debug(f"number of features to select: {num_features_to_select}")
        return (-selected_features).argsort()[:num_features_to_select]

    def gen_cv_task(self):
        if self._perf_eval_method is 'cv':
            if self._num_cv_reps_grad < 1:
                self._num_cv_reps_grad = 1
                SpfsrLog.logger.warning('cv repeats grad cannot be less than 1')

            if self._num_cv_reps_eval < 1:
                self._num_cv_reps_eval = 1
                SpfsrLog.logger.warning('cv repeats eval cannot be less than 1')

            if self._stratified_cv:
                if self._num_cv_reps_grad > 1:
                    self._cv_grad_avg = RepeatedStratifiedKFold(n_splits=self._num_cv_folds, n_repeats=self._num_cv_reps_grad)
                else:
                    self._cv_grad_avg = StratifiedKFold(n_splits=self._num_cv_folds)

                if self._num_cv_reps_eval > 1:
                    self._cv_feat_eval = RepeatedStratifiedKFold(n_splits=self._num_cv_folds, n_repeats=self._num_cv_reps_eval)
                else:
                    self._cv_feat_eval = StratifiedKFold(n_splits=self._num_cv_folds)

            else:
                if self._num_cv_reps_grad > 1:
                    self._cv_grad_avg = RepeatedKFold(n_splits=self._num_cv_folds, n_repeats=self._num_cv_reps_grad)
                else:
                    self._cv_grad_avg = KFold(n_splits=self._num_cv_folds)

                if self._num_cv_reps_eval > 1:
                    self._cv_feat_eval = RepeatedKFold(n_splits=self._num_cv_folds, n_repeats=self._num_cv_reps_eval)
                else:
                    self._cv_feat_eval = KFold(n_splits=self._num_cv_folds)
        else:
            self._cv_feat_eval = self._cv_grad_avg = None

    def eval_feature_set(self, cv_task, c_imp):
        selected_features = self.get_selected_features(c_imp)
        SpfsrLog.logger.debug(f'in eval_feature_set(), selected features: {selected_features}')
        x_fs = self._input_x[:, selected_features]

        if cv_task:
            scores = cross_val_score(self._wrapper,
                                     x_fs,
                                     self._output_y,
                                     cv=cv_task,
                                     scoring=self._measure,
                                     n_jobs=self._num_cores)

            best_value_mean = round(1-scores.mean(), 3)
            best_value_std = scores.std().round(3)
            del scores
        else:
            SpfsrLog.logger.debug('resubs error')
            self._wrapper.fit(x_fs, self._output_y)
            temp_pred = self._wrapper.predict(x_fs)
            best_value_mean = self._measure._score_func(self._output_y, temp_pred)
            best_value_std = 0.0
        SpfsrLog.logger.debug(f"mean score: {best_value_mean}")
        SpfsrLog.logger.debug(f"std score: {best_value_std}")

        return [best_value_mean, best_value_std]

    def run_spfsr(self):

        for iter_i in range(self._iter_max):
            g_matrix = np.array([]).reshape(0, self._p)

            for g in range(self._num_grad_avg):
                delta = np.where(np.random.sample(self._p) >= 0.5, 1, -1)

                imp_plus = self._curr_imp + self._perturb_amount * delta
                imp_plus = np.maximum(imp_plus, self._imp_min)
                imp_plus = np.minimum(imp_plus, self._imp_max)

                imp_minus = self._curr_imp - self._perturb_amount * delta
                imp_minus = np.maximum(imp_minus, self._imp_min)
                imp_minus = np.minimum(imp_minus, self._imp_max)

                y_plus = self.eval_feature_set(self._cv_grad_avg, imp_plus)[0]
                y_minus = self.eval_feature_set(self._cv_grad_avg, imp_minus)[0]

                g_curr = (y_plus-y_minus)/(2*self._perturb_amount * delta)

                g_matrix = np.vstack([g_matrix, g_curr])

            ghat_prev = self._ghat.copy()
            self._ghat = g_matrix.mean(axis=0)

            if np.count_nonzero(self._ghat) == 0:
                self._ghat = ghat_prev

            if self._gain_type == 'bb':
                if iter_i == 0:
                    self._gain = self._gain_min
                    self._raw_gain_seq.append(self._gain)
                else:
                    imp_diff = self._curr_imp - self._curr_imp_prev
                    ghat_diff = self._ghat - ghat_prev
                    bb_bottom = abs(sum(imp_diff * ghat_diff))
                    if bb_bottom > 0.:
                        self._gain = sum(imp_diff * imp_diff) / bb_bottom
                        self._gain = np.maximum(self._gain_min, (np.minimum(self._gain_max, self._gain)))
                    else:
                        self._gain = self._gain_min
                    self._raw_gain_seq.append(self._gain)
                    if iter_i >= self._num_gain_smoothing:
                        self._gain = np.mean(self._raw_gain_seq[(iter_i+1-self._num_gain_smoothing):(iter_i+1)])
            else:  # gain type == 'mon'
                self._gain = self._mon_gain_a / ((iter_i + self._mon_gain_big_a) ** self._mon_gain_alpha)
                self._raw_gain_seq.append(self._gain)

            SpfsrLog.logger.debug(f'iteration gain smooth = {self._gain:1.3f}')
            SpfsrLog.logger.debug(f'iteration gain raw = {self._raw_gain_seq[-1]:1.3f}')

            self._curr_imp_prev = self._curr_imp.copy()
            self._curr_imp = (self._curr_imp - self._gain * self._ghat).clip(min=self._imp_min, max=self._imp_max)

            self._selected_features_prev = self.get_selected_features(self._curr_imp_prev)
            self._selected_features = self.get_selected_features(self._curr_imp)

            same_feature_counter = 1
            curr_imp_orig = self._curr_imp.copy()
            while np.array_equal(self._selected_features_prev, self._selected_features):
                self._curr_imp = curr_imp_orig - same_feature_counter * self._gain_min * self._ghat
                self._curr_imp = self._curr_imp.clip(min=self._imp_min, max=self._imp_max)
                self._selected_features = self.get_selected_features(self._curr_imp)
                if same_feature_counter >= self._same_count_max:
                    break
                same_feature_counter = same_feature_counter + 1
            SpfsrLog.logger.debug("same_feature_counter = {same_feature_counter}")
            fs_perf_output = self.eval_feature_set(self._cv_feat_eval, self._curr_imp)

            self._iter_results['values'].append(round(self._opt_sign * fs_perf_output[0], 5))
            self._iter_results['stds'].append(round(fs_perf_output[1], 5))
            self._iter_results['gains'].append(round(self._gain, 4))
            self._iter_results['gains_raw'].append(round(self._raw_gain_seq[-1], 4))
            self._iter_results['importances'].append(self._curr_imp)
            self._iter_results['feature_names'].append(self.get_feature_names(self._selected_features))

            if (
                    (
                        (self._opt_sign == 1) &
                        (self._iter_results['values'][iter_i] <= self._best_value - self._stall_tolerance)
                    ) |
                    (
                        (self._opt_sign == -1) &
                        (self._iter_results['values'][iter_i] >= self._best_value + self._stall_tolerance)
                    )
            ):
                self._stall_counter = 1
                self._best_iter = iter_i
                self._best_value = self._iter_results['values'][iter_i]
                self._best_std = self._iter_results['stds'][iter_i]
                self._best_features = self._selected_features
                self._best_imps = self._curr_imp[self._best_features]
            else:
                self._stall_counter = self._stall_counter + 1
                
            if iter_i % self._print_freq == 0:
                SpfsrLog.logger.info(f"iter: {iter_i}, value: {self._iter_results['values'][iter_i]}, "
                            f"std: {self._iter_results['stds'][iter_i]}, "
                            f"num. features: {len(self._selected_features)}, "
                            f"best value: {self._best_value}")
            if self._stall_counter > self._stall_limit:
                break

        SpfsrLog.logger.info(f"SPFSR run completed.")

    def get_feature_names(self, selected_features):
        return [self._features_names[i] for i in selected_features]

    def parse_results(self):
        best_features_names = self.get_feature_names(self._best_features)
        selected_data = self._input_x[:, self._best_features]
        results_values = np.array(self._iter_results.get('values'))
        total_iter_for_opt = np.argmin(results_values) if (self._opt_sign == 1) else np.argmax(results_values)

        return {'wrapper': self._wrapper,
                'measure': self._measure,
                'selected_data': selected_data,
                'iter_results': self._iter_results,
                'features': best_features_names,
                'importance': self._best_imps,
                'num_features': len(self._best_features),
                'total_iter_overall': len(self._iter_results.get('values')),
                'total_iter_for_opt': total_iter_for_opt,
                'best_value': self._best_value,
                'best_std': self._best_std,
                }


import numpy as np
import pandas as pd
from Log import Log
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
import sklearn.metrics as metrics
from sklearn.utils import shuffle


class Spsa:

	def __init__(self, model, num_features_selected=0, num_cores=1):
		"""
		Spsa algorithm initialization parameters:
		"""
		self._opt_sign: int = 1
		self._measure = metrics.scorer.accuracy_scorer
		self._stall_counter: int = 1
		self._stall_limit = 20
		self._stall_tolerance = 10e-7
		self._gain_max = 1.0
		self._gain_min = 0.01
		self._best_value = self._opt_sign * np.inf
		self._best_std = np.inf
		self._imp_min = 0.0
		self._imp_max = 1.0
		self._input_x = None
		self._output_y = None
		self._same_count_max = self.same_features_counter_max()
		self._run_time = -1
		self._curr_imp = None
		self._p = None
		self._num_features_selected = num_features_selected
		self._features_to_keep = None
		self._selected_features = []
		self._selected_features_prev = []
		self._features_to_keep_idx = None
		self._best_features = []
		self._best_imps = []
		self._best_iter = -1
		self._gain = -1
		self._raw_gain_seq = []
		self._curr_imp_prev = None
		self._model = model
		self._imp = None
		self._iter_max = 100
		self._perturb_amount = 0.05
		self._ghat = None
		self._perf_eval_method = 'cv'
		self._num_cv_folds = 5
		self._num_cv_reps_eval = 3
		self._num_cv_reps_grad = 3
		self._num_cores = num_cores
		self._cv_feat_eval = None  # default goes to training error usage
		self._cv_grad_avg = None  # default goes to training error usage
		self._stratified_cv = False
		self._num_gain_smoothing = 3
		self._iter_results = self.prepare_results_dict()
		self._features_names = None
		self._num_grad_avg = 3

	def shuffle(self):
		if any([self._input_x is None, self._output_y is None]):
			raise ValueError('There is no data to shuffle')
		else:
			self._input_x, self._output_y = shuffle(self._input_x, self._output_y)

	@staticmethod
	def prepare_results_dict():
		iter_result = dict()
		iter_result['values'] = list()
		iter_result['stds'] = list()
		iter_result['gain_sequence'] = list()
		iter_result['importances'] = list()
		iter_result['feature_names'] = list()
		Log.logger.debug('empty result dictionary is created')
		return iter_result

	def same_features_counter_max(self):
		counter_max = np.round(self._gain_max/self._gain_min)
		Log.logger.debug(f"counter max = {counter_max}")
		counter_max = counter_max.clip(min=10, max=100)
		Log.logger.debug(f"counter max = {counter_max} - clipped between 10,100")
		return counter_max

	def load_data(self, data, y):
		# legacy, will not be used.
		Log.logger.debug(f"data shape:{data.shape}")
		if isinstance(data, pd.DataFrame):
			Log.logger.debug("data is DataFrame")
			if isinstance(y[0], str):
				Log.logger.debug("y is str")
				self._output_y = data[y].values
				self._input_x = data.drop(y, axis=0).values
			elif isinstance(y[0], int):
				Log.logger.debug("y is int")
				self._output_y = data.iloc[:, y].values
				self._input_x = data.drop(data.columns[y], axis=1).values
			else:
				ValueError('y must be int or str inside list for dataframe')
			self._features_names = data.drop(data.columns[y], axis=0).columns.tolist()
		elif isinstance(data, np.ndarray):
			Log.logger.debug("data is numpy array")
			if isinstance(y, int):
				Log.logger.debug("y is int")
				self._output_y = data[y]
				self._input_x = np.delete(data, y, axis=1)
			else:
				ValueError('y must be int for numpy ndarray')
			self._features_names = list(range(np.delete(data, y, axis=1).shape[1]))
		else:
			ValueError('data must be numpy ndarray or dataframe')
		Log.logger.debug(f"feature names: {self._features_names}")

	def enter_data(self, x, y):
		Log.logger.debug(f"x shape:{x.shape}")
		Log.logger.debug(f"y shape:{y.shape}")
		self._input_x = x
		self._output_y = y
		self._features_names = list(range(x.shape[1]))
		Log.logger.debug(f"feature names: {self._features_names}")

	def initializer(self,):
		Log.logger.info('Setting initial parameters')
		self._p = self._input_x.shape[1]
		Log.logger.debug(f'p value : {self._p}')
		self._curr_imp = np.repeat(0.5, self._p)
		Log.logger.debug(f'curr imp : {self._curr_imp}')
		self._ghat = np.repeat(0.0, self._p)
		Log.logger.debug(f'ghat : {self._ghat}')
		self._curr_imp_prev = self._curr_imp
		Log.logger.debug(f'curr imp prev : {self._ghat}')

	def select_features(self, imp):
		'''
		given the importance array, determines which features to select (as indices)
		:param imp: importance array
		:return: indices of selected features
		'''
		#imp = np.array([0.5, 0.4, 0.3, 0.8])
		#features_to_keep_idx = [1, 3]
		Log.logger.debug("func select_features:")
		selected_features = imp.copy()  # initialize
		#Log.logger.debug(f"\timps: {imp}")
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
		Log.logger.debug(f"number of features to select: {num_features_to_select}")
		return (-selected_features).argsort()[:num_features_to_select]

	def cv_task_gen(self):
		if self._perf_eval_method is 'cv':
			if self._num_cv_reps_grad < 1:
				self._num_cv_reps_grad = 1
				Log.logger.warning('repeat number can not be less than 1 default value (1) used')

			if self._num_cv_reps_eval < 1:
				self._num_cv_reps_eval = 1
				Log.logger.warning('repeat number can not be less than 1 default value (1) used')

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

	def safs_perf(self, cv_task, c_imp):
		Log.logger.debug('performance measure:')
		selected_features = self.select_features(c_imp)
		Log.logger.debug(f'selected features: {selected_features}')
		x_perf = self._input_x[:, selected_features]

		if cv_task:
			scores = cross_val_score(self._model, x_perf,
									 self._output_y,
									 cv=cv_task,
									 scoring=self._measure,
									 n_jobs=self._num_cores)

			best_value_mean = round(1-scores.mean(), 3)
			best_value_std = scores.std().round(3)
			del scores
		else:
			Log.logger.debug('resubs error')
			self._model.fit(x_perf, self._output_y)
			temp_pred = self._model.predict(x_perf)
			best_value_mean = self._measure._score_func(self._output_y, temp_pred)
			best_value_std = 0.0
		Log.logger.debug(f"mean score: {best_value_mean}")
		Log.logger.debug(f"std score: {best_value_std}")

		return [best_value_mean, best_value_std]

	def spsa_kernel(self):
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
				Log.logger.debug("delta plus:")
				y_plus = self.safs_perf(self._cv_grad_avg, imp_plus)[0] #todo
				Log.logger.debug("delta minus:")
				y_minus = self.safs_perf(self._cv_grad_avg, imp_minus)[0] #todo

				g_matrix = np.vstack([g_matrix, (y_plus-y_minus)/(2*self._perturb_amount * delta)])

				# end of loop
			ghat_prev = self._ghat.copy()
			self._ghat = g_matrix.mean(axis=0)

			if np.count_nonzero(self._ghat) == 0:
				self._ghat = ghat_prev

			if iter_i == 0:
				self._gain = self._gain_min
				self._raw_gain_seq.append(self._gain)
			else:
				imp_diff = self._curr_imp - self._curr_imp_prev
				ghat_diff = self._ghat - ghat_prev
				self._gain = sum(imp_diff * imp_diff) / abs(sum(imp_diff * ghat_diff))
				self._gain = np.where((np.isnan(self._gain) | np.isinf(self._gain)), self._gain, self._gain_min)
				self._gain = np.maximum(self._gain_min,(np.minimum(self._gain_max,self._gain)))
				self._raw_gain_seq.append(self._gain)

				if iter_i >= self._num_gain_smoothing: # because of iter start from 0 dont need + 1
					self._gain = np.mean(self._raw_gain_seq[(iter_i+1-self._num_gain_smoothing):(iter_i+1)])

			self._curr_imp_prev = self._curr_imp.copy()
			self._curr_imp = (self._curr_imp - self._gain * self._ghat).clip(min=self._imp_min, max=self._imp_max)

			self._selected_features_prev = self.select_features(self._curr_imp_prev)
			self._selected_features = self.select_features(self._curr_imp)

			same_feature_counter = 1
			curr_imp_orig = self._curr_imp.copy()
			while np.array_equal(self._selected_features_prev, self._selected_features):
				self._curr_imp = curr_imp_orig - same_feature_counter * self._gain_min * self._ghat
				self._curr_imp = self._curr_imp.clip(min=self._imp_min, max=self._imp_max)
				self._selected_features = self.select_features(self._curr_imp)
				if same_feature_counter >= self._same_count_max:
					break
				same_feature_counter = same_feature_counter + 1
			Log.logger.debug("Current importance performance:")
			fs_perf_output = self.safs_perf(self._cv_feat_eval, self._curr_imp)

			self._iter_results['values'].append(round(self._opt_sign * fs_perf_output[0], 5))
			self._iter_results['stds'].append(round(fs_perf_output[1], 5))
			self._iter_results['gain_sequence'].append(self._gain)
			self._iter_results['importances'].append(self._curr_imp)
			self._iter_results['feature_names'].append(self.get_feature_names(self._selected_features))

			if (
					(
							(self._opt_sign == 1) &
							(self._iter_results['values'][iter_i] <= self._best_value - self._stall_tolerance)
					) |
					(
							(self._opt_sign == -1) &
							(self._iter_results['values'][iter_i] <= self._best_value - self._stall_tolerance)
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
			Log.logger.info(f"iter: {iter_i}, value: {self._iter_results['values'][iter_i]}, "
							f"std: {self._iter_results['stds'][iter_i]}, features: {self._selected_features}, "
							f"best value: {self._best_value}")
			if self._stall_counter > self._stall_limit:
				break

	def get_feature_names(self, selected_features):
		return [self._features_names[i] for i in selected_features]

	def parse_results(self):
		best_features_names = self.get_feature_names(self._best_features)
		selected_data = self._input_x[:, self._best_features]

		return {'wrapper': self._model,
				'measure': self._measure,
				'selected_data': selected_data,
				'iter_results': self._iter_results,
				'features': best_features_names,
				'importance': self._best_imps,
				'num_features': len(self._best_features),
				'total_iter': len(self._iter_results.get('values')),
				'best_value': self._best_value,
				'best_std': self._best_std,
				}
















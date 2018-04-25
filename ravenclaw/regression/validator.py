from copy import deepcopy
from pandas import concat, DataFrame
from sklearn.model_selection import KFold
from slytherin import Progress
from slytherin.numbers import beautify_num
from numpy import mean as get_mean

from .regressor import  Regressor

class Validator:
	def __init__(self, regressor, folds = KFold(n_splits=2)):
		self._untrained_regressor = regressor
		self._folds = folds
		self._x = None
		self._y = None
		self._trained_regressors = []
		self._splits = []
		self._validation_results = []
		self._summaries = []

		self._final_regressor = None

	@property
	def regressor_name(self):
		return self._untrained_regressor.name

	def validate(self, data=None, x_cols=None, y_col=None, X=None, y=None, groups=None, echo=True, super_echo=False, other_data = None):
		if data is not None and x_cols is not None and y_col is not None: # data, x_cols, y_col
			pass
		elif data is not None and x_cols is not None: # data, x_cols --> y_col
			_all_cols = list(data.columns)
			_y_cols = [_y for  _y in _all_cols if _y not in x_cols]
			y_col = _y_cols[0]
		elif data is not None and y_col is not None: # data, y_col
			_all_cols = list(data.columns)
			x_cols = [_x for _x in _all_cols if _x != y_col]
		elif X is not None and y is not None: # # X, y
			x_cols = list(X.columns)
			try:
				y_col = y.name
			except:
				y_col = list(y.columns)[0]
			data = concat(objs=[X, y], axis=1)
		else:
			raise SyntaxError('missing arguments!')
		data = data.copy().reset_index(drop=True)

		data_available = data[data[y_col].isnull()==False].reset_index(drop=True)
		data_unavailable = data[data[y_col].isnull()].reset_index(drop=True)


		#X = data[x_cols]
		#y = data[y_col]
		X_available = data_available[x_cols]
		y_available = data_available[y_col]
		X_unavailable = data_unavailable[x_cols]
		y_unavailable = data_unavailable[y_col]

		#y = data[y_col]
		#X_unavailable = X[y.isnull()]
		if other_data is not None:
			other_data = other_data.copy()
			other_cols = [col for col in other_data.columns if col not in x_cols and col!=y_col]
			other_data = other_data[other_cols]
			other_data_available = other_data[y.isnull()==False].reset_index(drop=True)
			other_data_unavailable = other_data[y.isnull()].reset_index(drop=True)

		#self._validation_results = data
		#self._validation_results['regressor'] = self.regressor_name
		#self._validation_results['test_set'] = None
		#self._validation_results['prediction'] = None
		if echo:
			print('regressor:', self._untrained_regressor.name)


		progress = Progress(total=self._folds.get_n_splits())
		split_index = 0
		for train_index, test_index in self._folds.split(X=X_available, y=y_available, groups=groups):
			X_train, X_test = X_available.iloc[train_index], X_available.iloc[test_index]
			y_train, y_test = y_available.iloc[train_index], y_available.iloc[test_index]


			# copy and train the model
			regressor = deepcopy(self._untrained_regressor)
			regressor.train(X=X_train, y=y_train, echo=super_echo)
			self._trained_regressors.append(regressor)

			# test/predict using the model
			regressor.test(data=X_test, y_true=y_test)
			summary = regressor.get_summary_row()
			summary['split'] = split_index
			self._summaries.append(summary)

			data_train = concat([X_train, y_train], axis=1)
			data_test = concat([X_test, y_test], axis=1)
			data_extra = concat([X_unavailable, y_unavailable], axis=1)
			if other_data is not None:
				other_train, other_test = other_data_available.iloc[train_index], other_data_available.iloc[test_index]
				data_train = concat([data_train, other_train], axis=1)
				data_test = concat([data_test, other_test], axis=1)
				data_extra = concat([data_extra, other_data_unavailable], axis=1)
			data_train['set'] = 'training'
			data_test['set'] = 'test'
			data_extra['set'] = 'extra'
			results = concat([data_train, data_test, data_extra], axis=0).reset_index(drop=True)

			results['regressor'] = self._untrained_regressor.name
			results['split'] = split_index
			results['prediction'] = regressor.predict(data=results, echo=False)
			results['error'] = results['prediction'] - y_available

			self._validation_results.append(results)
			#self._validation_results.set_value(col='test_set', index=test_index, value=split_index)
			#self._validation_results.set_value(col='prediction', index=test_index, value=pred)

			self._splits.append({'training_index':train_index, 'test_index':test_index})
			split_index += 1
			if echo:
				progress.show(amount=split_index, text=' | ' + self.display_mean_performance())

		#if echo: print('  ', self.display_mean_performance())
		if echo: print('\n')
		#self._validation_results['error'] = self._validation_results['prediction'] - y

	#def get_validation_results(self):
		#return self._validation_results

	def get_validation_results(self):
		return concat(self._validation_results)

	def get_summary(self):
		summary = concat(self._summaries)
		return summary[['name', 'split', 'num_features', 'training_size', 'test_size', 'training_time', 'test_time', 'mape', 'rmse', 'nrmse']]

	def get_mape(self):
		return [x.get_mape() for x in self._trained_regressors]

	def get_rmse(self):
		return [x.get_rmse() for x in self._trained_regressors]

	def get_nrmse(self):
		return [x.get_nrmse() for x in self._trained_regressors]



	def test_new(self, data, y_true=None, include_cols=None, echo=True):
		progress = Progress(total=len(self.get_trained_regressors()))
		test_data = data.copy()
		results = []
		for index, regressor in enumerate(self.get_trained_regressors()):
			if include_cols is None:
				these_results = test_data.copy()
			else:
				these_results = test_data.copy()[include_cols]
			these_results['regressor'] = self.regressor_name
			these_results['validator_split'] = index


			if y_true is None:
				these_results['prediction'] = regressor.predict(data=test_data)
			else:
				these_results['prediction'] = regressor.test(data=test_data, y_true=y_true)
				these_results['error'] = these_results['prediction'] - y_true
				these_results['mape'] = regressor.get_mape()
				these_results['rmse'] = regressor.get_rmse()
				these_results['nrmse'] = regressor.get_nrmse()
			results.append(these_results)
			if echo: progress.show(amount=index+1, text=self.regressor_name)
		if echo: print('\n')
		final_result = concat(objs=results)
		return final_result

	def train(self, data=None, x_cols=None, y_col=None, X=None, y=None, echo=False):
		self._final_regressor = deepcopy(self._untrained_regressor)
		self._final_regressor.train(X=data, x_cols=x_cols, y_col=y_col, y=y, echo=echo)

	def predict(self, data, return_data = True, echo=False):
		pred = self._final_regressor.predict(data=data, echo=echo)
		if return_data:
			result = data.copy()
			result['regressor'] = self.regressor_name
			result['regressor_mape'] = get_mean(self.get_mape())
			result['regressor_rmse'] = get_mean(self.get_rmse())
			result['regressor_nrmse'] = get_mean(self.get_nrmse())
			result['prediction'] = pred
			return result
		else:
			return pred

	@staticmethod
	def predict_by_validators(validators, data, echo=True):
		progress = Progress(total=len(validators))
		results = []
		for index, validator in enumerate(validators):
			result = validator.predict(data=data, return_data=True, echo=False)
			results.append(result)
			progress.show(amount=index+1, text=validator.regressor_name)
		return concat(results)


	def display_mean_performance(self):
		perf = self.get_summary().mean()
		return f"MAPE:{beautify_num(perf['mape'])}, RMSE:{beautify_num(perf['rmse'])}, nRMSE:{beautify_num(perf['nrmse'])}"

	def get_trained_regressors(self):
		return self._trained_regressors
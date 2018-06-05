from copy import deepcopy
from pandas import concat, DataFrame
from sklearn.model_selection import KFold
from slytherin.progress import ProgressBar

from .Predictor import Predictor
from .Regressor import Regressor
from .Classifier import Classifier
from .convert_data_to_Xy import convert_data_to_Xy

class Validator:
	def __init__(self, predictor, folds = KFold(n_splits=2)):
		"""

		:type predictor: Predictor
		:param folds:
		"""
		self._untrained_predictor = predictor
		self._folds = folds
		self._x = None
		self._y = None
		self._trained_predictors = []
		self._splits = []
		self._validation_results = []
		self._summaries = []

		self._final_predictor = None

	@property
	def predictor_name(self):
		return self._untrained_predictor.name

	def validate(self, data=None, x_cols=None, y_col=None, X=None, y=None, groups=None, echo=True, super_echo=False, other_data = None):
		data_xy = convert_data_to_Xy(data=data, x_cols=x_cols, y_col=y_col, X=X, y=y)
		_data = data_xy['data']
		_x_cols = data_xy['x_cols']
		_y_col = data_xy['y_col']
		_data = _data.copy().reset_index(drop=True)

		data_available = _data[_data[_y_col].isnull() == False].reset_index(drop=True)
		data_unavailable = _data[_data[_y_col].isnull()].reset_index(drop=True)


		#X = data[x_cols]
		#y = data[y_col]
		X_available = data_available[_x_cols]
		y_available = data_available[_y_col]
		X_unavailable = data_unavailable[_x_cols]
		y_unavailable = data_unavailable[_y_col]

		#y = data[y_col]
		#X_unavailable = X[y.isnull()]
		if other_data is not None:
			other_data = other_data.copy()
			other_cols = [col for col in other_data.columns if col not in _x_cols and col != _y_col]
			other_data = other_data[other_cols]
			other_data_available = other_data[y.isnull()==False].reset_index(drop=True)
			other_data_unavailable = other_data[y.isnull()].reset_index(drop=True)
		else:
			other_data_available = None
			other_data_unavailable = None

		#self._validation_results = data
		#self._validation_results['regressor'] = self.regressor_name
		#self._validation_results['test_set'] = None
		#self._validation_results['prediction'] = None
		if echo:
			print('predictor:', self._untrained_predictor.name)


		progress = ProgressBar(total=self._folds.get_n_splits())
		split_index = 0
		for train_index, test_index in self._folds.split(X=X_available, y=y_available, groups=groups):
			X_train, X_test = X_available.iloc[train_index], X_available.iloc[test_index]
			y_train, y_test = y_available.iloc[train_index], y_available.iloc[test_index]


			# copy and train the model
			predictor = deepcopy(self._untrained_predictor)
			predictor.train(X=X_train, y=y_train, echo=super_echo)
			self._trained_predictors.append(predictor)

			# test/predict using the model
			predictor.test(data=X_test, y_true=y_test)
			summary = predictor.get_summary_row()
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

			results['regressor'] = self._untrained_predictor.name
			results['split'] = split_index
			results['prediction'] = predictor.predict(data=results, echo=False)

			if type(self._untrained_predictor) is Regressor:
				results['error'] = results['prediction'] - y_available
			elif type(self._untrained_predictor) is Classifier:
				results['correct_classification'] = results['prediction'] == y_available

			self._validation_results.append(results)
			#self._validation_results.set_value(col='test_set', index=test_index, value=split_index)
			#self._validation_results.set_value(col='prediction', index=test_index, value=pred)

			self._splits.append({'training_index':train_index, 'test_index':test_index})
			split_index += 1
			if echo:
				progress.show(amount=split_index, text=' | performance placeholder ')

		#if echo: print('  ', self.display_mean_performance())
		if echo: print('\n')
		#self._validation_results['error'] = self._validation_results['prediction'] - y

	#def get_validation_results(self):
		#return self._validation_results

	def get_validation_results(self):
		return concat(self._validation_results)

	def get_summary(self):
		summary = concat(self._summaries)
		return summary






	def train(self, data=None, x_cols=None, y_col=None, X=None, y=None, echo=False):
		self._final_predictor = deepcopy(self._untrained_predictor)
		self._final_predictor.train(data=data, x_cols=x_cols, y_col=y_col, X=X, y=y, echo=echo)

	def predict(self, data, return_data = True, echo=False):
		pred = self._final_predictor.predict(data=data, echo=echo)
		if return_data:
			result = data.copy()
			result['predictor'] = self.predictor_name
			result['prediction'] = pred
			return result
		else:
			return pred

	def test_new(self, data, y_true=None, include_cols=None, echo=True):
		progress = ProgressBar(total=len(self.get_trained_predictors()))
		test_data = data.copy()
		results = []
		for index, predictor in enumerate(self.get_trained_predictors()):
			if include_cols is None:
				these_results = test_data.copy()
			else:
				these_results = test_data.copy()[include_cols]
			these_results['predictor'] = self.predictor_name
			these_results['validator_split'] = index


			if y_true is None:
				these_results['prediction'] = predictor.predict(data=test_data)
			else:
				these_results['prediction'] = predictor.test(data=test_data, y_true=y_true)
				if type(predictor) is Regressor:
					these_results['error'] = these_results['prediction'] - y_true
					these_results['mape'] = predictor.performance.mape
					these_results['rmse'] = predictor.performance.rmse
					these_results['nrmse_mean'] = predictor.performance.nrmse_mean
					these_results['nrmse_range'] = predictor.performance.nrmse_range
				elif type(predictor) is Classifier:
					these_results['correct_classification'] = these_results['prediction'] == y_true
					these_results['accuracy'] = predictor.performance.accuracy
					these_results['precision'] = predictor.performance.precision
				else:
					print('Warning! unknown type of classifier!')
			results.append(these_results)
			if echo: progress.show(amount=index+1, text=self.predictor_name)
		if echo: print('\n')
		final_result = concat(objs=results)
		return final_result


	def get_trained_predictors(self):
		return self._trained_predictors

	def get_final_predictor(self):
		return self._final_predictor

	@staticmethod
	def predict_by_validators(validators, data, echo=True):
		"""

		:type validators: list of Validator
		:type data: DataFrame
		:type echo: bool
		:rtype: DataFrame
		"""
		progress = ProgressBar(total=len(validators))
		results = []
		for index, validator in enumerate(validators):
			result = validator.predict(data=data, return_data=True, echo=False)
			result['validator_index'] = index
			results.append(result)
			progress.show(amount=index + 1, text=validator.predictor_name)
		return concat(results)
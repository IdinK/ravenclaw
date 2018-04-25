
from pandas import concat
from slytherin.collections import has_duplicates, get_duplicates
from ..interpretation.get_coefficients import get_coefficients as get_coefs
from ..interpretation.performance import RegressionPerformance
from ..preprocessing.polynomialtransformer import PolynomialTransformer
from ..preprocessing.normalizer import Normalizer

class Predictor:
	def __init__(
			self, predictive_model,
			polynomial_model = PolynomialTransformer(degree=1), normalizer = Normalizer(),
			name = 'noname'
	):
		self._x_cols = None
		self._y_col = None
		self._model = predictive_model
		self._poly = polynomial_model
		self._normalizer = normalizer
		self._x_poly = None
		self._performance = None
		self._name = name

	@property
	def name(self):
		return self._name

	@property
	def x_cols(self):
		return self._x_cols

	@x_cols.setter
	def x_cols(self, x_cols):
		if has_duplicates(list(x_cols)):
			raise ValueError('x_cols has duplicates:', get_duplicates(list(x_cols)))
		self._x_cols = list(x_cols)

	@property
	def y_col(self):
		return self._y_col

	@y_col.setter
	def y_col(self, y_col):
		if type(y_col) is str:
			self._y_col = y_col
		else:
			raise TypeError('y_col should be a string. it is a', type(y_col))

	@property
	def model(self):
		return self._model

	def train(self, data=None, x_cols=None, y_col=None, X=None, y=None, echo=False):
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
		data = data.copy()
		X = data[x_cols]
		y = data[y_col]

		self.x_cols = x_cols
		self.y_col = y_col

		# polynomial transformation
		self._x_poly = self._poly.fit_transform(X=X)

		# normalization
		if self._normalizer is not None:
			self._normalizer.normalize(data=self._x_poly, inplace=True, echo=echo)
		self.model.fit(X=self._x_poly, y=y)

	fit = train

	def preprocess(self, data, echo=False):
		x_test = data[self.x_cols]
		x_test_poly = self._poly.transform(X=x_test)
		if self._normalizer is not None:
			self._normalizer.normalize(data=x_test_poly, inplace=True, echo=echo)
		return x_test_poly

	def predict(self, data, echo=False):
		x_test_poly = self.preprocess(data=data, echo=echo)
		return self.model.predict(X=x_test_poly)

	def test(self, data, y_true_col=None, y_true=None):


		if y_true_col is None and y_true is None:
			y_true_col = self.y_col

		if y_true is None:
			y_true = data[y_true_col]

		y_pred = self.predict(data=data)

		if y_pred.shape != y_true.shape:
			raise ValueError('y_pred is', y_pred.shape, 'but y_true is', y_true.shape)

		self._performance = RegressionPerformance(y_pred=y_pred, y_true=y_true)
		return y_pred

	def get_rmse(self):
		return self._performance.rmse

	def get_mape(self):
		return self._performance.mape

	def get_nrmse(self):
		return self._performance.nrmse


	def get_coefficients(self):
		return get_coefs(data=self.get_training_data(), polynomial=self._poly, model=self.model, normalizer = self._normalizer)

	def get_performance(self):
		return self._performance
from copy import deepcopy
from pandas import concat, DataFrame
from slytherin.collections import has_duplicates, get_duplicates
from slytherin.time import Timer
from ..preprocessing.polynomialtransformer import PolynomialTransformer
from ..preprocessing.normalizer import Normalizer



class Predictor:
	def __init__(
			self, predictive_model, name = 'predictor',
			polynomial_model = PolynomialTransformer(degree=1), normalizer = Normalizer()
	):
		self._trained = False
		self._tested = False

		self._x_cols = None
		self._y_col = None
		self._model = predictive_model
		self._name = name
		self._poly = polynomial_model
		self._normalizer = normalizer

		self._y_true = None
		self._y_pred = None

		self._x_cols = None
		self._y_col = None
		self._model = deepcopy(predictive_model)
		self._poly = deepcopy(polynomial_model)
		self._normalizer = deepcopy(normalizer)

		self._coefficients = None
		self._features = None
		self._num_features = None

		self._training_size = None
		self._test_size = None

		self._training_time = None
		self._prediction_time = None

		self._performance = None

	# properties
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

	@property
	def name(self):
		return self._name

	@property
	def prediction_time(self):
		return self._prediction_time

	@property
	def training_time(self):
		return self._training_time

	@property
	def num_features(self):
		return self._num_features

	@property
	def training_size(self):
		return self._training_size

	@property
	def test_size(self):
		return self._test_size

	def _learn_structure(self, data=None, x_cols=None, y_col=None, X=None, y=None):
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
		return {'data':data, 'X':X, 'y':y}


	def train(self, data=None, x_cols=None, y_col=None, X=None, y=None, echo=False):
		if self._trained:
			raise SyntaxError('you cannot train a model that is already trained!')
		timer = Timer()
		prepared_data = self._learn_structure(data=data, x_cols=x_cols, y_col=y_col, X=X, y=y)
		data = prepared_data['data']
		X = prepared_data['X']
		y = prepared_data['y']

		# polynomial transformation
		if self._poly is None:
			x_poly = X.copy()
		else:
			x_poly = self._poly.fit_transform(X=X)

		# normalization
		if self._normalizer is not None:
			self._normalizer.normalize(data=x_poly, inplace=True, echo=echo)
		self.model.fit(X=x_poly, y=y)
		self._training_time = timer.get_elapsed()

		self._num_features = x_poly.shape[1]
		self._training_size = x_poly.shape[0]
		self._trained = True

	fit = train

	def preprocess(self, data, echo=False):
		data = data.copy()
		x_test = data[self.x_cols]
		if self._poly is None:
			x_test_poly = x_test
		else:
			x_test_poly = self._poly.transform(X=x_test)
		if self._normalizer is not None:
			self._normalizer.normalize(data=x_test_poly, inplace=True, echo=echo)
		return x_test_poly

	def predict(self, data, echo=False):
		if not self._trained:
			raise SyntaxError('you cannot use an untrained model for prediction!')
		timer = Timer()
		x_test_poly = self.preprocess(data=data, echo=echo)
		self._y_pred = self.model.predict(X=x_test_poly)
		self._prediction_time = timer.get_elapsed()
		return self._y_pred

	def test(self, data, y_true_col=None, y_true=None, echo=False):
		if not self._trained:
			raise SyntaxError('you cannot use an untrained model for test!')

		if y_true_col is None and y_true is None:
			y_true_col = self.y_col

		if y_true is None:
			y_true = data[y_true_col]

		self.predict(data=data)
		self._test_size = data.shape[0]
		if self._y_pred.shape != y_true.shape:
			raise ValueError('y_pred is', self._y_pred.shape, 'but y_true is', y_true.shape)
		self._y_true = y_true
		self._performance = None
		self._tested = True
		return self._y_pred

	def get_summary_row(self):
		if not self._tested:
			raise SyntaxError('you cannot get the summary of an untested model!')
		if not self._trained:
			raise SyntaxError('you cannot get the summary of an untrained model!')
		return DataFrame(data={
			'name': self.name,
			'num_features': self.num_features,
			'training_size': self.training_size,
			'training_time': self.training_time,
			'test_size': self.test_size,
			'test_time': self.prediction_time
		}, index=[0])

from sklearn.linear_model import LinearRegression

from .regressor import Regressor
from ..preprocessing import Normalizer
from ..preprocessing import PolynomialTransformer

from slytherin.time import Timer

class RegressorGroup(Regressor):
	def __init__(self, predictive_model = LinearRegression(),
			polynomial_model = PolynomialTransformer(degree=1), normalizer = Normalizer(),
			name = 'noname'
	):
		super().__init__(
			predictive_model=predictive_model, polynomial_model=polynomial_model, normalizer=normalizer,
			name=name
		)
		self._regressors = {}
		self._group_cols = []

	@property
	def group_cols(self):
		return self._group_cols

	def train(self, data=None, x_cols=None, y_col=None, X=None, y=None, group_cols = None, echo=False):
		timer = Timer()
		prepared_data = self._prepare_data(data=data, x_cols=x_cols, y_col=y_col, X=X, y=y, echo=echo)
		data = prepared_data['data']


		if group_cols is not None:
			for group_key, group_data in data.groupby(group_cols):
				the_copy = group_data.copy()
				X = the_copy[self.x_cols].drop(axis=1, labels=group_cols)
				y = the_copy[self.y_col]

				regressor = Regressor(predictive_model=self._model, polynomial_model=self._poly, normalizer=self._normalizer, name=self.name)
				regressor.train(X=X, y=y, echo=echo)
				self._regressors[group_key] = regressor



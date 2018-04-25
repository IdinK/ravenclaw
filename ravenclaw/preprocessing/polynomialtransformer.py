from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame

class PolynomialTransformer:
	def __init__(self, degree=1, interaction_only=False, include_bias=True):
		self._poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
		self._features = None

	def fit(self, X, y=None):
		self._features = list(X.columns)
		self._poly.fit(X=X, y=y)
		return self

	@property
	def features(self):
		return self._features

	def get_feature_names(self):
		return self._poly.get_feature_names(input_features=self.features)

	def get_params(self, deep=True):
		return self._poly.get_params(deep=deep)

	def fit_transform(self, X, y=None, **fit_params):
		self._features = list(X.columns)
		X_new = self._poly.fit_transform(X=X, y=y, **fit_params)
		#print(type(X_new), X_new, X.index, self._features)
		X_new = DataFrame(data=X_new, index=X.index, columns=self.get_feature_names())
		return X_new

	def transform(self, X):
		X_new = self._poly.transform(X=X[self._features])
		X_new = DataFrame(data=X_new, index=X.index, columns=self.get_feature_names())
		return X_new







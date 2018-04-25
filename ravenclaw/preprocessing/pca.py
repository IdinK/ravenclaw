from sklearn.decomposition import PCA as ThePCA
from pandas import DataFrame, Series
from math import log10, floor

class PCA:
	def __init__(self, num_components, only_cols = None, except_cols = []):
		self._only_cols = only_cols
		self._except_cols = except_cols
		self._cols = None
		self._normalizers = {}
		self._trained = False
		self._num_components = num_components
		self._pca = ThePCA(n_components=num_components)

		leading_zeros = int(floor(log10(self._num_components))) + 1
		self._pca_features = ['pca_' + str(pca).zfill(leading_zeros) for pca in range(1, self.num_components + 1)]
		self._features = None

	@property
	def pca_features(self):
		return self._pca_features

	@property
	def features(self):
		return self._features

	@property
	def num_components(self):
		return self._num_components

	def train(self, X):
		if self._only_cols is None:
			columns = X.columns
		else:
			columns = self._only_cols

		columns = [col for col in columns if col not in self._except_cols]
		X = X.copy()[columns]
		self._features = columns

		result = DataFrame(data=self._pca.fit_transform(X=X), index=X.index, columns=self.pca_features)
		self._trained = True
		return result

	fit = train

	def get_explained_variance_ratio(self):
		return Series(data=self._pca.explained_variance_ratio_, index=self.pca_features)

	def get_components(self, transpose=False):
		result = DataFrame(data=self._pca.components_, columns=self.features, index=self.pca_features)
		if transpose:
			return DataFrame.transpose(result)
		else:
			return result

	def transform(self, X):
		X = X.copy()[self.features]
		return DataFrame(data=self._pca.transform(X=X), columns=self.pca_features)


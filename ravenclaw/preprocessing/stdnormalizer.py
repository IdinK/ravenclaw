from numpy import isnan, isinf
from pandas import Series

class StdNormalizer:
	def __init__(self, column):
		self.mean = None
		self.std = None
		self.column = column
		self.trained = False


	@property
	def proper(self):
		improper = isnan(self.mean) or isinf(self.mean) or isnan(self.std) or isinf(self.std) or (self.std==0)
		return not improper

	def train(self, data = None, series = None, means = None, stds = None, echo=True):
		if data is None:
			self.mean = means[self.column]
			self.std = stds[self.column]
		else:
			if type(data) is Series:
				series = data
			else:
				series = data[self.column]

			try:
				self.mean = series.mean()
			except:
				print('cannot normalize these:')
				print(series)
				raise ValueError('cannot normalize!')
			self.std = series.std()
		self.trained = True
		if echo: print(self.column, 'trained for normalization.')

	def normalize(self, data, inplace=False, echo=True):
		if inplace:
			new_data = data
		else:
			new_data = data.copy()

		if self.trained is False:
			raise ReferenceError('StdNormalizer not trained yet.')
		if self.proper:
			new_data[self.column] = (new_data[self.column] - self.mean)/self.std
			if echo: print(self.column, 'normalized.')
		else:
			if echo: print(self.column, 'not normalizable..')
		return new_data

	def denormalize(self, data, inplace=False, echo=True):
		if inplace:
			new_data = data
		else:
			new_data = data.copy()
		if self.trained is False:
			raise ReferenceError('StdNormalizer not trained yet.')
		if self.proper:
			new_data[self.column] = new_data[self.column] * self.std + self.mean
			if echo: print(self.column, 'denormalized.')
		else:
			if echo: print(self.column, 'not denormalizable.')
		return new_data

	def __repr__(self):
		return f"mean:{self.mean}  std:{self.std}  proper:{self.proper}"

class Normalizer:
	def __init__(self, only_cols = None, except_cols = []):
		self._only_cols = only_cols
		self._except_cols = except_cols
		self._cols = None
		self._normalizers = {}
		self._trained = False


	@property
	def trained(self):
		return self._trained

	@property
	def cols(self):
	    return self._cols

	@property
	def normalizers(self):
		return self._normalizers


	def train(self, data, echo=True):
		if self._only_cols is None:
			columns = data.columns
		else:
			columns = self._only_cols

		stds = data.std()
		means = data.mean()

		for col in columns:
			if col not in self._except_cols:
				normalizer = StdNormalizer(column=col)
				normalizer.train(data=data, means=means, stds=stds, echo=echo)
				self._normalizers[col] = normalizer
		self._trained = True

	def normalize(self, data, inplace=False, echo=True):
		if inplace:
			new_data = data
		else:
			new_data = data.copy()

		if not self._trained:
			self.train(data=data, echo=echo)

		for col, normalizer in self._normalizers.items():
			normalizer.normalize(data=new_data, inplace=True, echo=echo)
		return new_data

	def denormalize(self, data, inplace=False, echo=True):
		if inplace:
			new_data = data
		else:
			new_data = data.copy()

		if not self._trained:
			raise ReferenceError('normalizer not trained.')

		for col, normalizer in self._normalizers.items():
			normalizer.denormalize(data=new_data, inplace=True, echo=echo)
		return new_data



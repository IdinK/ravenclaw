from pandas import concat as pd_concat
from .wrangling import columns_are_unique

class ModelData:
	def __init__(self, X, y, Z=None):
		self._x_cols = list(X.columns)
		try:
			self._y_col = y.name
		except:
			self._y_col = list(y.columns)[0]

		if Z is None:
			self._z_cols = None
			the_data = pd_concat(objs=[X, y], axis=1)
		else:
			self._z_cols = list(Z.columns)
			the_data = pd_concat(objs=[X, Z, y], axis=1)

		self._data = the_data.copy().reset_index(drop=True)

	@classmethod
	def from_data(cls, data, x_cols, y_col, z_cols=None, reset_index=True):
		X = data[x_cols]
		y = data[y_col]
		if z_cols is None:
			Z=None
		else:
			Z = data[z_cols]
		return cls(X=X, y=y, Z=Z)

	@property
	def X(self):
		return self._data[self._x_cols]

	@property
	def y(self):
		return self._data[self._y_col]

	@property
	def Z(self):
		return self._data[self._z_cols]

	@property
	def data(self):
		return self._data

	def drop_z(self):
		return self.__init__(X=self.X, y=self.y)

	def select(self, indices=None, drop_z=False):
		result = self.from_data(data=self.data.iloc[indices], x_cols=self._x_cols, y_col=self._y_col, z_cols=self._z_cols)
		if drop_z:
			result = result.drop_z()
		return result

	def select_valid_y(self):
		return self.from_data(
			data=self.data[self.y.isnull()==False],
			x_cols=self._x_cols, y_col=self._y_col, z_cols=self._z_cols)

	def select_invalid_y(self):
		return self.from_data(
			data=self.data[self.y.isnull()==True],
			x_cols=self._x_cols, y_col=self._y_col, z_cols=self._z_cols)

	def add_column_to_z(self, values, column_name):
		if self._z_cols is None:
			self._z_cols = [column_name]
		else:
			self._z_cols.append(column_name)
		self.data[column_name] = values

	def concat(self, other = None, others = None):
		if other is not None and others is None:
			the_data = pd_concat(objs=[self.data, other.data], axis=0)
		elif others is not None and other is None:
			the_data = pd_concat(objs=[self.data] + [other.data for other in others], axis=0)
		else:
			raise ValueError("either other or others should be None.")
		return self.from_data(data=the_data, x_cols=self._x_cols, y_col=self._y_col, z_cols=self._z_cols, reset_index=True)

	def split(self, kfold):
		invalid_md = self.select_invalid_y()
		invalid_md.add_column_to_z(values='extra', column_name='set')

		valid_md = self.select_valid_y()
		for train_indices, test_indices in kfold.split(X=valid_md.X):

			training_md = valid_md.select(indices=train_indices)
			training_md.add_column_to_z(column_name='set', values='training')
			test_md = valid_md.select(indices=test_indices)
			test_md.add_column_to_z(column_name='set', values='test')

			yield training_md, test_md, invalid_md


	def split2(self, kfold):
		for train_indices, test_indices in kfold.split(X=self.select_valid_y().X):
			return train_indices, test_indices


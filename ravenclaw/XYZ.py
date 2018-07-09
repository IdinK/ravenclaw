from slytherin.collections import has_duplicates, get_duplicates
from gobbledegook.string import camel_to_snake
import pandas as pd

class XYZ:
	def __init__(self, data, **kwargs):
		"""
		:type data: pd.DataFrame
		:type xyz: dict
		:param xyz: a dictionary of column groups
		"""
		self._data = data
		self._xyz = {}
		for group_name, columns in kwargs.items():
			self._digest(group_name=group_name, columns=columns)

	def _digest(self, group_name, columns):
		"""
		:type group_name: str
		:type columns: str or list of str
		"""
		if type(columns) is not str:
			if has_duplicates(list(columns)):
				raise ValueError('columns has duplicates:', get_duplicates(list(columns)))

		group_name = camel_to_snake(group_name)
		self._xyz[group_name] = columns
		setattr(self, group_name, self._data[columns])





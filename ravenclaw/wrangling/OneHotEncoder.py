import numpy as np
import pandas as pd
from slytherin.progress import ProgressBar

class OneHotEncoder:
	def __init__(
			self, top=10, rank_method='first', encode_na=False, replacement='other',
			include=None, exclude=None
	):
		self._column_values = {}
		self._top = top
		self._method = rank_method
		self._encode_na = encode_na
		self._replacement = replacement
		self._include = include
		self._exclude = exclude
		self._one_hot_columns = None


	def train(self, data, echo=0):
		echo = max(0, echo)
		result = data.copy()
		one_hot_columns = []
		non_numeric_cols = data.select_dtypes(exclude=['bool', 'number']).columns

		if self._include is not None:
			non_numeric_cols = [col for col in non_numeric_cols if col in self._include]
		if self._exclude is not None:
			non_numeric_cols = [col for col in non_numeric_cols if col not in self._exclude]
		progress_bar = ProgressBar(total=len(non_numeric_cols))
		progress_amount = 0
		for col_name in non_numeric_cols:
			if echo: progress_bar.show(amount=progress_amount, text=f'DM training dummies for {col_name}')
			progress_amount += 1
			try:
				temp_data = data[[col_name]].copy()
				temp_data['count'] = temp_data.groupby(col_name)[col_name].transform('count')
				temp_data['rank'] = temp_data.groupby('count')['count'].rank(method=self._method, ascending=False)
				only_include = set(temp_data[temp_data['rank']>self._top][col_name].values)

				temp_data[col_name] = np.where(
					temp_data[col_name].isin(only_include), temp_data[col_name], self._replacement
				)
				dummies = pd.get_dummies(
					data=temp_data[[col_name]],
					prefix=col_name, prefix_sep='_', dummy_na=self._encode_na, sparse=True
				)

				result = pd.concat([result, dummies], axis=1)
				#result[f'{col_name}_rank'] = temp_data['rank']
				one_hot_columns += list(dummies.columns)
				self._column_values[col_name] = only_include
			except:
				continue
		self._one_hot_columns = one_hot_columns
		if echo: progress_bar.show(amount=progress_amount, text=f'DM trained dummies for {one_hot_columns}')
		return result

	def encode(self, data, echo=0):
		echo = max(0, echo)
		result = data.copy()
		progress_bar = ProgressBar(total=len(self._column_values))
		progress_amount = 0
		for col_name, only_include in self._column_values.items():
			if echo: progress_bar.show(amount=progress_amount, text=f'DM creating dummies for {col_name}')
			progress_amount += 1
			temp_data = result[[col_name]].copy()
			temp_data[col_name] = np.where(
				temp_data[col_name].isin(only_include), temp_data[col_name], self._replacement
			)
			dummies = pd.get_dummies(
				data=temp_data[[col_name]],
				prefix=col_name, prefix_sep='_', dummy_na=self._encode_na, sparse=True
			)
			result = pd.concat([result, dummies], axis=1)
		for col_name in self._one_hot_columns:
			if col_name not in result.columns:
				result[col_name] = 0
		if echo: progress_bar.show(amount=progress_amount, text=f'DM created dummies for {self._one_hot_columns}')
		return result


from numpy import where

class Suppressor:
	def __init__(self, quantiles = (0.05, 0.95)):
		self._trained = False

		self._min = min(quantiles)
		self._max = max(quantiles)

		self._upper_limits = None
		self._lower_limits = None

		self._q1s = None
		self._cols = None
		self._type = None

	@property
	def trained(self):
		return self._trained

	@property
	def upper_limits(self):
		return self._upper_limits

	@property
	def lower_limits(self):
		return self._lower_limits

	@property
	def cols(self):
		return self._cols

	def suppress(self, data, by = 'column', only_cols = None, except_cols = [], inplace = False, echo = True):
		# by = 'column' : normalize each column separately
		# by = anything else : normalize the whole dataframe with the same mean and std
		if inplace:
			new_data = data
		else:
			new_data = data.copy()

		if not self._trained:
			if echo:
				print('training the suppressor')
			self._upper_limits = new_data.quantile(self._max)
			self._lower_limits = new_data.quantile(self._min)


			self._cols = [col for col in self._upper_limits.index if col not in except_cols]
			if only_cols is not None:
				self._cols = [col for col in self._cols if col in only_cols]

			if by not in ['col', 'column', 'columns']:
				raise ValueError("this type is not implemented yet")
				self._type = 'by dataframe'
				self._stds = new_data[self.cols].stack().std()
				self._means = new_data[self.cols].stack().mean()
			else:
				self._type = 'by column'

			self._trained = True

		if echo:
			suppressed_cols = []
		for column_name in self.cols:
			try:
				if self._type == 'by column':
					the_upper_limit = self._upper_limits[column_name]
					the_lower_limit = self._lower_limits[column_name]
					new_data[column_name] = where(
						new_data[column_name] < the_lower_limit, the_lower_limit,
						where(new_data[column_name] > the_upper_limit, the_upper_limit,
							  new_data[column_name]
						)
					)
				else:
					raise ValueError("this type is not implemented yet")
					the_upper_limit = self._stds
					if the_upper_limit == 0:
						the_upper_limit = 1
					new_data[column_name] = (new_data[column_name] - self._means) / the_upper_limit
				if echo:
					suppressed_cols.append(str(column_name))
			except:
				if echo: print('Warning! Could not suppress', column_name)

		if echo:
			print('suppressing:', ', '.join(suppressed_cols))
		return new_data
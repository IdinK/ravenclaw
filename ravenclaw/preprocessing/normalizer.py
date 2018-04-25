



class Normalizer:
	def __init__(self, by = 'column', only_cols = None, except_cols = []):
		self._trained = False
		self._stds = None
		self._means = None
		self._only_cols = only_cols
		self._except_cols = except_cols
		self._cols = None
		self._by = by

	@property
	def trained(self):
		return self._trained

	@property
	def stds(self):
		return self._stds

	@property
	def means(self):
		return self._means

	@property
	def cols(self):
	    return self._cols

	def normalize(self, data, inplace = False, echo = True):
		# by = 'column' : normalize each column separately
		# by = anything else : normalize the whole dataframe with the same mean and std
		by = self._by
		only_cols = self._only_cols
		except_cols = self._except_cols

		if inplace:
			new_data = data
		else:
			new_data = data.copy()

		if not self._trained:
			if echo:
				print('training the normalizer')
			self._stds = new_data.std()
			self._means = new_data.mean()


			self._cols = [col for col in self._stds.index if col not in except_cols]
			if only_cols is not None:
				self._cols = [col for col in self._cols if col in only_cols]

			if by not in ['col', 'column', 'columns']:
				self._by = 'by dataframe'
				self._stds = new_data[self.cols].stack().std()
				self._means = new_data[self.cols].stack().mean()
			else:
				self._by = 'by column'

			self._trained = True

		if echo:
			normalized_cols = []
		for column_name in self.cols:
			try:
				if self._by == 'by column':
					the_std = self._stds[column_name]
					if the_std == 0:
						the_std = 1
					new_data[column_name] = (new_data[column_name] - self._means[column_name])/the_std
				else:
					the_std = self._stds
					if the_std == 0:
						the_std = 1
					new_data[column_name] = (new_data[column_name] - self._means) / the_std
				if echo:
					normalized_cols.append(str(column_name))
			except:
				if echo: print('Warning! Could not normalize', column_name)

		if echo:
			print('normalizing:', ', '.join(normalized_cols))
		return new_data


	def denormalize(self, data, inplace = False, echo=True):
		if inplace:
			new_data = data
		else:
			new_data = data.copy()

		if not self._trained:
			raise ValueError('model is not trained!')

		if echo:
			normalized_cols = []
		for column_name in self.cols:
			try:
				if self._by == 'by column':
					the_std = self._stds[column_name]
					if the_std == 0:
						the_std = 1
					new_data[column_name] = new_data[column_name]*the_std + self._means[column_name]
				else:
					the_std = self._stds
					if the_std == 0:
						the_std = 1
					new_data[column_name] = new_data[column_name]*the_std + self._means
				if echo:
					normalized_cols.append(str(column_name))
			except:
				if echo: print('Warning! Could not normalize', column_name)

		if echo:
			print('denormalizing:', ', '.join(normalized_cols))
		return new_data


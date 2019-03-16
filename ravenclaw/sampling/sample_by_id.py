from pandas import DataFrame

def sample_by_id(data, id_column, n=None, fraction=None, replace=False, random_state=None, **kwargs):
	"""
	this method samples the data, by considering rows with the same id as one and refusing to divide them
	it is useful when you want a sample of the data based on ids when the ids are repeated
	and you want to take all or none of each id, but not a partial sample of multiple rows with the same id
	:param DataFrame data: dataframe to be sampled
	:param str id_column: column to be used as unique identifier
	:param int n: number of items to return
	:param float fraction: fraction of items to return, cannot be used with `n`
	:param bool replace: with or without replacement
	:param int random_state: seed for the random number generator
	:rtype: DataFrame
	"""
	data = data.copy()
	data['__index__'] = data.index
	ids = data[[id_column]].drop_duplicates()
	sampled_ids = ids.sample(n=n, frac=fraction, replace=replace, random_state=random_state, **kwargs)
	result = sampled_ids.merge(right=data, on=id_column, how='left').set_index('__index__')
	result.index.name = data.index.name
	return result
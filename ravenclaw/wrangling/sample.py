
def nrows(data):
	return data.shape[0]

def ncols(data):
	return data.shape[1]

# shuffles data and adds a column with new index
def shuffle(data, col = None, inplace = False):
	result = data.sample(frac = 1).reset_index(drop = True)
	if col is not None:
		result[col] = result.index

	if inplace:
		data = result
	return result


# sample the data but make sure to have a balanced number of samples per each group
# the groups are defined by columns
def sample(data, n, group_by = []):

	# group the data
	grouped = data.groupby(by = group_by)

	# shuffle the data inside each group
	shuffled1 = grouped.apply(lambda x: shuffle(x, col = '__order1')).reset_index(drop = True)

	# group by the new rank and shuffle the data inside each rank group
	shuffled2 = shuffled1.groupby('__order1').apply(lambda x: shuffle(x)).reset_index(drop = True)

	# remove the __order1 column
	shuffled2.drop(['__order1'], axis = 1, inplace = True)

	return shuffled2.head(n)


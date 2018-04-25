
from pandas import concat

# shuffles data and adds a column with new index
def shuffle(data, col = None):
	result = data.sample(frac = 1).reset_index(drop = True)
	if col is not None:
		result[col] = result.index

	return result


# sample the data but make sure to have a balanced number of samples per each group
# the groups are defined by columns
def sample_per_group(data, group_by, ratio):

	# group the data
	data = data.copy()
	data['__order1'] = data.index
	grouped = data.groupby(by = group_by)


	training_sets = []
	test_sets = []

	for name, group in grouped:
		n = round(group.shape[0] * ratio)
		shuffled = shuffle(data=group).reset_index(drop=True)
		training_sets.append(shuffled.iloc[:n,])
		test_sets.append(shuffled.iloc[(n+1):,])

	training = concat(training_sets)
	test = concat(test_sets)

	training.reset_index(drop=True, inplace=True)
	test.reset_index(drop=True, inplace=True)

	training.index = training['__order1'].values
	test.index = test['__order1'].values
	training.sort_index(inplace=True)
	test.sort_index(inplace=True)

	training.drop(axis=1, labels = '__order1', inplace=True)
	test.drop(axis=1, labels='__order1', inplace=True)


	return training, test


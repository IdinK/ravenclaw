from pandas import DataFrame


def join_wisely(left, right, **kwargs):
	"""
	joins two dataframes and returns a dictionary with 3 members: left_only, right_only, and both (the results of the two joins)
	:type left: DataFrame
	:type right: DataFrame
	:type kwargs: dict
	:rtype: dict of DataFrames
	"""
	left=left.copy()
	right=right.copy()

	left['_left_id'] = range(left.shape[0])
	right['_right_id'] = range(right.shape[0])

	full_join = left.merge(right=right, how='outer', indicator=True, **kwargs)
	"""
	:type full_join: DataFrame
	"""

	split_result = {group:data.drop(labels='_merge', axis=1) for group, data in full_join.groupby(by='_merge')}

	if 'left_only' in split_result:
		left_only_data = split_result['left_only']
		left_only_data = left[left._left_id.isin(left_only_data._left_id)]
		split_result['left_only'] = left_only_data.drop(labels='_left_id', axis=1)

	if 'right_only' in split_result:
		right_only_data = split_result['right_only']
		right_only_data = right[right._right_id.isin(right_only_data._right_id)]
		split_result['right_only'] = right_only_data.drop(labels='_right_id', axis=1)

	if 'both' in split_result:
		split_result['both'] = split_result['both'].drop(labels=['_left_id', '_right_id'], axis=1)

	return split_result
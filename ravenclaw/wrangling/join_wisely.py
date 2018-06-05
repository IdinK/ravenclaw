from pandas import DataFrame


def join_wisely(left, right, **kwargs):
	"""
	joins two dataframes and returns a dictionary with 3 members: left_only, right_only, and both (the results of the two joins)
	:type left: DataFrame
	:type right: DataFrame
	:type kwargs: dict
	:rtype: dict of DataFrames
	"""

	full_join = left.merge(right=right, how='outer', indicator=True, **kwargs)
	"""
	:type full_join: DataFrame
	"""

	split_result = {group:data.drop(labels='_merge', axis=1) for group, data in full_join.groupby(by='_merge')}
	return split_result
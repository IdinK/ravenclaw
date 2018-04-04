

def select_extreme(data, col, group_by, extreme_type = 'max'):
	if extreme_type=='max':
		return data.groupby(by = group_by).apply(lambda x: x[x.index == x[col].idxmax()])
	else:
		return data.groupby(by = group_by).apply(lambda x: x[x.index == x[col].idxmin()])

def select_max(data, max_col, group_by):
	return select_extreme(data = data, col = max_col, group_by = group_by, extreme_type = 'max')

def select_min(data, min_col, group_by):
	return select_extreme(data = data, col = min_col, group_by = group_by, extreme_type = 'min')
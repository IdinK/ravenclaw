from pandas import concat

def convert_data_to_Xy(data=None, x_cols=None, y_col=None, X=None, y=None):
	if data is not None and x_cols is not None and y_col is not None:  # data, x_cols, y_col
		pass
	elif data is not None and x_cols is not None:  # data, x_cols --> y_col
		_all_cols = list(data.columns)
		_y_cols = [_y for _y in _all_cols if _y not in x_cols]
		y_col = _y_cols[0]
	elif data is not None and y_col is not None:  # data, y_col
		_all_cols = list(data.columns)
		x_cols = [_x for _x in _all_cols if _x != y_col]
	elif X is not None and y is not None:  # # X, y
		x_cols = list(X.columns)
		try:
			y_col = y.name
		except:
			y_col = list(y.columns)[0]
		data = concat(objs=[X, y], axis=1)
	else:
		raise SyntaxError('missing arguments!')

	return {'data': data, 'x_cols':x_cols, 'y_col':y_col, 'X':X, 'y':y}
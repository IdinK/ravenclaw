from lonelypy import remove_non_alphanumeric

def remove_non_alphanumeric_lower_and_join(x, replace_with = '_', join_by = '__', ignore_errors = False):

	if type(x) == tuple:
		x = list(x)

	# remove empty strings and remove non alphanumeric
	if type(x) == list:
		#x = [remove_non_alphanumeric(s, replace_with = replace_with) for s in x if s != '']
		x = [str(s).strip() for s in x if s != '']
		x = join_by.join(x)


	try:
		x = remove_non_alphanumeric(s=x, replace_with=' ')
		x = x.strip()
		x = remove_non_alphanumeric(s=x, replace_with=replace_with)
		x = x.lower()
	except Exception as e:
		if ignore_errors:
			print('Error was ignored for: "', x, '" ', e, sep='')
		else:
			raise e

	return x


def standardize_columns(data, inplace = False):
	if inplace:
		new_data = data
	else:
		new_data = data.copy()

	new_data.columns = list(map(
		lambda x: remove_non_alphanumeric_lower_and_join(x = x),
		list(new_data.columns)
	))

	return new_data


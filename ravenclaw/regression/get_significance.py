
from statsmodels.api import add_constant, OLS
from pandas import DataFrame
from slytherin.numbers import beautify_num

def get_significance(X, y):
	X2 = add_constant(X)
	est = OLS(endog=y, exog=X2)
	est2 = est.fit()
	return est2

def summary_to_dataframe(results):
	'''This takes the result of an statsmodel results table and transforms it into a dataframe'''
	pvals = results.pvalues
	coeff = results.params
	conf_lower = results.conf_int()[0]
	conf_higher = results.conf_int()[1]

	results_df = DataFrame({
		"p_value":pvals,
		"coefficient":coeff,
		"confidence_lower":conf_lower,
		"confidence_upper":conf_higher
	})
	results_df.index.name = 'feature'
	results_df.reset_index(inplace=True)
	return results_df

def backward_eliminate(X, y, pvalue_limit = 0.05, echo=True, only_echo_eliminated=True):
	x_copy = X.copy()
	significances = summary_to_dataframe(get_significance(X=x_copy, y=y))
	max_p_value = significances['p_value'].max()
	significances.sort_values(ascending=False, by='p_value', inplace=True)
	all_features = list(significances['feature'])
	if echo and not only_echo_eliminated:
		print(beautify_num(max_p_value), all_features)

	while max_p_value>pvalue_limit:
		x_copy.drop(axis=1, labels=all_features[0], inplace=True)
		if echo:
			print('eliminating:', all_features[0])

		significances = summary_to_dataframe(get_significance(X=x_copy, y=y))
		max_p_value = significances['p_value'].max()
		significances.sort_values(ascending=False, by='p_value', inplace=True)
		all_features = list(significances['feature'])
		if echo and not only_echo_eliminated:
			print(beautify_num(max_p_value), all_features)

	if echo:
		print(significances)

	return all_features

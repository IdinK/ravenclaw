from slytherin.numbers import beautify_num
from pandas import DataFrame
from ..preprocessing.polynomialtransformer import PolynomialTransformer
from numpy import where

class Coefficient:
	def __init__(self, name, value, normalizer_mean = 0, normalizer_std = 1, normalized = True):
		self.name = name
		self.value = value
		self.normalizer_mean = normalizer_mean
		self.normalizer_std = normalizer_std
		self.normalized = normalized

	def __eq__(self, other):
		return abs(self.value) == abs(other.value)

	def __ne__(self, other):
		return not self.__eq__(other)

	def __le__(self, other):
		return abs(self.value) <= abs(other.value)

	def __gt__(self, other):
		return not self.__le__(other)

	def __ge__(self, other):
		return abs(self.value) >= abs(other.value)

	def __lt__(self, other):
		return not self.__ge__(other)

	def __repr__(self):
		return f"<{self.name}: {beautify_num(self.value)}>"



def get_coefficients(data, model, polynomial = None, normalizer = None, as_dataframe = True):
	coef_values = list(model.coef_)
	if type(polynomial) is PolynomialTransformer:
		coef_names = list(polynomial.get_feature_names())
	elif polynomial is None:
		coef_names = list(data.columns)
	else:
		coef_names = list(polynomial.get_feature_names(input_features=data.columns))

	if normalizer is None:
		normalizer_mean = 0
		normalizer_std = 1
		normalized = False
	else:
		normalizer_mean = None
		normalizer_std = None
		normalized = None

	result = []
	for coef_name, coef_value in zip(coef_names, coef_values):
		if normalizer is not None:
			try:
				normalizer_mean = normalizer.normalizers[coef_name].mean
				normalizer_std = normalizer.normalizers[coef_name].std
				normalized = normalizer.normalizers[coef_name].proper
			except:
				print(coef_name, 'not among', normalizer.normalizers.keys())

		result.append(Coefficient(name=coef_name, value=coef_value, normalizer_mean = normalizer_mean, normalizer_std = normalizer_std, normalized = normalized))
	result.append(Coefficient(name='intercept', value=model.intercept_, normalizer_mean=0, normalizer_std = 1, normalized = False))
	result.sort(reverse=True)
	result = [coef for coef in result if coef.value!=0]

	if as_dataframe:
		result = DataFrame({
			'coefficient':[coef.name for coef in result], 'value':[coef.value for coef in result],
			'normalizer_mean':[coef.normalizer_mean for coef in result],
			'normalizer_std':[coef.normalizer_std for coef in result],
			'normalized':[coef.normalized for coef in result]
		})
		result['actual_value'] = where(result['normalized']==True, result['value']/result['normalizer_std'], result['value'])

	return result

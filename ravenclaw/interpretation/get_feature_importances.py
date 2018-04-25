from pandas import DataFrame
from ..preprocessing.polynomialtransformer import PolynomialTransformer
from slytherin.numbers import beautify_num

class Feature:
	def __init__(self, name, importance):
		self.name = name
		self.importance = importance

	def __eq__(self, other):
		return (self.importance) == (other.importance)

	def __ne__(self, other):
		return not self.__eq__(other)

	def __le__(self, other):
		return (self.importance) <= (other.importance)

	def __gt__(self, other):
		return not self.__le__(other)

	def __ge__(self, other):
		return (self.importance) >= (other.importance)

	def __lt__(self, other):
		return not self.__ge__(other)

	def __repr__(self):
		return f"<{self.name}: {beautify_num(self.importance)}>"

def get_feature_importances(data, model, polynomial = None, as_dataframe = True):
	importance_values = list(model.feature_importances_)
	if type(polynomial) is PolynomialTransformer:
		feature_names = list(polynomial.get_feature_names())
	elif polynomial is None:
		feature_names = list(data.columns)
	else:
		feature_names = list(polynomial.get_feature_names(input_features=data.columns))

	result = []

	for feature_name, importance_value in zip(feature_names, importance_values):

		result.append(Feature(name=feature_name, importance=importance_value))

	result.sort(reverse=True)
	#result = [feature for feature in result if feature.importance != 0]

	if as_dataframe:
		result = DataFrame({
			'feature':[feature.name for feature in result],
			'importance':[feature.importance for feature in result]
		})

	return result
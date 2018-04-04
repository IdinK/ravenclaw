class Coefficient:
	def __init__(self, name, value):
		self.name = name
		self.value = value

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
		return f"{self.name}:{self.value}"



def get_coefficients(data, polynomial, model):
	coef_values = list(model.coef_)
	coef_names = list(polynomial.get_feature_names(input_features=data.columns))
	result = [Coefficient(name=coef_name, value=coef_value) for coef_name, coef_value in zip(coef_names, coef_values)]
	result.sort(reverse=True)
	return result

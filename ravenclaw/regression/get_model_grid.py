from slytherin.collections import get_dict_product

def get_model_grid(model, params, name=''):
	param_combinations = get_dict_product(params)
	if len(params)==0:
		sep = ''
	else:
		sep = '_'
	result = [
		{
			'name':name+sep+'_'.join([f"{param_name}:{param_value}" for param_name, param_value in param_set.items()]),
			'model':model(**param_set)
		}
		for param_set in param_combinations
	]
	return result
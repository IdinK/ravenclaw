from collections import OrderedDict

class Container(object):
	def __init__(self, dict = OrderedDict()):
		self.__dict__ = dict

	def get(self, key):
		return self.__dict__[key]

	def keys(self):
		return self.__dict__.keys()

	@property
	def dict(self):
		return self.__dict__

	def rename(self, key, new_key):
		if type(self.dict) is OrderedDict:
			self.__dict__ = OrderedDict((new_key if k==key else k, v) for k, v in self.dict.items())
		else:
			self.dict[new_key] = self.dict.pop(key)
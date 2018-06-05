from sklearn.metrics import accuracy_score, precision_score, average_precision_score, roc_curve
from slytherin.numbers import beautify_num
from .ConfusionMatrix import ConfusionMatrix
from ..prediction import Classifier

class ClassificationPerformance:
	def __init__(self, classifier):
		"""

		:type classifier: Classifier
		"""
		if not classifier._trained: raise SyntaxError('cannot measure the performance of an untrained classifier!')
		if not classifier._tested: raise SyntaxError('cannot measure the performance of an untested classifier!')
		y_true = classifier._y_true
		y_pred = classifier._y_pred
		y_score = classifier._y_pred_prob
		self._confusion_matrix = ConfusionMatrix(classifier=classifier)
		self._accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
		self._precision = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
		try:
			self._average_precision = average_precision_score(y_true=y_true, y_score=y_score, average='micro')
		except:
			self._average_precision = None

		try:
			self._roc_curve = roc_curve(y_true=y_true, y_score=y_score)
		except:
			self._roc_curve = None

	@property
	def confusion_matrix(self):
		return self._confusion_matrix

	@property
	def accuracy(self):
		return self._accuracy

	@property
	def precision(self):
		return self._precision

	@property
	def average_precision(self):
		return self._average_precision

	@property
	def roc_curve(self):
		return self._roc_curve


	def __repr__(self):
		return (
			f"accuracy: {beautify_num(self.accuracy)} ,"
			f"precision: {beautify_num(self.precision)}"
		)
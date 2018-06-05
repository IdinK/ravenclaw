
from slytherin.time import Timer
from pandas import DataFrame
from .Predictor import Predictor
from ..interpretation.ClassificationPerformance import ClassificationPerformance
from ..preprocessing.polynomialtransformer import PolynomialTransformer
from ..preprocessing.normalizer import Normalizer

class Classifier(Predictor):
	def __init__(
			self, predictive_model, name='classifier',
			polynomial_model = PolynomialTransformer(degree=1), normalizer = Normalizer()
	):
		super().__init__(predictive_model=predictive_model, name=name, polynomial_model=polynomial_model, normalizer=normalizer)
		self._y_pred_prob = None
		self._y_pred_prob_df = None


	def predict_probability(self, data, echo=False):
		timer = Timer()
		x_test_poly = self.preprocess(data=data, echo=echo)
		self._y_pred_prob = self.model.predict_proba(X=x_test_poly)
		self._y_pred_prob_df = DataFrame(self._y_pred_prob, columns=self.model.classes_)
		self._prediction_time = timer.get_elapsed()
		return self._y_pred_prob

	def test(self, data, y_true_col=None, y_true=None, echo=False):
		super().test(data=data, y_true_col=y_true_col, y_true=y_true, echo=echo)
		self.predict_probability(data=data, echo=echo)
		return self._y_pred

	@property
	def performance(self):
		if self._performance is None:
			self._performance = ClassificationPerformance(classifier=self)
		return self._performance

	def get_summary_row(self):
		if not self._tested:
			raise SyntaxError('you cannot get the summary of an untested model!')
		if not self._trained:
			raise SyntaxError('you cannot get the summary of an untrained model!')
		return DataFrame(data={
			'name': self.name,
			'num_features':self.num_features,
			'training_size':self.training_size,
			'training_time':self.training_time,
			'test_size':self.test_size,
			'test_time':self.prediction_time,
			'accuracy':self.performance.accuracy,
			'precision':self.performance.precision
		}, index=[0])

class MulticlassClassifier(Classifier):
	pass

class BinaryClassifier(Classifier):
	def get_summary_row(self):
		return DataFrame(data={
			'name': self.name,
			'num_features': self.num_features,
			'training_size': self.training_size,
			'training_time': self.training_time,
			'test_size': self.test_size,
			'test_time': self.prediction_time,
			'accuracy': self.performance.accuracy,
			'precision': self.performance.precision
		}, index=[0])

from pandas import DataFrame
from .Predictor import Predictor
from ..preprocessing.polynomialtransformer import PolynomialTransformer
from ..preprocessing.normalizer import Normalizer
from ..interpretation import RegressionPerformance, ClassificationPerformance

class Regressor(Predictor):
	def __init__(
			self, predictive_model, name='regressor',
			polynomial_model=PolynomialTransformer(degree=1), normalizer=Normalizer()
	):
		super().__init__(predictive_model=predictive_model, name=name, polynomial_model=polynomial_model, normalizer=normalizer)

	@property
	def performance(self):
		if self._performance is None:
			self._performance = RegressionPerformance(regressor=self)
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
			'mape':self.performance.mape,
			'rmse':self.performance.rmse,
			'nrmse_mean':self.performance.nrmse_mean,
			'nrmse_range':self.performance.nrmse_range
		}, index=[0])

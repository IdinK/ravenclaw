from pandas import crosstab

class ConfusionMatrix:
	def __init__(self, classifier, margins=True):
		self._confusion_df = crosstab(
			classifier._y_true, columns=classifier._y_pred,
			rownames=['True'], colnames=['Predicted'],
			margins=margins
		)

	def __repr__(self):
		return self._confusion_df

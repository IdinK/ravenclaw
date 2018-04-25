import numpy as np
from slytherin.numbers import beautify_num

def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() ** .5

class RegressionPerformance:
    def __init__(self, y_pred, y_true):
        self._rmse = get_rmse(y_true=y_true, y_pred=y_pred)
        self._mape = get_mape(y_true=y_true, y_pred=y_pred)
        self._nrmse = get_rmse(y_true=y_true, y_pred=y_pred)/y_true.mean()

    @property
    def rmse(self):
        return self._rmse

    @property
    def mape(self):
        return self._mape

    @property
    def nrmse(self):
        return self._nrmse

    def __repr__(self):
        return f"RMSE: {beautify_num(self.rmse)} ,  MAPE: {beautify_num(self.mape)}% ,  nRMSE: {beautify_num(self.nrmse)}"



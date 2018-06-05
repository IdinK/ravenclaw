import numpy as np
from slytherin.numbers import beautify_num
from ..prediction import Regressor

def get_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def get_rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean() ** .5

def get_nrmse(y_true, y_pred, normalize_by = 'mean'):
    if normalize_by=='mean':
        return get_rmse(y_true=y_true, y_pred=y_pred)/y_true.mean()
    else: # normalize_by=='range'
        return get_rmse(y_true=y_true, y_pred=y_pred)/(y_true.max() - y_true.min())


class RegressionPerformance:
    def __init__(self, regressor):
        """

        :type regressor: Regressor
        """
        if not regressor._trained: raise SyntaxError('cannot measure the performance of an untrained regressor!')
        if not regressor._tested: raise SyntaxError('cannot measure the performance of an untested regressor!')
        y_true = regressor._y_true
        y_pred = regressor._y_pred
        self._rmse = get_rmse(y_true=y_true, y_pred=y_pred)
        self._mape = get_mape(y_true=y_true, y_pred=y_pred)
        self._nrmse_mean = get_nrmse(y_true=y_true, y_pred=y_pred, normalize_by='mean')
        self._nrmse_range = get_nrmse(y_true=y_true, y_pred=y_pred, normalize_by='range')

    @property
    def rmse(self):
        return self._rmse

    @property
    def mape(self):
        return self._mape

    @property
    def nrmse_mean(self):
        return self._nrmse_mean

    @property
    def nrmse_range(self):
        return self._nrmse_range

    def __repr__(self):
        return (
            f"RMSE: {beautify_num(self.rmse)} ,"
            f"MAPE: {beautify_num(self.mape)}% ,  "
            f"nRMSE (mean): {beautify_num(self.nrmse)} , "
            f"nRMSE (range): {beautify_num(self.nrmse_range)}"
        )


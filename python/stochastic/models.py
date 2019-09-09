from sklearn import tree, linear_model, neural_network, svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import statsmodels.regression.quantile_regression as smrq

import numpy as np
import pandas as pd

import pytups.superdict as sd

import stochastic.model_upper_bound as mub
import stochastic.graphs as graphs
import stochastic.tools as aux

####################
# Forecasting
####################

def create_X_Y(result_tab, x_vars, y_var, test_perc):
    X = result_tab[x_vars].copy()
    Y = result_tab[y_var].copy()
    return \
        train_test_split(X, Y, test_size=test_perc, random_state=1)


def test_regression(result_tab, x_vars, y_var, method, bound='', plot_args=None, test_perc=0.3, plot=True, **kwargs):
    x_vars = sorted(x_vars)
    X_train, X_test, y_train, y_test = create_X_Y(result_tab, x_vars, y_var, test_perc)
    X_train_norm = aux.normalize_variables(X_train)
    mean_std = aux.get_mean_std(X_train)
    clf = predict_factory(X_train=X_train_norm, y_train=y_train, **kwargs, method=method)
    X_test_norm = aux.normalize_variables(X_test, mean_std)
    y_pred = clf.predict(X_test_norm)
    args = (y_test, y_pred)
    above = (y_pred > y_test).sum() / y_pred.shape[0] * 100
    below = (y_pred < y_test).sum() / y_pred.shape[0] * 100
    print('MAE={}'.format(metrics.mean_absolute_error(*args)))
    print('MSE={}'.format(metrics.mean_squared_error(*args)))
    print('R^2={}'.format(metrics.r2_score(*args)))
    print("above: {}%".format(round(above, 2)))
    print("below: {}%".format(round(below, 2)))
    if not plot:
        return clf, mean_std
    X_all_norm = aux.normalize_variables(result_tab[x_vars], mean_std)
    y_pred = clf.predict(X_all_norm)
    X = result_tab.copy()
    X['pred'] = y_pred
    graph_name = '{}_mean_consum_{}_{}'.format(method, bound, y_var)
    if plot_args is not None:
        _args = dict(x='mean_consum', y=y_var, y_pred='pred', graph_name=graph_name, smooth=False)
        _args.update(plot_args)
        graphs.plotting(X, **_args)
    else:
        graphs.plotting(X, x='mean_consum', y=y_var, y_pred='pred', graph_name=graph_name, smooth=False)
    return clf, mean_std


def predict_factory(X_train, y_train, method='regression', **kwargs):
    if method=='regression':
        clf = linear_model.LinearRegression(**kwargs)
    elif method == 'trees':
        clf = tree.DecisionTreeRegressor(**kwargs)
    elif method == 'neural':
        clf = neural_network.MLPRegressor(**kwargs)
    elif method == 'GBR':
        clf = GradientBoostingRegressor(**kwargs)
    elif method == 'SVR':
        clf = svm.SVR(**kwargs)
    elif method == 'QuantReg':
        mod = smrq.QuantReg(endog=y_train, exog=X_train)
        res = mod.fit(**kwargs)
        return res
    elif method == 'superquantiles':
        coef0, coefs = mub.regression_VaR(X=X_train, Y=y_train, **kwargs)

        class Clf(object):

            def __init__(self, coefs, coef0):
                self.coefs = coefs
                self.coef0 = coef0
                self.params = dict(coefs)
                self.params['intercept'] = coef0
                self.params = pd.Series(self.params)

            def predict(self, new_x_data):
                return np.sum([v * new_x_data[k] for k, v in self.coefs.items()], axis=0) + self.coef0

        return Clf(coefs, coef0)
    else:
        raise ValueError('method argument has no correct value.')
    clf.fit(X=X_train, y=y_train)
    return clf


if __name__ == '__main__':
    pass

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
import stochastic.auxiliary as aux

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
    else:
        raise ValueError('method argument has no correct value.')
    clf.fit(X=X_train, y=y_train)
    return clf


def test_superquantiles(result_tab, x_vars, predict_var, plot=True, upper_bound=True, _print=True, **kwargs):

    result_tab_f =  result_tab[x_vars + [predict_var]].copy()
    bound = 'upper'
    if not upper_bound:
        result_tab_f[predict_var] *= -1
        bound = 'lower'
    result_tab_norm = aux.normalize_variables(result_tab_f)
    mean_std = aux.get_mean_std(result_tab_f)
    X_norm = result_tab_norm[x_vars].copy()
    Y_norm = result_tab_norm[predict_var].copy()
    # X_norm = normalize_variables(X)
    # mean_std = get_mean_std(X)
    # Y_norm = normalize_variables(Y)
    # mean_std_y = get_mean_std(Y)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, Y_norm, test_size=0.3)
    coef0, coefs = mub.regression_VaR(X=X_train, Y=y_train, **kwargs)
    if not upper_bound:
        coef0 *= -1
        coefs = sd.SuperDict.from_dict(coefs).vapply(lambda v: -v)
    coefs_sans_inter = sd.SuperDict(coefs)
    coefs.update({'intercept': coef0})
    if not _print:
        return coefs
    X_out_norm = X_test.copy()
    X_out = aux.denormalize(X_out_norm, mean_std)
    y_test_dn = aux.denormalize(pd.DataFrame({predict_var:y_test}), mean_std)[predict_var]
    y_test_dn.reset_index(drop=True, inplace=True)
    if not upper_bound:
        y_test_dn *= -1
    X_out[predict_var] = y_test_dn
    y_pred = np.sum([v*X_out_norm[k] for k, v in coefs_sans_inter.items()], axis=0) + coef0
    y_pred = aux.denormalize(pd.DataFrame({predict_var:y_pred}), mean_std)[predict_var]
    y_pred.reset_index(drop=True, inplace=True)
    X_out['pred'] = y_pred
    above = (y_pred > y_test_dn).sum() / y_pred.shape[0] * 100
    print("above: {}%".format(round(above, 2)))
    if not plot:
        return coefs
    X_out = X_out.join(result_tab, rsuffix='_other')
    # print(X_out)
    graph_name = 'superquantiles_mean_consum_init_{}_{}'.format(predict_var, bound)
    graphs.plotting(table=X_out, x='mean_consum', y=predict_var, y_pred='pred',
                    graph_name=graph_name, smooth=False, color='status',
                    shape='has_special', facet='init_cut ~ var_consum_cut')

    return coefs, mean_std


    # return (X_norm * X.std()) + X.mean()

if __name__ == '__main__':
    pass

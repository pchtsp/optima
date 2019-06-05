from sklearn import tree
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import pytups.superdict as sd

import stochastic.model_upper_bound as mub
import stochastic.graphs as graphs

####################
# Forecasting
####################

def create_X_Y(result_tab, x_vars, y_var):
    X = result_tab[x_vars].copy()
    Y = result_tab[y_var].copy()
    # 70% training and 30% test
    return \
        train_test_split(X, Y, test_size=0.3, random_state=1)


def decision_tree(result_tab, x_vars, y_var):
    X_train, X_test, y_train, y_test = create_X_Y(result_tab, x_vars, y_var)
    clf = tree.DecisionTreeRegressor(max_depth=10)
    clf.fit(X=X_train, y=y_train)
    y_pred = clf.predict(X_test)
    metrics.mean_absolute_error(y_test, y_pred)
    metrics.mean_squared_error(y_test, y_pred)


def regression(result_tab, x_vars, y_var):
    X_train, X_test, y_train, y_test = create_X_Y(result_tab, x_vars, y_var)
    clf = linear_model.LinearRegression()
    clf.fit(X=X_train, y=y_train)
    y_pred = clf.predict(X_test)
    args = (y_test, y_pred)
    print('MAE={}'.format(metrics.mean_absolute_error(*args)))
    print('MSE={}'.format(metrics.mean_squared_error(*args)))
    print('R^2={}'.format(metrics.r2_score(*args)))
    return (clf.coef_, clf.intercept_)


def test_regression(result_tab, x_vars, plot=True):
    predict_var= 'cycle_2M_min'
    filter = np.all([~pd.isna(result_tab[predict_var]),
                     result_tab.mean_consum.between(150, 300)],
                    axis=0)
    coefs, intercept = regression(result_tab[filter], x_vars=x_vars, y_var=predict_var)
    coef_dict = sd.SuperDict(zip(x_vars, coefs))
    if not plot:
        return coef_dict, intercept
    y_pred = np.sum([result_tab[k]*c for k, c in coef_dict.items()], axis=0) + intercept
    X = result_tab.copy()
    X['pred'] = y_pred
    graph_name = 'regression_mean_consum_g{}_init_{}'.format(5, predict_var)
    graphs.plotting(X, 'mean_consum', predict_var, 'pred', graph_name)
    return coef_dict, intercept


def classify(result_tab, x_vars, y_var):
    X_train, X_test, y_train, y_test = create_X_Y(result_tab, x_vars, y_var)
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf.fit(X=X_train, y=y_train, sample_weight=y_train*10 +1)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return metrics.confusion_matrix(y_test, y_pred)


def regression_superquantiles(result_tab, x_vars, predict_var, plot=True, upper_bound=True, **kwargs):

    X = result_tab[x_vars].copy()
    Y = result_tab[predict_var].copy()
    bound = 'upper'
    if not upper_bound:
        Y *= -1
        bound = 'lower'
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    coef0, coefs = mub.regression_VaR(X=X_train, Y=y_train, **kwargs)
    if not upper_bound:
        coef0 *= -1
        coefs = sd.SuperDict.from_dict(coefs).vapply(lambda v: -v)
    print(coefs)

    X_out = X_test.copy()
    if not upper_bound:
        y_test *= -1
    X_out[predict_var] = y_test
    X_out['pred'] = y_pred = np.sum([v*X_out[k] for k, v in coefs.items()], axis=0) + coef0
    above = (y_pred > y_test).sum() / y_pred.shape[0] * 100
    print("above: {}%".format(round(above, 2)))

    if not plot:
        return
    X_out = X_out.join(result_tab[['init_cut', 'name', 'status']])
    graph_name = 'superquantiles_mean_consum_init_{}_{}'.format(predict_var, bound)
    graphs.plotting(X_out, 'mean_consum', predict_var, 'pred', graph_name)


if __name__ == '__main__':
    result_tab = None
    var_coefs = {}
    for predict_var in ['maints', 'mean_2maint', 'mean_dist']:
        for (bound, sign, alpha) in [('max', 1, 0.8), ('min', -1, 0.99)]:
            x_train = result_tab[['mean_consum', 'init']].copy()
            x_train['mean_consum2'] = x_train.mean_consum**2
            x_train['mean_consum3'] = x_train.mean_consum**3
            y_train = result_tab[predict_var]
            coef0, coefs = mub.regression_VaR(x_train, y_train, _lambda=0.1, alpha=alpha, sign=sign)
            coefs['intercept'] = coef0
            var_coefs[bound+'_'+predict_var] = coefs


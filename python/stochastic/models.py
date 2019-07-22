from sklearn import tree, linear_model, neural_network
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

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


def test_non_regression(result_tab, x_vars, y_var, method, bound='', **kwargs):

    X_train, X_test, y_train, y_test = create_X_Y(result_tab, x_vars, y_var)
    clf = predict_factory(X_train=X_train, y_train=y_train, **kwargs, method=method)

    y_pred = clf.predict(X_test)
    args = (y_test, y_pred)
    above = (y_pred > y_test).sum() / y_pred.shape[0] * 100
    below = (y_pred < y_test).sum() / y_pred.shape[0] * 100
    print('MAE={}'.format(metrics.mean_absolute_error(*args)))
    print('MSE={}'.format(metrics.mean_squared_error(*args)))
    print('R^2={}'.format(metrics.r2_score(*args)))
    print('R^2={}'.format(metrics.r2_score(*args)))
    print("above: {}%".format(round(above, 2)))
    print("below: {}%".format(round(below, 2)))

    y_pred = clf.predict(result_tab[x_vars])
    X = result_tab.copy()
    X['pred'] = y_pred
    graph_name = '{}_mean_consum_{}_{}'.format(method, bound, y_var)
    graphs.plotting(X, x='mean_consum', y=y_var, y_pred='pred', graph_name=graph_name, smooth=False)
    return clf


def predict_factory(X_train, y_train, method='regression', **kwargs):
    if method=='regression':
        clf = linear_model.LinearRegression(**kwargs)
    elif method == 'trees':
        clf = tree.DecisionTreeRegressor(**kwargs)
    elif method=='neural':
        clf = neural_network.MLPRegressor(**kwargs)
    elif method=='GBR':
        clf = GradientBoostingRegressor(**kwargs)
    elif method=='QuantReg':
        import statsmodels.regression.quantile_regression as smrq
        mod = smrq.QuantReg(endog=y_train, exog=X_train)
        res = mod.fit(**kwargs)
        return res
    else:
        raise ValueError('method argument has no correct value.')
    clf.fit(X=X_train, y=y_train)
    return clf


def test_regression(result_tab, x_vars, y_var, plot=True):
    X_train, X_test, y_train, y_test = create_X_Y(result_tab, x_vars, y_var)
    clf = predict_factory(result_tab, X_train=X_train, y_train=y_train)
    coefs, intercept = clf.coef_, clf.intercept_
    coef_dict = sd.SuperDict(zip(x_vars, coefs))
    coef_dict_sans_int = sd.SuperDict(coef_dict)
    coef_dict['intercept'] = intercept
    if not plot:
        return coef_dict
    y_pred = np.sum([result_tab[k]*c for k, c in coef_dict_sans_int.items()], axis=0) + intercept
    X = result_tab.copy()
    X['pred'] = y_pred
    graph_name = 'regression_mean_consum_g{}_{}'.format(5, y_var)
    graphs.plotting(X, x='mean_consum', y=y_var, y_pred='pred', graph_name=graph_name, smooth=False)
    return coef_dict


def classify(result_tab, x_vars, y_var):
    X_train, X_test, y_train, y_test = create_X_Y(result_tab, x_vars, y_var)
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf.fit(X=X_train, y=y_train, sample_weight=y_train*10 +1)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return metrics.confusion_matrix(y_test, y_pred)


def test_superquantiles(result_tab, x_vars, predict_var, plot=True, upper_bound=True, _print=True, **kwargs):

    result_tab_f =  result_tab[x_vars + [predict_var]].copy()
    bound = 'upper'
    if not upper_bound:
        result_tab_f[predict_var] *= -1
        bound = 'lower'
    result_tab_norm = normalize_variables(result_tab_f)
    mean_std = get_mean_std(result_tab_f)
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
    X_out = denormalize(X_out_norm, mean_std)
    y_test_dn = denormalize(pd.DataFrame({predict_var:y_test}), mean_std)[predict_var]
    y_test_dn.reset_index(drop=True, inplace=True)
    if not upper_bound:
        y_test_dn *= -1
    X_out[predict_var] = y_test_dn
    y_pred = np.sum([v*X_out_norm[k] for k, v in coefs_sans_inter.items()], axis=0) + coef0
    y_pred = denormalize(pd.DataFrame({predict_var:y_pred}), mean_std)[predict_var]
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


def normalize_variables(X):
    return (X - X.mean())/X.std()


def get_mean_std(X):
    stds = X.std().to_dict()
    means = X.mean().to_dict()
    return {k: {'std': v, 'mean': means[k]} for k, v in stds.items()}


def denormalize(X_norm, mean_std):
    keys = list(X_norm.columns)
    mean_std_f = sd.SuperDict.from_dict(mean_std).filter(keys)
    tab = pd.DataFrame.from_dict(mean_std_f, orient='index')
    return ((X_norm * tab['std']) + tab['mean'])

    # return (X_norm * X.std()) + X.mean()

if __name__ == '__main__':
    pass

import package.experiment as exp
import package.params as params
import package.data_input as di
import stochastic.model_upper_bound as mub

import stochastic.instance_stats as istats
import stochastic.solution_stats as sol_stats
import stochastic.graphs as graphs
import stochastic.params as sto_params

import pytups.superdict as sd
import pytups.tuplist as tl

import os
import numpy as np
import shutil
import orloge as ol
import pandas as pd

from sklearn import tree
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split

# from importlib import reload

#####################
# EXPORT THINGS
####################

# rep.print_table_md(result_tab_summ)
# result_tab.to_csv(path_graphs + 'table.csv', index=False)
# rpyg.gantt_experiment(e+'/')



def clean_remakes():
    path_remake = params.PATHS['results'] + 'dell_20190515_remakes'
    ll = di.experiment_to_instances(path_remake)
    lle = sd.SuperDict(ll).to_dictup().vapply(exp.Experiment.from_dir).\
        clean(func= lambda v: v is None).keys_l()
    lled = tl.TupList(lle).filter_list_f(lambda x: x[1]!='index.txt').\
        apply(lambda x: os.path.join(path_remake, x[0], x[1]))

    lled.apply(shutil.rmtree)

def write_index():
    _filter = np.all([status_df.sol_code==ol.LpSolutionIntegerFeasible,
                     status_df.gap_abs >= 50], axis=0)
    remakes_df = status_df[_filter].sort_values(['gap_abs'], ascending=False)

    remakes = remakes_df.name.tolist()
    remakes_path = r'C:\Users\franco.peschiera.fr\Documents\optima_results\dell_20190515_remakes/base/index.txt'

    with open(remakes_path, 'w') as f:
        f.write('\n'.join(remakes))


def instance_status(experiments, vars_extract):
    log_paths = sd.SuperDict({os.path.basename(e): os.path.join(e, 'results.log')
                              for e in experiments})
    ll = \
        log_paths.\
        clean(func=os.path.exists).\
        vapply(lambda v: ol.get_info_solver(v, 'CPLEX', get_progress=False)). \
        vapply(lambda x: {var: x[var] for var in vars_extract})

    master = \
        pd.DataFrame({'sol_code': [ol.LpSolutionIntegerFeasible, ol.LpSolutionOptimal,
                                   ol.LpSolutionInfeasible, ol.LpSolutionNoSolutionFound],
                      'status': ['IntegerFeasible', 'Optimal', 'Infeasible', 'NoIntegerFound']})
    return \
        pd.DataFrame.\
        from_dict(ll, orient='index').\
        rename_axis('name').\
        reset_index().merge(master, on='sol_code')

def get_table(experiments):
    result = []
    cases = [exp.Experiment.from_dir(e) for e in experiments]
    basenames = [os.path.basename(e) for e in experiments]

    for p, e in enumerate(experiments):
        # print(e)
        case = cases[p]
        # we clean errors.
        if case is None:
            continue
        consumption = istats.get_consumptions(case.instance)
        aircraft_use = istats.get_consumptions(case.instance, hours=False)
        rel_consumption = istats.get_rel_consumptions(case.instance)
        cycle_2M_size = sol_stats.get_1M_2M_dist(case)
        cycle_1M_size = sol_stats.get_prev_1M_dist(case)
        cycle_2M_quants = cycle_2M_size.quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
        cycle_1M_size_values = cycle_1M_size.values_l()
        cycle_1M_quants = pd.Series(cycle_1M_size_values).quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
        l_maint_date = sol_stats.get_last_maint_date(case).values_l()
        init_hours = istats.get_init_hours(case.instance)
        cy_sum = cycle_2M_size.agg(['mean', 'max', 'min']).tolist()
        airc_sum = aircraft_use.agg(['mean', 'max', 'var']).tolist()
        cons_sum = consumption.agg(['mean', 'max', 'var']).tolist()
        l_maint_date_stat = pd.Series(l_maint_date).agg(['mean', 'max', 'min']).tolist()
        pos_consum = [istats.get_argmedian(consumption, prop) for prop in [0.5, 0.75, 0.9]]
        pos_aircraft = [istats.get_argmedian(aircraft_use, prop) for prop in [0.5, 0.75, 0.9]]
        geomean_airc = istats.get_geomean(aircraft_use)
        geomean_cons = istats.get_geomean(consumption)
        quantsw = rel_consumption.rolling(12).mean().shift(-11).quantile(q=[0.5, 0.75, 0.9]).tolist()
        init_sum = init_hours.agg(['mean']).tolist()
        num_maints = [sol_stats.get_num_maints(case)]
        cons_min_assign = istats.min_assign_consumption(case.instance).agg(['mean', 'max']).tolist()
        num_special_tasks = istats.get_num_special(case.instance)
        num_errors = 0
        errors = di.load_data(e + '/errors.json')
        _case_name = basenames[p]
        if errors:
            num_errors = sd.SuperDict(errors).to_dictup().len()
        result.append([_case_name] + init_sum +
                      cons_sum +
                      airc_sum +
                      num_maints +
                      cons_min_assign + quantsw +
                      pos_consum +
                      pos_aircraft +
                      [geomean_cons, geomean_airc] + [num_errors] +
                      [num_special_tasks] +
                      cy_sum +
                      cycle_2M_quants +
                      cycle_1M_quants +
                      l_maint_date_stat)

    names = ['name', 'init',
             'mean_consum', 'max_consum', 'var_consum',
             'mean_airc', 'max_airc', 'var_airc',
             'maints',
             'cons_min_mean', 'cons_min_max', 'quant5w', 'quant75w', 'quant9w',
             'pos_consum5', 'pos_consum75', 'pos_consum9',
             'pos_aircraft5', 'pos_aircraft75', 'pos_aircraft9',
             'geomean_cons', 'geomean_airc', 'num_errors',
             'spec_tasks',
             'mean_dist', 'max_dist', 'min_dist',
             'cycle_2M_min', 'cycle_2M_25', 'cycle_2M_50', 'cycle_2M_75', 'cycle_2M_max',
             'cycle_1M_min', 'cycle_1M_25', 'cycle_1M_50', 'cycle_1M_75', 'cycle_1M_max',
             'mean_2maint',  'max_2maint', 'min_2maint']

    renames = {p: n for p, n in enumerate(names)}
    result_tab = pd.DataFrame(result).rename(columns=renames)
    # result_tab = result_tab[result_tab.num_errors==0]

    result_tab['mean_consum_cut'] = pd.qcut(result_tab.mean_consum, q=10).astype(str)
    indeces = pd.DataFrame({'mean_consum_cut': sorted(result_tab['mean_consum_cut'].unique())}).\
        reset_index().rename(columns={'index': 'mean_consum_cut_2'})
    result_tab = result_tab.merge(indeces, on='mean_consum_cut')

    for col in ['init', 'quant75w', 'quant9w', 'quant5w',
                'var_consum', 'cons_min_mean', 'pos_consum5', 'pos_consum9', 'spec_tasks']:
        result_tab[col+'_cut'] = pd.qcut(result_tab[col], q=3, duplicates='drop').astype(str)

    for grade in range(2, 6):
        result_tab['mean_consum' + str(grade)] = result_tab.mean_consum ** grade

    status_df = get_status_df(experiments)
    result_tab = result_tab.merge(status_df, on=['name'], how='left')

    return result_tab

def get_status_df(experiments):
    status_df = instance_status(experiments, ['sol_code', 'status_code', 'time', 'gap', 'best_bound', 'best_solution'])
    status_df['gap_abs'] = status_df.best_solution - status_df.best_bound
    # status_df[status_df.sol_code==ol.LpSolutionInfeasible]
    # status_df[status_df.gap_abs > 100]
    # status_df.columns
    return status_df



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
    X = X.merge(status_df, on='name', how='left')
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

#####################
# check instances that do not fit
#####################
def test5():
    few_maints = result_tab.query('maints <= 27 & mean_consum >= 170').copy()
    few_maints['freq_maints'] = 'few'
    many_maints = result_tab.query('maints == 31 & mean_consum <= 150').copy()
    many_maints['freq_maints'] = 'many'
    weird_maints = pd.concat([few_maints, many_maints])

    experiments_few_mains = [os.path.basename(experiments[p]) for p in few_maints.index]
    experiments_many_maints = [os.path.basename(experiments[p]) for p in many_maints.index]

    result_tab_t = result_tab.join(weird_maints[['freq_maints']])
    result_tab_t.freq_maints.fillna('other', inplace=True)
    result_tab_t.init_cut = result_tab_t.init_cut.astype(str).str[1]
    result_tab_t.groupby(['freq_maints', 'init_cut']).agg('mean').stack().unstack(0)


    e = experiments_few_mains[0]
    case = exp.Experiment.from_dir(e)
    len(case.get_maintenance_starts())


if __name__ == '__main__':
    os.environ['path'] += r';C:\Program Files (x86)\Graphviz2.38\bin'
    name = sto_params.name
    path = params.PATHS['results'] + name +'/base/'

    experiments = [os.path.join(path, i) for i in os.listdir(path)]
    basenames = [os.path.basename(e) for e in experiments]

    result_tab = get_table(experiments)
    status_df = get_status_df(experiments)
    status_df.agg('mean')[['gap_abs', 'time', 'best_solution']]
    status_df.groupby('status').agg('count')['name']
    status_df.groupby('status').agg('max')['gap_abs']
    status_df.groupby('status').agg('max')['gap']
    status_df.groupby('status').agg('median')['gap_abs']
    status_df.groupby('status').agg('median')['gap']

    result_tab.loc[result_tab.num_errors == 0, 'has_errors'] = 'no errors'
    result_tab.loc[result_tab.num_errors > 0, 'has_errors'] = '>=1 errors'
    (result_tab.num_errors==0).sum()
    graphs.cons_init(result_tab, var='cycle_2M_min', color='status', smooth=False)

    for var in ['maints', 'mean_dist', 'max_dist', 'min_dist', 'mean_2maint']:
        graphs.draw_hist(var)

    graphs.draw_hist('mean_2maint')

    # cases = [exp.Experiment.from_dir(e) for e in experiments]
    # hist_no_agg(basenames, cases)

    for var in ['maints', 'mean_dist', 'mean_2maint', 'cycle_2M_min']:
        graphs.cons_init(result_tab, var, color='status')

    result_tab_summ = \
        result_tab.groupby(['mean_consum_cut_2', 'init_cut']). \
            agg({'mean_dist': ['mean', 'var'],
                 'maints': ['mean', 'var']})

    # regression(result_tab, ['mean_consum', 'init', 'pos_consum5'], 'mean_dist')
    # regression(result_tab, ['mean_consum', 'init', 'geomean_cons'], 'maints')
    x_vars = ['mean_consum', 'mean_consum2', 'mean_consum3', 'init', 'pos_consum9',
              'pos_consum5', 'quant5w', 'quant75w', 'quant9w', 'max_consum', 'var_consum',
              'cons_min_mean', 'cons_min_max']
    x_vars += ['spec_tasks']
    # x_vars += ['geomean_cons']
    predict_var = 'cycle_2M_min'

    bound_options = [(True, 0.7), (False, 0.5)]
    bound_options = [(False, 0.5)]
    table = result_tab
    # table = result_tab.query('mean_consum >=180 and gap_abs <= 30')
    # table = result_tab[result_tab.mean_consum.between(150, 300)]
    table = result_tab.query('mean_consum >=150 and gap_abs <= 80')
    for upper, alpha in bound_options:
        regression_superquantiles(table, x_vars=x_vars,
                                  predict_var=predict_var,
                                  _lambda=10, alpha=alpha, plot=True, upper_bound=upper)
    data = test_regression(result_tab, x_vars, plot=False)
    data = test_regression(result_tab, x_vars)
    (result_tab.num_errors>0).sum()
    # 'mean_consum', 'init', 'mean_consum_2', 'mean_consum_3'

    for _var in ['maints', 'mean_2maint', 'mean_dist']:
        for grade in range(3, 4):
            for (alpha, sign) in zip([0.99, 0.8], [-1, 1]):
                print(_var, grade)
                regression_superquantiles(result_tab, status_df, _var, _lambda=0.1, alpha=alpha, sign=sign)
                print()


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


# result_tab.groupby('spec_tasks_cut').name.agg('count')
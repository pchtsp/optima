import package.experiment as exp
import package.params as params
import data.data_input as di

import stochastic.instance_stats as istats
import stochastic.solution_stats as sol_stats
import stochastic.graphs as graphs
import stochastic.params as sto_params
import stochastic.models as models

import pytups.superdict as sd
import pytups.tuplist as tl
import zipfile

import os
import numpy as np
import shutil
import orloge as ol
import pandas as pd


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

def write_index(status_df, remakes_path):
    # sum(_filter)
    # sum(status_df.sol_code==ol.LpSolutionOptimal)
    not_optimal_status = [ol.LpSolutionIntegerFeasible, ol.LpSolutionNoSolutionFound]
    _filter = np.in1d(status_df.sol_code, not_optimal_status)
    # _filter = np.all([status_df.sol_code==ol.LpSolutionIntegerFeasible,
    #                  status_df.gap_abs >= 10], axis=0)
    # _filter = np.any([status_df.sol_code==ol.LpSolutionNoSolutionFound, _filter], axis=0)
    remakes_df = status_df[_filter].sort_values(['gap_abs'], ascending=False)
    status_df.groupby('sol_code').agg('count')
    remakes = remakes_df.name.tolist()
    with open(remakes_path, 'w') as f:
        f.write('\n'.join(remakes))


def get_table(cases, _types=1):
    result = []
    # cases = [exp.Experiment.from_dir(e) for e in experiments]
    # basenames = [os.path.basename(e) for e in experiments]

    for _type in range(_types):
        for p, (_case_name, case) in enumerate(cases.items()):
            # we clean errors.
            if case is None:
                continue
            # instance dtata:
            consumption = istats.get_consumptions(case.instance, _type=_type)
            aircraft_use = istats.get_consumptions(case.instance, hours=False, _type=_type)
            rel_consumption = istats.get_rel_consumptions(case.instance, _type=_type)
            init_hours = istats.get_init_hours(case.instance, _type=_type)
            airc_sum = aircraft_use.agg(['mean', 'max', 'var']).tolist()
            cons_sum = consumption.agg(['mean', 'max', 'var']).tolist()
            pos_consum = [istats.get_argmedian(consumption, prop) for prop in [0.5, 0.75, 0.9]]
            pos_aircraft = [istats.get_argmedian(aircraft_use, prop) for prop in [0.5, 0.75, 0.9]]
            geomean_airc = istats.get_geomean(aircraft_use)
            geomean_cons = istats.get_geomean(consumption)
            cons_min_assign = istats.min_assign_consumption(case.instance, _type=_type).agg(['mean', 'max']).tolist()
            num_special_tasks = istats.get_num_special(case.instance, _type=_type)
            init_sum = init_hours.agg(['mean']).tolist()
            quantsw = rel_consumption.rolling(12).mean().shift(-11).quantile(q=[0.5, 0.75, 0.9]).tolist()

            # solution dtata:
            cycle_2M_size = sol_stats.get_1M_2M_dist(case, _type=_type)
            cycle_2M_size_complete = sol_stats.get_1M_2M_dist(case, _type=_type, count_1M=True)
            cycle_1M_size = sol_stats.get_prev_1M_dist(case, _type=_type)
            cycle_2M_quants = cycle_2M_size.quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
            cycle_1M_size_values = cycle_1M_size.values_l()
            cycle_1M_quants = pd.Series(cycle_1M_size_values).quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
            cycle_2M_acum = cycle_2M_size.agg(['mean']).tolist()
            cycle_2M_complete_acum = cycle_2M_size_complete.agg(['mean', 'max', 'min']).tolist()
            l_maint_date = sol_stats.get_post_2M_dist(case, _type=_type).values_l()
            l_maint_date_stat = pd.Series(l_maint_date).agg(['mean', 'max', 'min', 'sum']).tolist()
            num_maints = sol_stats.get_num_maints(case, _type=_type)
            result.append([_case_name] + init_sum +
                          cons_sum +
                          airc_sum +
                          cons_min_assign + quantsw +
                          pos_consum +
                          pos_aircraft +
                          [geomean_cons, geomean_airc] +
                          [num_special_tasks] +

                          [num_maints]  +
                          cycle_2M_acum +
                          cycle_2M_complete_acum +
                          cycle_2M_quants +
                          cycle_1M_quants +
                          l_maint_date_stat)

    names = ['name', 'init',
             'mean_consum', 'max_consum', 'var_consum',
             'mean_airc', 'max_airc', 'var_airc',
             'cons_min_mean', 'cons_min_max', 'quant5w', 'quant75w', 'quant9w',
             'pos_consum5', 'pos_consum75', 'pos_consum9',
             'pos_aircraft5', 'pos_aircraft75', 'pos_aircraft9',
             'geomean_cons', 'geomean_airc',
             'spec_tasks'] + \
             ['maints',
             'mean_dist',
             'mean_dist_complete', 'max_dist_complete', 'min_dist_complete',
             'cycle_2M_min', 'cycle_2M_25', 'cycle_2M_50', 'cycle_2M_75', 'cycle_2M_max',
             'cycle_1M_min', 'cycle_1M_25', 'cycle_1M_50', 'cycle_1M_75', 'cycle_1M_max',
             'mean_2maint',  'max_2maint', 'min_2maint', 'sum_2maint']

    renames = {p: n for p, n in enumerate(names)}
    result_tab = pd.DataFrame(result).rename(columns=renames)
    return result_tab
    # result_tab = result_tab[result_tab.num_errors==0]

def treat_table(result_tab):

    for col in ['init', 'quant75w', 'quant9w', 'quant5w', 'geomean_cons',
                'var_consum', 'cons_min_mean', 'pos_consum5', 'pos_consum9', 'spec_tasks', 'mean_consum']:
        result_tab[col+'_cut'] = pd.qcut(result_tab[col], q=3, duplicates='drop').astype(str)

    # we generate auxiliary variations on consumption.
    # we also scale them.

    for grade in range(2, 6):
        result_tab['mean_consum' + str(grade)] = \
            (result_tab.mean_consum ** grade)
    return result_tab

def get_status_df(logInfo):
    vars_extract = ['sol_code', 'status_code', 'time', 'gap', 'best_bound', 'best_solution']
    ll = logInfo.vapply(lambda x: {var: x[var] for var in vars_extract})

    master = \
        pd.DataFrame({'sol_code': [ol.LpSolutionIntegerFeasible, ol.LpSolutionOptimal,
                                   ol.LpSolutionInfeasible, ol.LpSolutionNoSolutionFound],
                      'status': ['IntegerFeasible', 'Optimal', 'Infeasible', 'NoIntegerFound']})
    status_df =\
        pd.DataFrame.\
        from_dict(ll, orient='index').\
        rename_axis('name').\
        reset_index().merge(master, on='sol_code')

    status_df['gap_abs'] = status_df.best_solution - status_df.best_bound
    return status_df

def get_cases(relpath):
    path = params.PATHS['results'] + relpath
    experiments = [os.path.join(path, i) for i in os.listdir(path)]
    basenames = [os.path.basename(e) for e in experiments]
    return {c: exp.Experiment.from_dir(e) for c, e in zip(basenames, experiments)}

def get_logs(relpath):
    path = params.PATHS['results'] + relpath
    experiments = [os.path.join(path, i) for i in os.listdir(path)]
    log_paths = sd.SuperDict({os.path.basename(e): os.path.join(e, 'results.log')
                              for e in experiments})
    return log_paths. \
            clean(func=os.path.exists). \
            vapply(lambda v: ol.get_info_solver(v, 'CPLEX', get_progress=False))

def get_cases_zip(name, relpath):
    path = params.PATHS['results'] + name + '.zip'
    zipobj = zipfile.ZipFile(path)
    experiments = [f[:-1] for f in zipobj.namelist() if f.startswith(relpath) and f.endswith('/') and f.count("/") == 3]
    basenames = [os.path.basename(e) for e in experiments]
    return {c: exp.Experiment.from_zipfile(zipobj, e) for c, e in zip(basenames, experiments)}

def get_logs_zip(name, relpath):
    path = params.PATHS['results'] + name + '.zip'
    zipobj = zipfile.ZipFile(path)
    experiments = [f[:-1] for f in zipobj.namelist() if f.startswith(relpath) and f.endswith('/') and f.count("/") == 3]
    log_paths = sd.SuperDict({os.path.basename(e): e + '/results.log' for e in experiments})

    def _read_zip(x):
        try:
            return zipobj.read(x)
        except:
            return 0

    return log_paths. \
            vapply(_read_zip). \
            clean(). \
            vapply(lambda x: str(x, 'utf-8')). \
            vapply(lambda v: ol.get_info_solver(v, 'CPLEX', get_progress=False, content=True))

def get_errors(relpath):
    path = params.PATHS['results'] + relpath
    experiments = tl.TupList(os.path.join(path, i) for i in os.listdir(path))
    base_exp = sd.SuperDict.from_dict({os.path.basename(e): e for e in experiments})
    load_data = di.load_data
    errors = \
        base_exp.\
            vapply(lambda v: v + '/errors.json').\
            vapply(load_data).\
            clean().\
            vapply(sd.SuperDict.from_dict).\
            vapply(lambda v: v.to_dictup()).\
            to_lendict()

    errors = errors.fill_with_default(base_exp)
    return errors

def get_errors_zip(name, relpath):
    path = params.PATHS['results'] + name + '.zip'
    zipobj = zipfile.ZipFile(path)
    experiments = [f[:-1] for f in zipobj.namelist() if f.startswith(relpath) and f.endswith('/') and f.count("/") == 3]
    base_exp = sd.SuperDict.from_dict({os.path.basename(e): e for e in experiments})
    load_data = lambda v: di.load_data_zip(zipobj=zipobj, path=v)
    errors = \
        base_exp. \
        vapply(lambda v: v + '/errors.json'). \
        vapply(load_data). \
        clean(). \
        vapply(sd.SuperDict.from_dict). \
        vapply(lambda v: v.to_dictup()). \
        to_lendict()
    errors = errors.fill_with_default(base_exp)
    return errors

# #########
# BOUNDS:
# #########
def superquantiles():
    # x_vars += ['pos_consum9', 'pos_consum5', 'quant5w', 'quant75w', 'quant9w']
    bound_options = [
        ('maints', True, 0.9),
        ('cycle_2M_min', False, 0.9),
        ('mean_dist', False, 0.9),
        ('mean_2maint', True, 0.9),
        ('sum_2maint', True, 0.9),
        ('sum_2maint', False, 0.9),
        ('mean_2maint', False, 0.9)
    ]
    table = result_tab
    # table = result_tab.query('mean_consum >=180 and gap_abs <= 30')
    # table = result_tab[result_tab.mean_consum.between(150, 300)]
    table = result_tab.query('mean_consum >=150 and mean_consum <=250')
    # table = result_tab.query('mean_consum >=150 and mean_consum <=250 and num_errors==0')

    x_vars = ['mean_consum', 'mean_consum2', 'mean_consum3', 'mean_consum4', 'init', 'max_consum', 'var_consum',
              'cons_min_mean', 'cons_min_max', 'spec_tasks']
    data = {}
    for predict_var, upper, alpha in bound_options:
        b = 'min_' + predict_var
        if upper:
            b = 'max_' + predict_var
        _dict, norm_info = models.test_superquantiles(
            table, x_vars=x_vars, predict_var=predict_var,
            _lambda=0.02, alpha=alpha, plot=True, upper_bound=upper)
        data[b] = sd.SuperDict.from_dict(_dict).clean()

# #########
# REGRESSIONS:
# #########
def regressions(table):
    x_vars = ['mean_consum', 'mean_consum2', 'init', 'max_consum',
              'var_consum', 'spec_tasks', 'geomean_cons']
    y_vars = ['cycle_2M_min', 'maints', 'mean_dist', 'mean_2maint', 'sum_2maint']
    method = 'regression'
    data = {}
    for predict_var in y_vars:
        coefs_df = models.test_regression(table, x_vars, plot=True,
                                          method=method, y_var=predict_var)
        data[predict_var] = coefs_df
# #########
# TREES:
# #########
def trees(table):
    method='trees'
    data = {}
    x_vars = ['mean_consum', 'mean_consum2', 'init', 'max_consum',
              'var_consum', 'spec_tasks', 'geomean_cons']
    y_vars = ['cycle_2M_min', 'maints', 'mean_dist', 'mean_2maint', 'sum_2maint']
    for predict_var in y_vars:
        coefs_df = models.test_regression(table, x_vars, method=method,
                                          y_var=predict_var, max_depth=4)
        data[predict_var] = coefs_df
# #########
# NN:
# #########
def neural_networks(table):
    x_vars = ['init', 'mean_consum', 'max_consum', 'var_consum', 'mean_airc',
       'max_airc', 'var_airc', 'maints', 'cons_min_mean', 'cons_min_max',
       'quant5w', 'quant75w', 'quant9w', 'pos_consum5', 'pos_consum75',
       'pos_consum9', 'pos_aircraft5', 'pos_aircraft75', 'pos_aircraft9',
       'geomean_cons', 'geomean_airc', 'spec_tasks', 'mean_consum2', 'mean_consum3',
       'mean_consum4', 'mean_consum5']

    # x_vars = ['mean_consum', 'init', 'var_consum', 'spec_tasks', 'geomean_cons']
    options = [
        dict(y_var='mean_dist_complete'),
        dict(y_var='maints'),
        dict(y_var='mean_2maint')
    ]

    default = dict(method = 'neural', plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init'))
    options = tl.TupList(options).apply(lambda v: {**default, **v})

    coefs_df = []
    for opt in options:
        coefs_df += [models.test_regression(table, x_vars, **opt)]


# #########
# GradientBoostRegression:
# #########
def gradient_boosting_regression(table, **kwargs):
    x_vars = ['init',
             'mean_consum', 'max_consum', 'var_consum',
             'mean_airc', 'max_airc', 'var_airc',
             'cons_min_mean', 'cons_min_max', 'quant5w', 'quant75w', 'quant9w',
             'pos_consum5', 'pos_consum75', 'pos_consum9',
             'pos_aircraft5', 'pos_aircraft75', 'pos_aircraft9',
             'geomean_cons', 'geomean_airc',
             'spec_tasks']
    x_vars = ['mean_consum', 'mean_consum2',
              'mean_consum3',
              'pos_consum75', 'pos_consum9',
              'init', 'max_consum',
              'var_consum', 'spec_tasks', 'geomean_cons']
    x_vars = \
        ['pos_aircraft75',
         'max_consum',
         'var_consum',
         'pos_consum5',
         'quant9w',
         'quant5w',
         'init',
         'geomean_airc',
         'geomean_cons',
         'mean_consum',
         'mean_airc',
         'cons_min_mean',
         'pos_consum75',
         'var_airc',
         'pos_consum9',
         'spec_tasks']
    x_vars = ['mean_consum', 'mean_consum2', 'mean_consum3', 'init', 'max_consum',
              'var_consum', 'spec_tasks', 'geomean_cons']
    x_vars = ['mean_consum', 'mean_consum2', 'mean_consum3', 'init', 'var_consum',
              'spec_tasks', 'geomean_cons']
    x_vars = ['mean_consum', 'init', 'var_consum', 'spec_tasks', 'geomean_cons']

    options = [
        dict(alpha=0.1, plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x = 'init'),
             bound='lower', y_var='mean_dist_complete'),
        dict(alpha=0.9, plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init'),
             bound='upper', y_var='mean_dist_complete'),
        dict(alpha=0.9, bound='upper', y_var='maints',
             plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init')),
        dict(alpha=0.9, bound='upper', y_var='mean_2maint',
             plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init')),
        dict(alpha=0.1, bound='lower', y_var='mean_2maint',
             plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init'))
    ]

    default = dict(method = 'GBR', loss = "quantile", n_estimators = 1000, **kwargs)
    options = tl.TupList(options).apply(lambda v: {**default, **v})

    coefs_df = []
    for opt in options:
        coefs_df += [models.test_regression(table, x_vars, **opt)]

    # for x, u in coefs_df:
    #     info = pd.DataFrame.from_records(
    #         zip(x_vars, x.feature_importances_)).\
    #         sort_values(1)[-8:]
    #     print(info)

    mean_std = \
        sd.SuperDict.from_dict(coefs_df[0][1]).to_dictup().to_tuplist().\
        to_dict(2, indices=[1, 0], is_list=False).to_dictdict()

    real_coefs = coefs_to_dict(options, coefs_df, get_params=False)
    real_coefs.update(mean_std)
    return real_coefs

def quantile_regressions(table, **kwargs):
    # sklearn.feature_selection.RFECV¶

    x_vars = ['mean_consum', 'mean_consum2', 'mean_consum3', 'mean_consum4', 'init', 'var_consum', 'spec_tasks', 'geomean_cons']
    # x_vars = ['mean_consum', 'init', 'var_consum', 'spec_tasks', 'geomean_cons']

    # options = [dict(q=0.1, bound='lower', plot_args=dict(facet=None), y_var='mean_dist'),
    #            dict(q=0.9, bound='upper', y_var='maints'),
    #            dict(q=0.1, bound='lower', plot_args=dict(facet=None),  y_var='cycle_2M_min'),
    #            dict(q=0.9, bound='upper',  y_var='mean_2maint'),
    #            dict(q=0.1, bound='lower',  y_var='mean_2maint')
    #            ]

    options = [
        dict(q=0.1, plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x = 'init'),
             bound='lower', y_var='mean_dist_complete'),
        dict(q=0.9, plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init'),
             bound='upper', y_var='mean_dist_complete'),
        dict(q=0.9, bound='upper', y_var='maints',
             plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init')),
        dict(q=0.9, bound='upper', y_var='mean_2maint',
             plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init')),
        dict(q=0.1, bound='lower', y_var='mean_2maint',
             plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init'))
    ]
    default = dict(method = 'QuantReg', max_iter = 10000, **kwargs)
    options = tl.TupList(options).apply(lambda v: {**default, **v})

    coefs_df = []
    for opt in options:
        coefs_df += [models.test_regression(table, x_vars, **opt)]

    # [print(c[0].params) for c in coefs_df]
    # coef = coefs_df[0]

    mean_std = \
        sd.SuperDict.from_dict(coefs_df[0][1]).to_dictup().to_tuplist().\
        to_dict(2, indices=[1, 0], is_list=False).to_dictdict()

    real_coefs = coefs_to_dict(options, coefs_df)
    real_coefs.update(mean_std)
    return real_coefs

def coefs_to_dict(options, coefs, get_params=True):
    values = {}
    for opt, coef in zip(options, coefs):
        bound = 'max'
        if opt['bound']=='lower':
            bound = 'min'
        name = bound + '_' + opt['y_var']
        if get_params:
            values[name] = coef[0].params.to_dict()
        else:
            values[name] = coef[0]
    return values

def support_vector_regression(table):
    x_vars = ['mean_consum', 'mean_consum2', 'mean_consum3', 'init', 'var_consum', 'spec_tasks', 'geomean_cons']

    options = [dict(plot_args=dict(facet=None), y_var='mean_dist'),
               dict(y_var='maints'),
               dict(plot_args=dict(facet=None), y_var='cycle_2M_min'),
               dict(y_var='mean_2maint')
               ]
    default = dict(method = 'SVR', bound = 'center')
    options = tl.TupList(options).apply(lambda v: {**v, **default})


    coefs_df = []
    for opt in options:
        coefs_df += [models.test_regression(table, x_vars, **opt)]


if __name__ == '__main__':

    def reload_all():
        from importlib import reload
        reload(sto_params)
        reload(params)
        reload(graphs)
        reload(models)
        reload(istats)

    name = sto_params.name
    use_zip = sto_params.use_zip
    relpath = name +'/base/'

    if use_zip:
        cases = get_cases_zip(name, relpath)
        log_info = get_logs_zip(name, relpath)
        errors = get_errors_zip(name, relpath)
    else:
        cases = get_cases(relpath)
        log_info = get_logs(relpath)
        errors = get_errors(relpath)

    result_tab_origin = get_table(cases, 1)
    result_tab = treat_table(result_tab_origin)
    status_df = get_status_df(log_info)
    result_tab = result_tab.merge(status_df, on=['name'], how='left')
    result_tab.loc[result_tab.spec_tasks > 0, 'has_special'] = 'yes'
    result_tab.loc[result_tab.spec_tasks == 0, 'has_special'] = 'no'
    result_tab['num_errors'] = result_tab.name.map(errors)
    result_tab.loc[result_tab.num_errors == 0, 'has_errors'] = 'no errors'
    result_tab.loc[result_tab.num_errors > 0, 'has_errors'] = '>=1 errors'

    # start = cases['201907160210_928'].instance.get_param('start')
    # _func = lambda v: len([st for st in v.get_maintenance_starts() if st[1]==start])>0
    # rrr = sd.SuperDict.from_dict(cases).clean(func=lambda v: v is not None).vapply(_func)
    # rrr.clean()
    # cases['201907160213_297'].get_maintenance_periods()
    # cases['201907160213_297'].get_all_maintenance_cycles()
    # raise ValueError('Stop!')

    # result_tab.best_solution
    # result_tab.best_bound
    result_tab.loc[result_tab.gap_abs <= 60, 'gap_stat'] = 'gap_abs<=60'
    result_tab.loc[result_tab.gap_abs > 60, 'gap_stat'] = 'gap_abs>=60'

    status_df = result_tab
    status_df.agg('mean')[['gap_abs', 'time', 'best_solution']]
    status_df.groupby('status').agg('count')['name']
    status_df.groupby('status').agg('max')['gap_abs']
    status_df.groupby('status').agg('max')['gap']
    status_df.groupby('status').agg('median')['gap_abs']
    status_df.groupby('status').agg('median')['gap']


    for var in ['maints', 'mean_dist', 'max_dist', 'min_dist', 'mean_2maint', 'mean_dist_complete']:
        graphs.draw_hist(result_tab, var, bar=False)

    graphs.draw_hist('mean_2maint')

    # #########
    # PLOTTING:
    # #########
    col = 'geomean_cons'
    result_tab[col + '_cut'] = pd.qcut(result_tab[col], q=3, duplicates='drop').astype(str)

    for y in ['maints', 'mean_dist', 'mean_2maint', 'cycle_2M_min', 'mean_dist_complete', 'min_dist_complete']:
        x = 'mean_consum'
        facet = 'init_cut ~ .'
        x = 'init'
        facet = 'mean_consum_cut ~ .'
        graph_name = '{}_vs_{}'.format(x, y)
        graphs.plotting(result_tab, x=x,  y=y, color='status',
                        facet=facet , graph_name=graph_name, smooth=True)

    x = 'mean_consum'
    # x= 'init'
    y = 'sum_2maint'
    graph_name = '{}_vs_{}_nofacet'.format(x, y)
    graphs.plotting(result_tab, x=x, y=y,
                    facet=None, graph_name=graph_name, smooth=True)
    for y in ['maints', 'mean_dist', 'mean_2maint', 'cycle_2M_min']:
        x = 'mean_consum'
        graph_name = '{}_vs_{}_nocolor'.format(x, y)
        graphs.plotting(result_tab, x=x, y=y, graph_name=graph_name, smooth=False)
    # result_tab.best_solution

    # #########
    # PREDICTING:
    # #########
    case = sd.SuperDict(cases).values_l()[100]
    # result_tab[x_vars].iloc[[100]]
    # coefs_df[1].predict(result_tab[x_vars].iloc[[100]])
    table = result_tab[(result_tab.gap_abs < 30) & (result_tab.num_errors == 0)]
    table = result_tab[(result_tab.num_errors == 0)]
    # test_perc= 0.1, plot=False
    real_coefs_gbr = gradient_boosting_regression(table, test_perc= 0.3)
    mean_std_gbr = sd.SuperDict.from_dict(real_coefs_gbr).filter(['mean', 'std'])
    istats.predict_stat(case.instance, real_coefs_gbr['max_maints'], _type=0, mean_std=mean_std_gbr)
    istats.predict_stat(case.instance, real_coefs_gbr['min_mean_dist_complete'], _type=0, mean_std=mean_std_gbr)
    istats.predict_stat(case.instance, real_coefs_gbr['max_mean_dist_complete'], _type=0, mean_std=mean_std_gbr)


    real_coefs_qr = quantile_regressions(table, test_perc= 0.1)
    mean_std_gr = sd.SuperDict.from_dict(real_coefs_qr).filter(['mean', 'std'])
    istats.calculate_stat(case.instance, real_coefs_qr['max_maints'], 0, mean_std=mean_std_gr)
    istats.calculate_stat(case.instance, real_coefs_qr['min_mean_dist_complete'], 0, mean_std=mean_std_gr)
    istats.calculate_stat(case.instance, real_coefs_qr['max_mean_dist_complete'], 0, mean_std=mean_std_gr)

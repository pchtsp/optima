import package.params as params
import package.batch as ba

import stochastic.instance_stats as istats
import stochastic.solution_stats as sol_stats
import stochastic.graphs as graphs
import stochastic.params as sto_params
import stochastic.models as models

import pytups.superdict as sd
import pytups.tuplist as tl

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
    batch = ba.ZipBatch(path_remake)
    cases = batch.get_cases()
    lle = cases.clean(func= lambda v: v is None).keys_l()
    lled = tl.TupList(lle).filter_list_f(lambda x: x[1]!='index.txt').\
        apply(lambda x: os.path.join(path_remake, x[0], x[1]))

    lled.apply(shutil.rmtree)

def write_index(status_df, remakes_path):
    not_optimal_status = [ol.LpSolutionIntegerFeasible, ol.LpSolutionNoSolutionFound]
    _filter = np.in1d(status_df.sol_code, not_optimal_status)

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
            result.append(list(_case_name) +
                          init_sum +
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

    names = ['scenario', 'name']+ \
            ['init',
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
                'var_consum', 'cons_min_mean', 'pos_consum5', 'pos_consum9',
                'spec_tasks', 'mean_consum', 'geomean_cons']:
        try:
            labels = tl.TupList(['1/3', '2/3', '3/3']).apply(lambda v: '{}: {}'.format(col, v))
            result_tab[col+'_cut'] = \
                pd.qcut(result_tab[col], q=3, duplicates='drop', labels=labels).\
                astype(str)
        except ValueError:
            labels = tl.TupList(['2/3', '3/3']).apply(lambda v: '{}: {}'.format(col, v))
            result_tab[col+'_cut'] = \
                pd.qcut(result_tab[col], q=3, duplicates='drop', labels=labels).\
                astype(str)
    # we generate auxiliary variations on consumption.
    # we also scale them.
    for grade in range(2, 6):
        result_tab['mean_consum' + str(grade)] = \
            (result_tab.mean_consum ** grade)

    result_tab.loc[result_tab.spec_tasks > 0, 'has_special'] = 'yes'
    result_tab.loc[result_tab.spec_tasks == 0, 'has_special'] = 'no'
    result_tab['num_errors'].fillna(0, inplace=True)
    result_tab.loc[result_tab.num_errors == 0, 'has_errors'] = 'no errors'
    result_tab.loc[result_tab.num_errors > 0, 'has_errors'] = '>=1 errors'
    result_tab.loc[result_tab.gap_abs <= 60, 'gap_stat'] = 'gap_abs<=60'
    result_tab.loc[result_tab.gap_abs > 60, 'gap_stat'] = 'gap_abs>=60'

    return result_tab


def superquantiles(table, **kwargs):
    x_vars = ['init',
             'mean_consum', 'max_consum', 'var_consum',
             'mean_airc', 'max_airc', 'var_airc',
             'cons_min_mean', 'cons_min_max', 'quant5w', 'quant75w', 'quant9w',
             'pos_consum5', 'pos_consum75', 'pos_consum9',
             'pos_aircraft5', 'pos_aircraft75', 'pos_aircraft9',
             'geomean_cons', 'geomean_airc',
             'spec_tasks'] + \
             ['mean_consum2', 'mean_consum3', 'mean_consum4', 'mean_consum5']
    # x_vars = ['mean_consum', 'mean_consum2', 'mean_consum3', 'mean_consum4', 'init', 'var_consum', 'spec_tasks', 'geomean_cons']

    options = [
        dict(alpha=0.1, plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x = 'init'),
             bound='lower', y_var='mean_dist_complete'),
        dict(alpha=0.9, plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init'),
             bound='upper', y_var='mean_dist_complete'),
        # dict(alpha=0.9, bound='upper', y_var='maints',
        #      plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init')),
        # dict(alpha=0.9, bound='upper', y_var='mean_2maint',
        #      plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init')),
        # dict(alpha=0.1, bound='lower', y_var='mean_2maint',
        #      plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init'))
    ]
    default = dict(method = 'superquantiles', **kwargs)
    options = tl.TupList(options).apply(lambda v: {**default, **v})

    coefs_df = []
    for opt in options:
        coefs_df += [models.test_regression(table, x_vars, **opt)]

    mean_std = \
        sd.SuperDict.from_dict(coefs_df[0][1]).to_dictup().to_tuplist().\
        to_dict(2, indices=[1, 0], is_list=False).to_dictdict()

    real_coefs = coefs_to_dict(options, coefs_df)
    real_coefs.update(mean_std)
    return real_coefs

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
       'mean_consum4', 'mean_consum5'] + ['mean_consum2', 'mean_consum3', 'mean_consum4']

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
    # sklearn.feature_selection.RFECV

    x_vars = ['mean_consum', 'mean_consum2', 'mean_consum3', 'mean_consum4', 'init', 'var_consum', 'spec_tasks', 'geomean_cons']

    options = [
        dict(q=0.1, plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x = 'init'),
             bound='lower', y_var='mean_dist_complete'),
        dict(q=0.9, plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init'),
             bound='upper', y_var='mean_dist_complete'),
        dict(q=0.9, bound='upper', y_var='maints',
             plot_args=dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init')),
        dict(q=0.1, bound='lower', y_var='maints',
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
    if 'intercept' not in real_coefs:
        real_coefs['intercept'] = 0
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

def get_batch():
    name = sto_params.name
    use_zip = sto_params.use_zip

    if use_zip:
        batch = ba.ZipBatch(params.PATHS['results'] + name)
    else:
        batch = ba.Batch(params.PATHS['results'] + name)

    return batch

def get_table_semi_treated(batch):
    cases = batch.get_cases()
    errors = batch.get_errors()
    errors = errors.to_tuplist().to_dict(2, indices=[1], is_list=False)
    status_df = batch.get_status_df()

    result_tab_origin = get_table(cases, 1)
    result_tab = result_tab_origin.merge(status_df, on=['scenario', 'name'], how='left')
    result_tab['num_errors'] = result_tab.name.map(errors)

    return result_tab

if __name__ == '__main__':

    def reload_all():
        from importlib import reload
        reload(sto_params)
        reload(params)
        reload(graphs)
        reload(models)
        reload(istats)
        reload(ba)

    batch = get_batch()
    result_tab = get_table_semi_treated(batch)
    result_tab = treat_table(result_tab)

    for var in ['maints', 'mean_dist', 'max_dist', 'min_dist', 'mean_2maint', 'mean_dist_complete']:
        graphs.draw_hist(result_tab, var, bar=False)

    graphs.draw_hist(result_tab, 'mean_dist_complete', bar=False)


    # #########
    # PLOTTING:
    # #########
    for y in ['maints', 'mean_dist', 'mean_2maint', 'cycle_2M_min', 'mean_dist_complete', 'min_dist_complete']:
        x = 'mean_consum'
        facet = 'init_cut ~ .'
        x = 'init'
        facet = 'mean_consum_cut ~ .'
        graph_name = '{}_vs_{}'.format(x, y)
        graphs.plotting(result_tab, x=x,  y=y, color='status',
                        facet=facet , graph_name=graph_name, smooth=True)
    # result_tab.columns
    x = 'init'
    # x= 'init_cut ~ .'
    y = 'mean_dist_complete'
    facet = 'mean_consum_cut ~ .'
    _args = dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init', y = 'mean_dist_complete')
    graph_name = '{}_vs_{}_nofacet'.format(x, y)
    graphs.plotting(result_tab, x=x, y=y, facet=facet, graph_name=graph_name, smooth=True)
    graphs.plotting(result_tab, graph_name=graph_name, **_args, smooth=True, color='status')

    for y in ['maints', 'mean_dist', 'mean_2maint', 'cycle_2M_min']:
        x = 'mean_consum'
        graph_name = '{}_vs_{}_nocolor'.format(x, y)
        graphs.plotting(result_tab, x=x, y=y, graph_name=graph_name, smooth=False)

    table = result_tab[result_tab.mean_consum_cut == result_tab.mean_consum_cut[0]]
    # result_tab.best_solution

    # #########
    # PREDICTING:
    # #########
    cases = batch.get_cases()
    case = sd.SuperDict(cases).values_l()[100]
    # result_tab[x_vars].iloc[[100]]
    # coefs_df[1].predict(result_tab[x_vars].iloc[[100]])
    table = result_tab[(result_tab.gap_abs < 100) & (result_tab.num_errors == 0)]
    table = result_tab[(result_tab.num_errors == 0)]
    # test_perc= 0.1, plot=False

    real_coefs_gbr = gradient_boosting_regression(table, test_perc= 0.3)
    mean_std_gbr = sd.SuperDict.from_dict(real_coefs_gbr).filter(['mean', 'std'])
    istats.predict_stat(case.instance, real_coefs_gbr['max_maints'], _type=0, mean_std=mean_std_gbr)
    istats.predict_stat(case.instance, real_coefs_gbr['min_mean_dist_complete'], _type=0, mean_std=mean_std_gbr)
    istats.predict_stat(case.instance, real_coefs_gbr['max_mean_dist_complete'], _type=0, mean_std=mean_std_gbr)


    real_coefs_qr = quantile_regressions(table, test_perc= 0.3)
    real_coefs_qr['max_mean_dist_complete']
    mean_std_gr = sd.SuperDict.from_dict(real_coefs_qr).filter(['mean', 'std'])
    istats.calculate_stat(case.instance, real_coefs_qr['max_maints'], 0, mean_std=mean_std_gr)
    istats.calculate_stat(case.instance, real_coefs_qr['min_mean_dist_complete'], 0, mean_std=mean_std_gr)
    istats.calculate_stat(case.instance, real_coefs_qr['max_mean_dist_complete'], 0, mean_std=mean_std_gr)


    real_coefs_sq = superquantiles(table, test_perc= 0.3, _lambda=0.0001)
    real_coefs_sq['max_mean_dist_complete']
    mean_std_sq = sd.SuperDict.from_dict(real_coefs_sq).filter(['mean', 'std'])
    istats.calculate_stat(case.instance, real_coefs_sq['max_maints'], 0, mean_std=mean_std_sq)
    istats.calculate_stat(case.instance, real_coefs_sq['min_mean_dist_complete'], 0, mean_std=mean_std_sq)
    istats.calculate_stat(case.instance, real_coefs_sq['max_mean_dist_complete'], 0, mean_std=mean_std_sq)
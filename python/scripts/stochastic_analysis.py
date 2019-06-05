import package.experiment as exp
import package.params as params
import package.data_input as di

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

def get_table(experiments, _types=1):
    result = []
    cases = [exp.Experiment.from_dir(e) for e in experiments]
    basenames = [os.path.basename(e) for e in experiments]

    for _type in range(_types):
        for p, e in enumerate(experiments):
            # print(e)
            case = cases[p]
            # we clean errors.
            if case is None:
                continue
            consumption = istats.get_consumptions(case.instance, _type=_type)
            aircraft_use = istats.get_consumptions(case.instance, hours=False, _type=_type)
            rel_consumption = istats.get_rel_consumptions(case.instance, _type=_type)
            cycle_2M_size = sol_stats.get_1M_2M_dist(case, _type=_type)
            cycle_1M_size = sol_stats.get_prev_1M_dist(case, _type=_type)
            cycle_2M_quants = cycle_2M_size.quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
            cycle_1M_size_values = cycle_1M_size.values_l()
            cycle_1M_quants = pd.Series(cycle_1M_size_values).quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
            l_maint_date = sol_stats.get_last_maint_date(case, _type=_type).values_l()
            init_hours = istats.get_init_hours(case.instance, _type=_type)
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
            num_maints = [sol_stats.get_num_maints(case, _type=_type)]
            cons_min_assign = istats.min_assign_consumption(case.instance, _type=_type).agg(['mean', 'max']).tolist()
            num_special_tasks = istats.get_num_special(case.instance, _type=_type)
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

    result_tab.loc[result_tab.num_errors == 0, 'has_errors'] = 'no errors'
    result_tab.loc[result_tab.num_errors > 0, 'has_errors'] = '>=1 errors'

    return result_tab

def get_status_df(experiments):
    status_df = instance_status(experiments, ['sol_code', 'status_code', 'time', 'gap', 'best_bound', 'best_solution'])
    status_df['gap_abs'] = status_df.best_solution - status_df.best_bound
    # status_df[status_df.sol_code==ol.LpSolutionInfeasible]
    # status_df[status_df.gap_abs > 100]
    # status_df.columns
    return status_df



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
    # os.environ['path'] += r';C:\Program Files (x86)\Graphviz2.38\bin'
    # from importlib import reload
    name = sto_params.name
    path = params.PATHS['results'] + name +'/base/'

    experiments = [os.path.join(path, i) for i in os.listdir(path)]
    basenames = [os.path.basename(e) for e in experiments]

    result_tab = get_table(experiments, 2)

    result_tab.loc[result_tab.gap_abs <= 60, 'gap_stat'] = 'gap_abs<=60'
    result_tab.loc[result_tab.gap_abs > 60, 'gap_stat'] = 'gap_abs>=60'

    status_df = get_status_df(experiments)

    status_df.agg('mean')[['gap_abs', 'time', 'best_solution']]
    status_df.groupby('status').agg('count')['name']
    status_df.groupby('status').agg('max')['gap_abs']
    status_df.groupby('status').agg('max')['gap']
    status_df.groupby('status').agg('median')['gap_abs']
    status_df.groupby('status').agg('median')['gap']


    (result_tab.num_errors==0).sum()
    graphs.cons_init(result_tab, y='cycle_2M_min', color='status', smooth=False)

    for var in ['maints', 'mean_dist', 'max_dist', 'min_dist', 'mean_2maint']:
        graphs.draw_hist(var)

    graphs.draw_hist('mean_2maint')

    # cases = [exp.Experiment.from_dir(e) for e in experiments]
    # hist_no_agg(basenames, cases)

    for y in ['maints', 'mean_dist', 'mean_2maint', 'cycle_2M_min']:
        x = 'mean_consum'
        graph_name = '{}_vs_{}'.format(x, y)
        graphs.plotting(result_tab[result_tab.gap_abs<=80], x=x,  y=y, color='status',
                        facet='init_cut ~ .' , graph_name=graph_name, smooth=True)

    x_vars = ['mean_consum', 'mean_consum2', 'mean_consum3', 'init', 'pos_consum9',
              'pos_consum5', 'quant5w', 'quant75w', 'quant9w', 'max_consum', 'var_consum',
              'cons_min_mean', 'cons_min_max']
    x_vars += ['spec_tasks']
    # x_vars += ['geomean_cons']
    bound_options = [('maints', True, 0.8), ('maints', False, 0.8), ('cycle_2M_min', False, 0.8)]
    table = result_tab
    # table = result_tab.query('mean_consum >=180 and gap_abs <= 30')
    # table = result_tab[result_tab.mean_consum.between(150, 300)]
    table = result_tab.query('mean_consum >=150 and mean_consum <=250 and gap_abs <= 80')
    # table = result_tab.query('mean_consum >=150 and mean_consum <=250 and num_errors==0')
    for predict_var, upper, alpha in bound_options:
        data = models.test_superquantiles(table, x_vars=x_vars,
                                   predict_var=predict_var,
                                   _lambda=10, alpha=alpha, plot=True, upper_bound=upper)
    data = models.test_regression(result_tab, x_vars, plot=False)
    data = models.test_regression(result_tab, x_vars)
    (result_tab.num_errors>0).sum()
    # 'mean_consum', 'init', 'mean_consum_2', 'mean_consum_3'

    for _var in ['maints', 'mean_2maint', 'mean_dist']:
        for grade in range(3, 4):
            for (alpha, sign) in zip([0.99, 0.8], [-1, 1]):
                print(_var, grade)
                models.test_superquantiles(result_tab, status_df, _var, _lambda=0.1, alpha=alpha, sign=sign)
                print()

# result_tab.groupby('spec_tasks_cut').name.agg('count')
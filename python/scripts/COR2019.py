import os
import pandas as pd
import numpy as np
import orloge as ol
import dfply as dp
from dfply import X

import package.reports as rep
import package.experiment as exp
import package.params as pm
import package.superdict as sd
import package.rpy_graphs as rg
import scripts.names as na
import package.heuristics_maintfirst as heur
import package.data_input as di

path = '/home/pchtsp/Documents/projects/COR2019/'
path = r'C:\Users\pchtsp\Documents\projects\COR2019/'

def boxplot_times(table, experiment):
    str_tup = 'time_out', 'Solving time', '_times'
    boxplot_var(table, experiment, str_tup)


def boxplot_gaps(table, experiment):
    str_tup = 'gap_out', 'Relative gap', '_gaps'
    boxplot_var(table, experiment, str_tup)


def bars_var(table, experiment, str_tup):
    col, ylab, path_ext = str_tup

    plot = rg.bars(table, x='scenario', y=col, xlab='Scenarios', ylab=ylab)
    plot.save(os.path.join(path, 'img', experiment + path_ext + '.png'))

def bars_no_int(table, experiment):
    table_n = table >> \
              dp.group_by(X.scenario) >> \
              dp.summarize(no_int=X.no_int.sum())

    str_tup = 'no_int', 'No integer solutions', '_no_int'
    bars_var(table_n, experiment, str_tup)


def bars_inf(table, experiment):
    table_n = table >> \
              dp.group_by(X.scenario) >> \
              dp.summarize(inf=X.inf.sum())

    str_tup = 'inf', 'Infeasible solutions', '_inf'
    bars_var(table_n, experiment, str_tup)


def boxplot_var(table, experiment, str_tup):
    col, ylab, path_ext = str_tup
    table_n = \
        table >> dp.filter_by(~X.inf) >> dp.mutate(filt=X.scenario == 'base') >> \
        dp.ungroup() >> dp.arrange(X.filt)
    # force float to avoid errors
    table_n[col] = table_n[col].astype('float')
    plot = rg.boxplot(table_n, x='case', y=col, xlab='Scenario', ylab=ylab)
    aspect_ratio = 2
    plot.save(os.path.join(path, 'img', experiment + path_ext + '.png'),
              height=7, width=7 * aspect_ratio)


def summary_to_latex(table, experiment):
    table_summary = rep.summary_table(table)
    rep.summary_to_latex(experiment=experiment, table=table_summary,
                         path=path + 'tables/')


def get_scenarios_to_compare(scenarios):

    df_list = {scen:
                   rep.get_results_table(pm.PATHS['results'] + p, get_exp_info=False)
               for scen, p in scenarios.items()
               }
    return pd.concat(df_list).reset_index() >> \
           dp.rename(scenario=X.level_0,
                     instance=X.level_1)

def get_relaxations(table):
    rename = {'obj_end': 'best_solution', 'rel_end': 'best_bound', 'rel_init': 'first_relaxed'}
    rename_cuts = {'obj_cuts': 'best_solution', 'rel_cuts': 'best_bound'}

    @dp.make_symbolic
    def _round(series):
        return np.round(series, 2)

    table_n = table >> \
              dp.rename(**rename) >> \
              dp.bind_cols(table.cut_info.apply(pd.Series)) >> \
              dp.rename(**rename_cuts)

    table_nn = \
        table_n >> \
        dp.select(X.scenario, X.instance, X.rel_init, X.rel_cuts, X.rel_end) >> \
        dp.mutate(rel_init=_round(X.rel_init),
                  rel_cuts=_round(X.rel_cuts),
                  rel_end=_round(X.rel_end))
    return table_nn.pivot(index="instance", columns='scenario')

def get_cuts(table):
    # table_nn.columns
    @dp.make_symbolic
    def has_cuts(series):
        return pd.Series(len(n)>0 if type(n) is dict else False for n in series)

    table_n = \
        table >> \
        dp.select(X.scenario, X.instance) >>\
        dp.bind_cols(table.cut_info.apply(pd.Series)) >> \
        dp.filter_by(has_cuts(X.cuts))

    table_nn = \
        table_n >> \
        dp.select(X.scenario, X.instance) >> \
        dp.bind_cols(table_n.cuts.apply(pd.Series)) >> \
        dp.gather('cut', 'num', dp.columns_from(2)) >> \
        dp.filter_by(X.num > 0) >>\
        dp.spread(X.scenario, X.num)

    return table_nn

def get_efficiency(table):
    table_nn = \
        table >> \
        dp.select(X.scenario, X.instance, X.nodes, X.time, X.gap)

    return table_nn.pivot(index="instance", columns='scenario')

def get_preprocessing(table):
    table_n = \
        table >> \
        dp.select(X.scenario, X.instance, X.presolve, X.matrix) >>\
        dp.bind_cols(table.presolve.apply(pd.Series)) >> \
        dp.bind_cols(table.matrix_post.apply(pd.Series)) >> \
        dp.select(X.scenario, X.instance, X.constraints, X.variables, X.time)

    table_nn = table_n.pivot(index="instance", columns='scenario')
    table_nn.columns = rep.col_names_collapsed(table_nn)
    return table_nn


# nsp = na.names_no_spaces()
# names_fr = na.simulation_params_fr()
# nsp_fr = {v: names_fr[k] for k, v in nsp.items()}
# pd.DataFrame.from_dict(nsp, orient='index')
# vars = ['minavailpercent', 'minavailvalue', 'minhoursperc', 'minusageperiod', 'numparalleltasks', 'numperiod',
#         'tminassign']

# sd.SuperDict(nsp_fr).filter(vars)

# plot = gp.boxplot_instances(table >> dp.filter_by(~X.inf))
# plot = gp.boxplot_instances(table >> dp.filter_by(~X.inf), column='gap_out')
# plot.save(path + 'img/' + experiment1 + '_gap.png')
# print(plot)

# name = experiment1 + '_' + experiment
    name = experiment


############################ TEST:

def test2():
    experiment = "clust1_20190408"
    path_exps = pm.PATHS['results'] + experiment
    exps = {p: os.path.join(path_exps, p) + '/' for p in os.listdir(path_exps)}


    # results_list = {k: get_results_table(v, get_exp_info=False) for k, v in exps.items()}
    exps = {k: exp.list_experiments(path_abs, get_exp_info=False) for k, path_abs in exps.items()}
    exps = sd.SuperDict.from_dict(exps)


    inf = {k: v.get_property('status') for k, v in exps.items()}
    cut_times = {k: v.get_property('cut_info').get_property('time') for k, v in exps.items()}
    cuts = {k: v.get_property('cut_info').get_property('cuts') for k, v in exps.items()}

    data = sd.SuperDict(inf).to_dictup().to_tuplist().to_list()
    data = sd.SuperDict(cuts).to_dictup().to_tuplist().to_list()
    data = sd.SuperDict(cut_times).to_dictup().to_tuplist().to_list()

    base_names = ['scenario', 'instance']
    names = base_names + ['status']
    names = base_names + ['cut', 'num']
    names = base_names + ['num']
    table = pd.DataFrame.from_records(data, columns=names).reset_index()

    table_sum  = \
        table >> \
        dp.group_by(X.scenario, X.instance) >> \
        dp.summarize(num = X.num.sum())

    scenarios = table >> dp.distinct(X.scenario) >> dp.select(X.scenario)
    scenarios = scenarios.reset_index(drop=True) >> dp.mutate(code=X.index)

    table_sum = table_sum >> dp.left_join(scenarios, on='scenario')

    # rg.rpy2_boxplot(table_sum, x='scenario', y='num')

    t = table >> dp.group_by(X.scenario, X.cut) >> dp.summarize(num = X.num.sum())
    t >> \
    dp.spread(X.scenario, X.num) >> \
    dp.select(X.cut, X.base, X.maxelapsedtime_40, X.maxelapsedtime_80, X.elapsedtimesize_40, X.elapsedtimesize_20) >> \
    dp.rename(mt80=X.maxelapsedtime_80, mt40=X.maxelapsedtime_40, es40=X.elapsedtimesize_40, es20=X.elapsedtimesize_20)

    plot = rep.boxplot_instances(table_sum, column='num_cuts')


    print(plot)
    # pd.io.json.json_normalize(cuts, record_path=['*', '*'])

def statistics_experiment(experiment):
    table = rep.get_simulation_results(experiment)
    boxplot_times(table, experiment)
    boxplot_gaps(table, experiment)
    summary_to_latex(table, experiment)

def cut_comparison():
    scenarios = \
        dict(
            MIN_HOUR_5="clust_params2_cplex/minusageperiod_5",
            MIN_HOUR_15="clust_params2_cplex/minusageperiod_15",
            MIN_HOUR_20="clust_params2_cplex/minusageperiod_20",
            BASE="clust_params2_cplex/base",
        )

    table = get_scenarios_to_compare(scenarios)

    # functions = [get_efficiency, get_preprocessing, get_relaxations]
    # functions = [get_cuts]
    # for f in functions:
    #     rep.print_table_md(f(table))

    cuts_table = get_cuts(table)
    for scenario in scenarios.keys():
        plot = rg.boxplot(cuts_table, 'cut', scenario)
        plot.save(os.path.join(path, 'img', scenario + '.png'))

    summary_medians = \
        cuts_table >> \
        dp.group_by(X.cut) >> \
        dp.summarize(BASE = X.BASE.median(),
                     MIN_5=X.MIN_HOUR_5.median(),
                     MIN_15=X.MIN_HOUR_15.median(),
                     MIN_20=X.MIN_HOUR_20.median(),
                     )
    print(summary_medians.\
        to_latex(bold_rows=True, index=False, float_format='%.0f'))


def cuts_relaxation_comparison():
    scenarios = \
        dict(
            elapsedtimesize_20="clust_params1_cplex/elapsedtimesize_20",
            elapsedtimesize_40="clust_params1_cplex/elapsedtimesize_40/",
            base="clust_params1_cplex/base",
        )

    table = get_scenarios_to_compare(scenarios)
    # table.columns
    names_df = na.config_to_latex(table.scenario)
    table_n = \
        table >> \
        dp.select(X.scenario, X.instance) >>\
        dp.bind_cols(table.cut_info.apply(pd.Series)) >> \
        dp.select(X.scenario, X.instance, X.best_bound, X.best_solution) >> \
        dp.mutate(gap_out= 100*(X.best_solution - X.best_bound)/X.best_solution) >> \
        dp.left_join(names_df, on="scenario")

    table_n['inf'] = pd.isna(table_n.gap_out)

    experiment = 'elapsed_time'
    # str_tup = 'gap_out', 'Relative gap', '_gaps'
    boxplot_gaps(table_n, experiment)


def statistics_relaxations(experiment):
    # experiment = "clust1_20181121"
    cols_rename = {
        'index': 'id', 'best_solution': 'best_solution',
        'best_bound': 'bound', 'sol_code': 'sol_code', 'status_code': 'status_code',
        'nodes': 'nodes', 'first_relaxed': 'first_relaxed', 'cut_info': 'cut_info'
    }
    table = rep.get_simulation_results(experiment, cols_rename)
    table_cuts = table.cut_info.apply(pd.Series)
    table_cuts.columns = ['cuts_' + str(i) for i in table_cuts.columns]
    table_n =\
        table >>\
            dp.bind_cols(table_cuts) >>\
            dp.mutate(opt = X.sol_code == ol.LpSolutionOptimal,
                      gap_init = 100*(X.best_solution - X.first_relaxed)/X.best_solution,
                      gap_cuts = 100*(X.best_solution - X.cuts_best_bound)/X.best_solution,
                      gap_cuts_int = 100*(X.cuts_best_solution - X.best_solution) / X.best_solution
                      ) >>\
            dp.select(X.scenario, X.instance, X.opt,  X.nodes, X.gap_init, X.gap_cuts, X.gap_cuts_int)

    table_n_sum =\
        table_n >>\
            dp.group_by(X.scenario) >>\
            dp.summarize_each([np.mean], X.gap_init, X.gap_cuts, X.gap_cuts_int)

    table_n_sum2 =\
        table_n >> \
            dp.filter_by(X.opt) >> \
            dp.group_by(X.scenario) >> \
            dp.summarize_each([np.mean], X.nodes)

    names_df = na.config_to_latex(table_n_sum.scenario)
    table_nn = \
        table_n_sum >> \
        dp.left_join(table_n_sum2) >> \
        dp.left_join(names_df, on="scenario") >> \
        dp.select(~X.scenario, ~X.name) >> \
        dp.rename(rcuts = X.gap_cuts_mean,
                  rinit= X.gap_init_mean,
                  icuts = X.gap_cuts_int_mean,
                  nodes = X.nodes_mean)
    cols = ['case'] + [c for c in table_nn.columns if c != 'case']
    table_nnn = table_nn >> dp.select(cols)
    latex = table_nnn.to_latex(float_format='%.1f', escape=False, index=False)
    file_path = os.path.join(path + 'tables/', '{}_cut_statistics.tex'.format(experiment))
    with open(file_path, 'w') as f:
        f.write(latex)

def table_heuristic(experiment):
    path_exps = pm.PATHS['results'] + experiment
    exps_inst = {name: {p: v for p, v in rep.path_contents_dict(path).items()}
                 for name, path in rep.path_contents_dict(path_exps).items()}
    exps_inst = sd.SuperDict.from_dict(exps_inst).to_dictup()
    # for each instance, we get info from the logs.
    # also, we calculate the objective function from the solution.
    paths_log = exps_inst.apply(lambda k, v: heur.parse_log_file(v + 'output.log'))
    exps_dict = exps_inst.apply(lambda k, v: exp.Experiment.from_dir(v))
    opts_dict = exps_inst.apply(lambda k, v: di.load_data(v + 'options.json'))
    exps_obj = exps_dict.apply(lambda k, v: v.get_objective_function(opts_dict[k]))

    results = pd.DataFrame.from_dict(paths_log, orient='index').reset_index()
    results.rename(columns=dict(level_0='scenario', level_1='date'), inplace=True)
    results['no_int'] = results.errors > 0
    results.set_index(['scenario', 'date'], inplace=True)
    results['best_solution'] = results.index.map(exps_obj)
    results.reset_index(inplace=True)

    return results

if __name__ == "__main__":
    ####################
    # Scenario analysis
    ####################
    experiments = ["clust1_20181121"]
    experiments = ['clust_params2_cplex_v2', 'clust_params1_cplex_v2']
    experiments = ['clust_params1_cplex_v2']
    experiments = ['clust1_20190408_remake']
    for experiment in experiments:
        # statistics_experiment(experiment)
        # statistics_relaxations(experiment)
        pass
    # cuts_relaxation_comparison()

    ####################
    # Scenario comparison
    ####################
    # cut_comparison()

    ####################
    # Heuristic vs model
    ####################
    cols_rename = {
        'time': 'time_out', 'index': 'id', 'best_solution': 'objective',
        'gap': 'gap_out', 'best_bound': 'bound', 'first_solution': 'first_solution',
        'sol_code': 'sol_code', 'status_code': 'status_code', 'date': 'date'
    }
    # make  get_first_solution return time, nodes. Same for first_relax.
    table_solver = rep.get_simulation_results(experiment='clust1_20190322',
                                              cols_rename=cols_rename)
    table_solver.sort_values(['scenario', 'date'], inplace=True)
    table_solver['id'] = \
        table_solver.groupby('scenario')['date'].\
        transform(lambda x: range(len(x)))

    table_heur = table_heuristic('clust1_20190408')
    table_heur.sort_values(['scenario', 'date'], inplace=True)
    table_heur['id'] = \
        table_heur.groupby('scenario')['date'].\
        transform(lambda x: range(len(x)))

    table_mh = table_solver.merge(table_heur, on=['scenario', 'id'], how='inner')

    table_mh['gap_abs_heur'] = table_mh.best_solution - table_mh.objective
    table_mh['gap_rel_heur'] = table_mh.gap_abs_heur / table_mh.objective * 100

    # we get only feasible instances
    feasible = table_mh.sol_code.isin([ol.LpSolutionOptimal, ol.LpSolutionIntegerFeasible])
    table_filt = table_mh[feasible].copy()
    table_filt['known'] = table_filt.groupby('scenario')['sol_code'].transform(func=lambda x: len(x))

    # now, we count how many we found an initial solution

    heur_stats = \
        table_filt[~table_mh.no_int_y].\
        groupby('scenario').\
        agg({'no_int_y':'count', 'time': 'mean'}).\
        rename(columns={'no_int_y': 'initial', 'time': 'timeH'})

    # Index(['scenario', 'instance', 'time_out', 'id', 'objective', 'gap_out',
    #        'bound', 'status', 'sol_code', 'status_code', 'matrix', 'cons', 'vars',
    #        'nonzeros', 'case', 'name_x', 'code', 'no_int_x', 'inf', 'name_y',
    #        'time', 'iters', 'temperature', 'errors', 'best', 'no_int_y',
    #        'best_solution', 'gap_abs_heur', 'gap_rel_heur'],
    #       dtype='object')
    names = na.config_to_latex(table_filt.scenario)

    results_agg = \
        table_filt.\
            groupby('scenario').\
            agg({'time_out': 'mean', 'known': 'first', 'gap_rel_heur': 'mean'}).\
            rename(columns={'time_out': 'timeM', 'gap_rel_heur': 'gap'}).\
            merge(heur_stats, on='scenario').\
            merge(names, on='scenario')
    results_agg['perc_init'] = results_agg.initial / results_agg.known * 100
    for col in ['perc_init', 'timeH', 'timeM', 'gap']:
        results_agg[col] = round(results_agg[col], 1)
    results_agg = results_agg.filter(['case', 'timeH', 'timeM', 'perc_init', 'gap'])
    latex = results_agg.to_latex(bold_rows=True, index=False, float_format='%.1f')
    file_path = os.path.join(path + 'tables/', 'heuristic_comp.tex')
    rep.print_table_md(results_agg)

    with open(file_path, 'w') as f:
        f.write(latex)

    # ###################
    # remake vs model
    # ###################
    # table_solver and table_heur from above
    table_remake = rep.get_simulation_results(experiment='clust1_20190408_remake',
                                              cols_rename=cols_rename)

    table_remake.sort_values(['scenario', 'date'], inplace=True)
    table_remake['id'] = table_remake.groupby('scenario')['date'].transform(lambda x: range(len(x)))

    table_solver['experiment'] = 'mip'
    table_remake['experiment'] = 'remake'
    table_m_r = pd.concat([table_solver, table_remake])

    # filter only cases when given a real initial solution:
    key = ['scenario', 'id']
    key2 = ['experiment', 'scenario']
    heur_has_init = table_heur[~table_heur.no_int][key]
    table_filt = table_m_r.merge(heur_has_init, on=key, how='inner')
    # table = table_solver.merge(table_remake, on=['scenario', 'id'])
    # table.columns
    num_sols = heur_has_init.groupby('scenario').agg(lambda x: len(x))
    table_filt.no_int += 0

    results_agg = \
        table_filt.\
            groupby(['experiment', 'scenario']).\
            agg({'time_out': 'mean',
                 'gap_out': 'mean',
                 'no_int': 'sum'}).round(1)
    results_agg2 = \
        results_agg.unstack(0).\
        merge(num_sols, on='scenario', how='left')

    rep.print_table_md(results_agg2)
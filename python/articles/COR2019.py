from articles import MOSIM2018 as rep
import os
import pandas as pd
import numpy as np
import package.params as pm
import dfply as dp
from dfply import X
import reports.rpy_graphs as rg
import strings.names as na
import orloge as ol
import pytups.superdict as sd

path = '/home/pchtsp/Documents/projects/COR2019/'


def boxplot_times(table, experiment):
    str_tup = 'time_out', 'Solving time (seconds)', '_times'
    boxplot_var(table, experiment, str_tup)


def boxplot_gaps(table, experiment):
    str_tup = 'gap_out', 'Relative gap (%)', '_gaps'
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
    plot = rg.boxplot(table_n, x='case', y=col, xlab='Configuration', ylab=ylab)
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
#     name = experiment


############################ TEST:

def statistics_experiment(experiment, scenarios=None):
    # experiment = "clust1_20181121"
    # experiment = 'clust1_20181128'
    table = rep.get_simulation_results(experiment, scenarios=scenarios)
    equiv = \
        sd.SuperDict(numparalleltasks_2 = 30, numparalleltasks_3 = 45, numparalleltasks_4 = 60).\
            vapply(lambda v: "$|I|={}$".format(v))
    table.case = table.scenario.map(equiv).fillna(table.case)
    # boxplot_times(table, experiment)
    # boxplot_gaps(table, experiment)
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


def statistics_relaxations(experiment, write=True, scenarios=None):
    # experiment = "clust1_20181121"
    cols_rename = {
        'index': 'id', 'best_solution': 'best_solution',
        'best_bound': 'bound', 'sol_code': 'sol_code', 'status_code': 'status_code',
        'nodes': 'nodes', 'first_relaxed': 'first_relaxed', 'cut_info': 'cut_info'
    }
    table = rep.get_simulation_results(experiment, cols_rename, scenarios=scenarios)
    table_cuts = table.cut_info.apply(pd.Series)
    table_cuts.columns = ['cuts_' + str(i) for i in table_cuts.columns]
    table_n =\
        table >>\
            dp.bind_cols(table_cuts) >>\
            dp.mutate(opt = X.sol_code==ol.LpSolutionOptimal,
                      gap_init = 100*(X.best_solution - X.first_relaxed)/X.best_solution,
                      gap_cuts = 100*(X.best_solution - X.cuts_best_bound)/X.best_solution,
                      gap_cuts_int = 100*(X.cuts_best_solution - X.best_solution) / X.best_solution
                      )  >>\
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
    if write:
        latex = table_nnn.to_latex(float_format='%.1f', escape=False, index=False)
        file_path = os.path.join(path + 'tables/', '{}_cut_statistics.tex'.format(experiment))
        with open(file_path, 'w') as f:
            f.write(latex)
    return table_nnn

def remake_boxplots():
    experiment = 'clust1_20190322'
    path_exps = '/home/pchtsp/Documents/projects/OPTIMA_documents/results/{}/'.format(experiment)
    scenarios = ['base', 'numparalleltasks_2', 'numparalleltasks_3', 'numparalleltasks_4',
                 'numperiod_120', 'numperiod_140']
    table = rep.get_simulation_results(experiment='', scenarios=scenarios, path_exps=path_exps)
    equiv = \
        sd.SuperDict(numparalleltasks_2 = 30, numparalleltasks_3 = 45, numparalleltasks_4 = 60).\
            vapply(lambda v: "$| I |={}$".format(v))
    table.case = table.scenario.map(equiv).fillna(table.case)
    table.drop(['matrix'], axis=1, inplace=True)
    table.groupby(['scenario', 'sol_code']).sol_code.count()
    table[table.scenario=='numperiod_140']
    # we edit the table to reflect new column names
    boxplot_times(table, experiment)
    boxplot_gaps(table, experiment)

    # experiment = 'IT000125_20190917'
    # path_exps = '/home/pchtsp/Documents/projects/optima_results/{}.zip'.format(experiment)
    # scenarios = ['base', 'numparalleltasks_2', 'numparalleltasks_3', 'numparalleltasks_4']
    # table = rep.get_simulation_results(experiment='', scenarios=scenarios, path_exps=path_exps, zip=True)
    # str_tup = 'gap_abs', 'Absolute gap', '_absgaps'
    # boxplot_var(table, experiment, str_tup)
    # np.percentile(table.gap_abs[~np.isnan(table.gap_abs)], 50)
    # boxplot_gaps(table, experiment)



if __name__ == "__main__":
    remake_boxplots()
    ####################
    # Scenario analysis
    ####################
    # experiments = ["clust1_20181121"]
    # experiments = ['clust_params2_cplex', 'clust_params1_cplex']
    # for experiment in experiments:
    #     # statistics_experiment(experiment)
    #     statistics_relaxations(experiment)
    # cuts_relaxation_comparison()


    ####################
    # Scenario comparison
    ####################
    # cut_comparison()

    ####################
    # Other?
    ####################

        # latex = table_n.to_latex(float_format='%.1f', escape=False, index=False)
        # file_path = os.path.join(path, '{}.tex'.format(experiment))
        # with open(file_path, 'w') as f:
        #     f.write(latex)
#
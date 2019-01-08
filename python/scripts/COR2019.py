import package.reports as rep
import os
import package.experiment as exp
import pandas as pd
import numpy as np
import package.params as pm
import package.superdict as sd
# import scripts.names as na
# import numpy as np
import dfply as dp
from dfply import X
import package.rpy_graphs as rg

path = '/home/pchtsp/Documents/projects/COR2019/'


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
        table >> dp.filter_by(~X.inf) >> dp.select(X.scenario, [col]) >>\
        dp.mutate(filt=X.scenario == 'base') >> dp.ungroup() >> dp.arrange(X.filt)
    # force float to avoid errors
    table_n[col] = table_n[col].astype('float')
    plot = rg.boxplot(table_n, x='scenario', y=col, xlab='Scenario', ylab=ylab)
    plot.save(os.path.join(path, 'img', experiment + path_ext + '.png'))


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
    return


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
    experiment = ""
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
    # experiment = "clust1_20181121"
    # experiment = 'clust1_20181128'
    table = rep.get_simulation_results(experiment)
    boxplot_times(table, experiment)
    boxplot_gaps(table, experiment)
    bars_no_int(table, experiment)
    bars_inf(table, experiment)
    summary_to_latex(table, experiment)


if __name__ == "__main__":
    ####################
    # Scenario analysis
    ####################
    experiments = ["clust1_20181121"]
    # experiments = 'clust1_20181128'
    experiments = ['clust_params2_cplex', 'clust_params1_cplex']
    for experiment in experiments:
        statistics_experiment(experiment)

    ####################
    # Solver comparison
    ####################

    # scenarios = \
    #     dict(
    #         GUROBI = "clust1_20181112/base/",
    #         CPLEX = "clust1_20181107/base/"
    #     )
    #
    # table = get_scenarios_to_compare(scenarios)
    #
    # functions = [get_efficiency, get_preprocessing, get_relaxations]
    # for f in functions:
    #     rep.print_table_md(f(table))

    ####################
    # Other?
    ####################

        # latex = table_n.to_latex(float_format='%.1f', escape=False, index=False)
        # file_path = os.path.join(path, '{}.tex'.format(experiment))
        # with open(file_path, 'w') as f:
        #     f.write(latex)
#
# def scrap1():
#     table.sol_code
#     table.status_code
#     table >> dp.select(X.status, X.gap_out)
#     table >> \
#         dp.filter_by(X.scenario=='minusageperiod_20') >> \
#         dp.select(X.instance, X.gap_out, X.inf, X.objective, X.no_int) >>\
#         dp.arrange(X.gap_out)
#     table.columns

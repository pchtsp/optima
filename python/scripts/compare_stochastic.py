import pandas as pd
import numpy as np

import package.batch as ba
import package.params as params

import stochastic.params as sto_params
import stochastic.solution_stats as sol_stats
#####################
# INFO


def get_sum_variances(case, types):
    return sum(sol_stats.get_variance_dist(case, t) for t in types)


def get_solstats(batch):
    cases = batch.get_cases().clean(func=lambda v: v)
    types = cases.vapply(lambda v: v.instance.get_types())
    variances = cases.apply(lambda k, v: {'variance': get_sum_variances(v, types[k])})
    var_table = batch.format_df(variances).drop('name', axis=1)
    return var_table


def get_df_comparison(exp_list, scenarios=None):
    """
    :param list exp_list: list of names of experiments
    :param list scenarios: optional list of scenarios to filter batches
    :return: pandas data frame
    """
    raw_results_dict = {}
    for name, experiment in enumerate(exp_list):
        # print(name)
        if experiment is None:
            continue
        batch1 = ba.ZipBatch(path=params.PATHS['results'] + experiment, scenarios=scenarios)
        table = batch1.get_log_df()
        table_errors = batch1.get_errors_df().drop('name', axis=1)
        sol_stats_table = get_solstats(batch1)
        table_n = \
            table.\
            merge(table_errors, on=['scenario', 'instance'], how='left').\
            merge(sol_stats_table, on=['scenario', 'instance'], how='left').\
            sort_values(['scenario', 'instance']).\
            reset_index(drop=True)
        raw_results_dict[name] = table_n

    return \
        pd.concat(raw_results_dict).\
        reset_index().\
        drop(['level_1'], axis=1).\
        rename(columns=dict(level_0='experiment')).\
        set_index(['instance','experiment']).reset_index()

# The following is obsolete because we now do the comparison in R:

def compare_perc_0_1(table):
    return (table[0] - table[1])/table[1] *100


def get_large_instances():
    exp_list = \
        ['IT000125_20190730', 'IT000125_20190801']
    df = get_df_comparison(exp_list)
    df = df[df.scenario == 'numparalleltasks_3']
    return df

def get_medium_instances():
    exp_list = \
        ['IT000125_20190730', 'IT000125_20190729']

    df = get_df_comparison(exp_list)
    df = df[df.scenario == 'numparalleltasks_2']
    return df

def get_small_instances():
    exp_list = \
        ['IT000125_20190725',
         'IT000125_20190716']

    df = get_df_comparison(exp_list)
    return df


if __name__ == '__main__':

    from rpy2.robjects import pandas2ri
    import rpy2.robjects.lib.ggplot2 as ggplot2
    import rpy2.robjects as ro
    import orloge as ol
    ro.pandas2ri.activate()

    # get info to analyze.
    # df = get_small_instances()
    # df = get_medium_instances()
    df = get_medium_instances()
    weird = df[~np.isnan(df.best_solution) & np.isnan(df.variance)]
    # import package.experiment as exp
    #
    # path = params.PATHS['results'] + 'IT000125_20190730'

    infeasible = \
        df[df.status_code==ol.LpStatusInfeasible][
        ['instance', 'experiment', 'time']].\
        set_index(['instance', 'experiment']).\
        unstack(1)
    _filt = np.any(~infeasible.isna(), axis=1)
    ttt = infeasible[_filt]['time']
    ttt['dif'] = (ttt[0] - ttt[1])/ttt[1]*100
    ttt['dif'].mean()
    mean = ttt.mean()
    (mean[0] - mean[1])/mean[1]*100

    df[df.index.get_level_values('experiment')==1]
    # GENERAL STATS
    # df.agg('mean')[['time', 'best_solution']]
    df.groupby(['experiment', 'status']).agg('count')['name']
    # df.groupby(['experiment', 'status']).agg('max')['gap_abs']
    df.groupby(['experiment', 'status']).agg('max')['gap']
    # df.groupby(['experiment', 'status']).agg('median')['gap_abs']
    df.groupby(['experiment', 'status']).agg('median')['gap']


    # EXPERIMENTS comparing optimality degradation.
    ############################################
    # when both said optimal solution.
    # we make the difference between the two.
    # tolerance was 10.
    # 95% of optimal instances had 36 or less difference in optimal solutions.
    # 66 for 2 tasks
    df_qu = df.query('sol_code==1').unstack(1).copy()
    df_qu['min_value'] = np.nanmin(df_qu.best_solution, axis=1)
    ttt = df_qu.best_solution.subtract(df_qu.min_value, axis=0)
    _filt = np.any(ttt.isna(), axis=1)
    ttt[~_filt][0].quantile(0.95)

    # optimality differences when comparing all feasible solutions
    # 95% of instances. for 2 tasks, 690 in cuts againts 12k
    df_best_sol = df.query('sol_code>=1').unstack(1).best_solution
    df_best_sol['min_value'] = np.nanmin(df_best_sol, axis=1)
    ttt = df_best_sol.subtract(df_best_sol.min_value, axis=0)
    _filt = np.any(ttt.isna(), axis=1)
    ttt[~_filt].quantile(0.95)
    ttt[~_filt].mean()

    # EXPERIMENTS comparing errors
    ############################################
    # 95% of integer feasible instances have the same quantity of errors.
    # In 2 tasks, cuts is still 0. base has 4 errors or less. The mean is a lot higher too.
    df_err = df.query('sol_code>0')['errors'].fillna(0).unstack(1)
    df_err['min_value'] = np.nanmin(df_err, axis=1)
    _filt = np.any(df_err.isna(), axis=1)
    ttt = df_err.subtract(df_err.min_value, axis=0)[~_filt].drop("min_value", axis=1)
    ttt.quantile(0.95)
    ttt[0].sort_values()
    ttt[1].sort_values()
    ttt.mean()
    ttt2 = ttt.stack()
    ttt2 = ttt2[ttt2 > 0].groupby('experiment').agg("count")
    compare_perc_0_1(ttt2)

    # feasibility
    # 1 task, 4% more infeasible solutions (3 from 5000)
    # 2 tasks 14% more infeasible solutions (6 from 1000)
    df_infea = \
        df.query('status_code==-1')['status_code'].\
        groupby('experiment').agg('count')
    (89 - 86)/86* 100
    compare_perc_0_1(df_infea)

    # EXPERIMENTS comparing performance
    ############################################
    # we only compare time in instances were we found a solution in both
    # Average solving time has a 27% reduction for all integer common feasible instances
    # if we replace not found by 3600: it's 24% gain
    # In 2 tasks, 21% gain and 13%
    # In 3 tasks, 15%% gain and 9%
    df_perf = df.query('sol_code>=1').unstack(1).copy()['time']

    # alternative 1 tab: both found an integer solution
    _filt = np.any(df_perf.isna(), axis=1)
    tab = df_perf[~_filt]

    # alternative 2 tab: if only one found, put 3600
    tab = df_perf.fillna(3600)

    tab.mean()
    compare_perc_0_1(tab.mean())
    tab.quantile(0.95)
    t_main = \
        tab.reset_index().assign(sum=lambda x: x[1]).sort_values('sum').reset_index(drop=True).\
        drop(['instance', 'sum'], axis=1).\
        stack().reset_index().rename(columns={'level_0': 'instance', 0: 'time'})

    _rename = {0: 'with_cuts', 1: 'no_cuts'}
    t_main.experiment = t_main.experiment.map(_rename)

    # alternative t_main:

    t0 = tab[0].sort_values().reset_index(drop=True)
    t1 = tab[1].sort_values().reset_index(drop=True)

    t_main = \
        pd.DataFrame(dict(with_cuts=t0, no_cuts=t1)).stack().\
            rename_axis(['instance', 'experiment']).reset_index(name='time')

    # end alternative table
    exp_list = ['aaaa', 'aaaa']
    t_main.experiment = t_main.experiment.astype(str)
    graph_name = 'comparison/{}_instance_time'.format('_'.join(exp_list))
    plot = ggplot2.ggplot(t_main) + \
           ggplot2.aes_string(x = 'instance', y='time', color='experiment') + \
           ggplot2.theme_minimal()

    plot += ggplot2.geom_point(size=0.5)
    # plot += ggplot2.geom_jitter(height=100, size=0.5)
    path_out = sto_params.path_graphs + r'{}.png'.format(graph_name)
    plot.save(path_out)

    # now we compare time differences in cases where an optimal solution was found.
    # For common optimal solutions, the reduction was 30% in time
    # if we replace not found solutions by 3600: 50%
    # In 2 taks, it's 43% for both.
    # In 3 taks, it's 35% and 41%.
    df_perf3 = df.query('sol_code==1').unstack(1).copy()['time']
    _filt = np.any(df_perf3 .isna(), axis=1)
    tab = df_perf3[~_filt]

    # alternative 2 tab: if only one found, put 3600
    tab = df_perf3.fillna(3600)

    tab.mean()
    compare_perc_0_1(tab.mean())

    # EXPERIMENTS finding feasible solutions
    ############################################

    # now we compare how many solutions
    # were found in one and not in the other
    # From 234, we get 53 => big reduction (77%)
    # In 2 tasks, it's 61%
    df_perf2 = \
        df.query('sol_code==0')['time'].\
            groupby('experiment').agg('count')

    compare_perc_0_1(df_perf2)

    # now we ask how many from each side
    _filt = np.any(df_perf.isna(), axis=1)
    df_perf[_filt].stack(0).groupby('experiment').agg('count')

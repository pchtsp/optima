import pandas as pd

import package.batch as ba
import package.params as params

import stochastic.solution_stats as sol_stats

#####################
# INFO
#####################


def get_sum_variances(case, types):
    return sum(sol_stats.get_variance_dist(case, t) for t in types)


def get_solstats(batch):
    # we do not add objective function because it depends on the original class
    # this should be done by injecting a solstats_func as argument
    # this is just an example on what the function structure should have
    cases = batch.get_cases().clean(func=lambda v: v)
    types = cases.vapply(lambda v: v.instance.get_types())
    variances = cases.kvapply(lambda k, v: {'variance': get_sum_variances(v, types[k])})
    var_table = batch.format_df(variances).drop('name', axis=1)
    return var_table


def empty_logs(batch):
    col = batch.get_instances_paths().vapply(lambda v: {'no_log': 1})
    return batch.format_df(col)


def get_df_comparison(exp_list, scenarios=None, get_progress=False, zip=True, get_log=True, solstats_func=None, solver=None):
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
        if zip:
            batch1 = ba.ZipBatch(path=params.PATHS['results'] + experiment, scenarios=scenarios)
        else:
            batch1 = ba.Batch(path=params.PATHS['results'] + experiment, scenarios=scenarios)
        if get_log:
            _solver = None
            if solver:
                _solver = solver.get(experiment)
            table = batch1.get_log_df(get_progress=get_progress, solver=_solver)
        else:
            table = empty_logs(batch1)
        table_errors = batch1.get_errors_df().drop('name', axis=1)
        if not solstats_func:
            solstats_func = get_solstats
        sol_stats_table = solstats_func(batch1)
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

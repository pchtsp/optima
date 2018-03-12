import pulp as pl
import package.aux as aux
import package.data_input as di
import package.config as conf
import numpy as np
import package.tests as test
import package.instance as inst
import package.solution as sol


######################################################
# TODO: add minimum mission duration assignment
# TODO: contraintes les posibilités des candidates: maximum X candidates par task ou par resource

# def cluster_candidates(instance, options=None):
#     l = instance.get_domains_sets()
#     av = list(set(aux.tup_filter(l['avt'], [0, 1])))
#     a_v = aux.tup_to_dict(av, result_col=0, is_list=True)
#     candidate = pl.LpVariable.dicts("cand", av, 0, 1, pl.LpInteger)
#
#     model = pl.LpProblem("Candidates", pl.LpMinimize)
#     for v, num in instance.get_tasks('num_resource').items():
#         model += pl.lpSum(candidate[(a, v)] for a in a_v[v]) >= max(num + 4, num * 1.1)
#
#     # # objective function:
#     # max_unavail = pl.LpVariable("max_unavail")
#     model += pl.lpSum(candidate[tup] for tup in av)
#
#     # MODEL
#
#     # # OBJECTIVE:
#     # model += max_unavail + max_maint * maint_weight
#
#     config = conf.Config(options)
#     result = config.solve_model(model)
#
#     return {}


def solve_model(instance, options=None):
    l = instance.get_domains_sets()
    ub = instance.get_bounds()
    last_period = instance.get_param('end')
    # last_period_lessM = aux.shift_month(last_period, -instance.get_param('maint_duration'))
    consumption = instance.get_tasks('consumption')
    requirement = instance.get_tasks('num_resource')
    ret_init = instance.get_initial_state("elapsed")
    rut_init = instance.get_initial_state("used")
    ret_obj = sum(ret_init[a] for a in l['resources'])
    rut_obj = sum(rut_init[a] for a in l['resources'])
    num_resource_working = instance.get_total_period_needs()
    num_resource_maint = aux.fill_dict_with_default(instance.get_total_fixed_maintenances(), l['periods'])
    num_resource_maint_cluster = aux.dict_to_lendict(instance.get_fixed_maintenances_cluster())
    instance.get_total_fixed_maintenances()
    maint_weight = instance.get_param("maint_weight")
    unavail_weight = instance.get_param("unavail_weight")
    maint_capacity = instance.get_param('maint_capacity')
    max_elapsed = instance.get_param('max_elapsed_time')
    max_usage = instance.get_param('max_used_time')
    min_percent = 0.10
    min_value = 1

    c_needs = instance.get_cluster_needs()
    c_candidates = instance.get_cluster_candidates()
    c_num_candidates = aux.dict_to_lendict(c_candidates)
    c_slack = {tup: c_num_candidates[tup[0]] - c_needs[tup] - num_resource_maint_cluster.get(tup, 0)
               for tup in c_needs}
    c_min = {(k, t): min(
        c_slack[(k, t)],
        max(
            int(c_num_candidates[k] * min_percent),
            min_value)
        ) for (k, t) in c_slack
    }

    # VARIABLES:
    # binary:
    task = pl.LpVariable.dicts("task", l['avt'], 0, 1, pl.LpInteger)
    start = pl.LpVariable.dicts("start", l['at_start'], 0, 1, pl.LpInteger)

    # numeric:
    ret = pl.LpVariable.dicts("ret", l['at0'], 0, ub['ret'], pl.LpContinuous)
    rut = pl.LpVariable.dicts("rut", l['at0'], 0, ub['rut'], pl.LpContinuous)

    # objective function:
    num_maint = pl.LpVariable("num_maint")
    ret_obj_var = pl.LpVariable("ret_obj_var")
    rut_obj_var = pl.LpVariable("rut_obj_var")

    # MODEL
    model = pl.LpProblem("MFMP_v0002", pl.LpMinimize)

    # OBJECTIVE:
    model += num_maint * max_elapsed * 2 - \
             ret_obj_var - \
             rut_obj_var * max_elapsed / max_usage

    # CONSTRAINTS:

    # num resources:
    for (v, t) in l['a_vt']:
        model += pl.lpSum(task[(a, v, t)] for a in l['a_vt'][(v, t)]) == requirement[v]
    # max one task per period or no-task state:
    # TODO: it is possible that there are no possible tasks but maintenances (domains)
    for (a, t) in l['v_at']:
        if len(l['v_at'][(a, t)]) + len(l['t1_at2'][(a, t)]) > 0:
            model += pl.lpSum(task[(a, v, t)] for v in l['v_at'][(a, t)]) + \
                     pl.lpSum(start[(a, _t)] for _t in l['t1_at2'][(a, t)] if (a, _t) in l['at_start']) + \
                     ((a, t) in l['at_maint']) <= 1

    # remaining used time calculations:
    # remaining elapsed time calculations:
    for (a, t) in l['at']:
        model += rut[(a, t)] <= rut[(a, l["previous"][t])] - \
                                pl.lpSum(task[(a, v, t)] * consumption[v] for v in l['v_at'].get((a, t), [])) + \
                                ub['rut'] * start.get((a, t), 0)
        model += ret[(a, t)] <= ret[(a, l["previous"][t])] - 1 + \
                                ub['ret'] * start.get((a, t), 0)
        model += rut[(a, t)] >= ub['rut'] * start.get((a, t), 0)
        model += ret[(a, t)] >= ub['ret'] * start.get((a, t), 0)

    # the start period is given by parameters:
    for a in l['resources']:
        model += rut[(a, l['period_0'])] == rut_init[a]
        model += ret[(a, l['period_0'])] == ret_init[a]
        if ret_init[a] < len(l['periods']):
            # if ret is low: we know we need a maintenance
            model += pl.lpSum(start.get((a, t), 0) for pos, t in enumerate(l['periods'])
                              if pos < ret_init[a]) >= 1

    # model += pl.lpSum(start.get((a, t2), 0) for t2 in l['periods']
    #                   if t <= t2 <= aux.shift_month(t, 59) ) \
    #          >= start.get((a, t), 0)


    # minimum availability per cluster and period
    for k, t in c_needs:
        model += \
            pl.lpSum(start[(a, _t)] for (a, _t) in l['at1_t2'][t]
                     if (a, _t) in l['at_start']
                     if a in c_candidates[k]) + \
            c_needs[(k, t)] + \
            c_min[(k, t)] + \
            num_resource_maint_cluster.get((k, t), 0) \
            <= c_num_candidates[k]
        # maintenances decided by the model to candidates +
        # assigned resources to tasks in cluster +
        # minimum resources for cluster +
        # <= resources already in maintenance

    # count the number of maintenances:
    model += num_maint == pl.lpSum(start[(a, _t)] for (a, _t) in l['at_start'])

    # max number of maintenances:
    for t in l['periods']:
        model += pl.lpSum(start[(a, _t)] for (a, _t) in l['at1_t2'][t] if (a, _t) in l['at_start']) + \
                 num_resource_maint[t] <= maint_capacity

    # calculate the rem and ret:
    model += pl.lpSum(ret[(a, last_period)] for a in l['resources']) == ret_obj_var
    model += pl.lpSum(rut[(a, last_period)] for a in l['resources']) == rut_obj_var

    # SOLVING
    config = conf.Config(options)
    result = config.solve_model(model)

    if result != 1:
        print("Model resulted in non-feasible status")
        return None

    _task = aux.tup_to_dict(aux.vars_to_tups(task), result_col=1, is_list=False)
    _start = {k: 1 for k in aux.vars_to_tups(start)}
    _rut = {t: rut[t].value() for t in rut}
    _ret = {t: ret[t].value() for t in ret}

    _state = {tup: 'M' for tup in l['planned_maint']}
    _state.update({(a, t2): 'M' for (a, t) in _start for t2 in l['t2_at1'][(a, t)]})

    solution_data_pre = {
        'state': _state,
        'task': _task,
        'aux': {
            'start': _start,
            'rut': _rut,
            'ret': _ret
        }
    }

    solution_data = {k: aux.dicttup_to_dictdict(v)
                     for k, v in solution_data_pre.items() if k != "aux"}
    solution_data['aux'] = {k: aux.dicttup_to_dictdict(v)
                            for k, v in solution_data_pre['aux'].items()}
    solution = sol.Solution(solution_data)

    return solution


if __name__ == "__main__":
    pass
import pulp as pl
import package.auxiliar as aux
import package.config as conf
import package.solution as sol


######################################################
def solve_model(instance, options=None):
    l = instance.get_domains_sets()
    ub = instance.get_bounds()
    first_period = instance.get_param('start')
    last_period = instance.get_param('end')
    consumption = instance.get_tasks('consumption')
    requirement = instance.get_tasks('num_resource')
    ret_init = instance.get_initial_state("elapsed")
    rut_init = instance.get_initial_state("used")
    num_resource_maint = aux.fill_dict_with_default(instance.get_total_fixed_maintenances(), l['periods'])
    num_resource_maint_cluster = aux.dict_to_lendict(instance.get_fixed_maintenances_cluster())
    maint_capacity = instance.get_param('maint_capacity')
    max_usage = instance.get_param('max_used_time')

    # TODO: fill this:
    task_previous_res = {v: [] for v in l['tasks']}
    fixed_vars = [(a, v, t) for (a, v, t) in range(0)]
    # TODO: fill this (up)

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

    # Sometimes we want to force variables to be integer.
    var_type = pl.LpContinuous
    if options.get('integer', False):
        var_type = pl.LpInteger

    # VARIABLES:
    # binary:
    start_T = pl.LpVariable.dicts(name="start_T", indexs=l['avt'], lowBound=0, upBound=1, cat=pl.LpInteger)
    task = pl.LpVariable.dicts(name="task", indexs=l['avt'], lowBound=0, upBound=1, cat=pl.LpInteger)
    start_M = pl.LpVariable.dicts(name="start_M", indexs=l['at_start'], lowBound=0, upBound=1, cat=pl.LpInteger)

    # numeric:
    rut = pl.LpVariable.dicts(name="rut", indexs=l['at0'], lowBound=0, upBound=ub['rut'], cat=var_type)

    # objective function:
    num_maint = pl.LpVariable(name="num_maint", lowBound=0, upBound=ub['num_maint'], cat=var_type)
    rut_obj_var = pl.LpVariable(name="rut_obj_var", lowBound=0, upBound=ub['rut_end'], cat=var_type)

    # MODEL
    model = pl.LpProblem("MFMP_v0002", pl.LpMinimize)

    # OBJECTIVE:
    if options.get('integer', False):
        objective = pl.LpVariable(name="objective", cat=var_type)
        model += objective
        model += objective >= num_maint * max_usage - rut_obj_var
    else:
        model += num_maint * max_usage - rut_obj_var

    # To try Kozanidis objective function:
    # we sum the rut for all periods (we take out the periods under maintenance)
    # model += - sum(rut[tup] for tup in rut) + num_maint * max_usage * maint_duration

    # CONSTRAINTS:

    # max one task per period or unavailable state:
    for at in l['at']:
        a, t = at
        v_at = l['v_at'].get(at, [])  # possible missions for that "at"
        t1_at2 = l['t1_at2'].get(at, [])  # possible starts of maintenance to be in maintenance status at "at"
        if len(v_at) + len(t1_at2) == 0:
            continue
        model += pl.lpSum(task[(a, v, t)] for v in v_at) + \
                 pl.lpSum(start_M[(a, _t)] for _t in t1_at2 if (a, _t) in l['at_start']) + \
                 (at in l['at_maint']) <= 1

    # ##################################
    # Tasks and tasks starts
    # ##################################

    # num resources:
    for (v, t) in l['a_vt']:
        model += pl.lpSum(task[(a, v, t)] for a in l['a_vt'][(v, t)]) == requirement[v]

    # definition of task start:
    # if we have a task now but we didn't before: we started it
    for avt in l['avt']:
        a, v, t = avt
        avt_ant = a, v, l['previous'][t]
        if t > first_period:
            model += start_T[avt] >= task[avt] - task.get(avt_ant, 0)
        else:
            model += start_T[avt] >= task[avt] - (a in task_previous_res[v])

    # if we start a task in at least on earlier period, we need to assign a task
    for a, v, t1, t2 in l['avt']:
        avt2 = a, v, t2
        model += task[avt2] >= pl.lpSum(start_T[a, v, t1] for (t1) in l['t1_avt2'])

    # at the beginning of the planning horizon, we may have fixed assignments of tasks.
    # we need to fix the corresponding variable.
    for avt in fixed_vars:
        model += task[avt] == 1

    # ##################################
    # Clusters
    # ##################################

    # minimum availability per cluster and period
    # TODO: check this and add the minimum usage hours per cluster
    for k, t in c_needs:
        model += \
            pl.lpSum(start_M[(a, _t)] for (a, _t) in l['at1_t2'][t]
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

    # ##################################
    # Usage time
    # ##################################

    # remaining used time calculations:
    for at in l['at']:
        a, t = at
        model += rut[at] <= rut[(a, l["previous"][t])] - \
                                pl.lpSum(task[(a, v, t)] * consumption[v] for v in l['v_at'].get(at, [])) + \
                                ub['rut'] * start_M.get(at, 0)

        model += rut[at] >= ub['rut'] * start_M.get(at, 0)

    # calculate the rut:
    model += pl.lpSum(rut[(a, last_period)] for a in l['resources']) == rut_obj_var

    # ##################################
    # Maintenances
    # ##################################

    # TODO: add ret constraints.

    # the start_M period is given by parameters:
    for a in l['resources']:
        model += rut[(a, l['period_0'])] == rut_init[a]
        if ret_init[a] < len(l['periods']):
            # if ret is low: we know we need a maintenance
            model += pl.lpSum(start_M.get((a, t), 0) for pos, t in enumerate(l['periods'])
                              if pos < ret_init[a]) >= 1

    # This is a possible cut for the 60 period ret
    # model += pl.lpSum(start_M.get((a, t2), 0) for t2 in l['periods']
    #                   if t <= t2 <= aux.shift_month(t, 59) ) \
    #          >= start_M.get((a, t), 0)

    # count the number of maintenances:
    model += num_maint == pl.lpSum(start_M[(a, _t)] for (a, _t) in l['at_start'])

    # max number of maintenances:
    for t in l['periods']:
        model += pl.lpSum(start_M[(a, _t)] for (a, _t) in l['at1_t2'][t] if (a, _t) in l['at_start']) + \
                 num_resource_maint[t] <= maint_capacity

    # SOLVING
    config = conf.Config(options)
    # model.writeMPS(filename='MFMP_3.mps')
    # return None
    result = config.solve_model(model)

    if result != 1:
        print("Model resulted in non-feasible status")
        return None

    _task = aux.tup_to_dict(aux.vars_to_tups(task), result_col=1, is_list=False)
    _start = {k: 1 for k in aux.vars_to_tups(start_M)}
    _rut = {t: rut[t].value() for t in rut}

    _state = {tup: 'M' for tup in l['planned_maint']}
    _state.update({(a, t2): 'M' for (a, t) in _start for t2 in l['t2_at1'][(a, t)]})

    solution_data_pre = {
        'state': _state,
        'task': _task,
        'aux': {
            'start_M': _start,
            'rut': _rut,
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
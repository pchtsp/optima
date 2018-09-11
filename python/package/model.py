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
    rut_init = instance.get_initial_state("used")
    num_resource_maint = aux.fill_dict_with_default(instance.get_total_fixed_maintenances(), l['periods'])
    maint_capacity = instance.get_param('maint_capacity')
    max_usage = instance.get_param('max_used_time')
    cluster_data = instance.get_cluster_constraints()
    c_candidates = instance.get_cluster_candidates()

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
    for (v, t), a_list in l['a_vt'].items():
        model += pl.lpSum(task[a, v, t] for a in a_list) == requirement[v]

    # definition of task start:
    # if we have a task now but we didn't before: we started it
    for avt in l['avt']:
        a, v, t = avt
        avt_ant = a, v, l['previous'][t]
        if t > first_period:
            model += start_T[avt] >= task[avt] - task.get(avt_ant, 0)
        else:
            # we check if we have the assignment in the previous period.
            # this is stored in the fixed_vars set.
            model += start_T[avt] >= task[avt] - (avt_ant in l['at_mission_m'])

    # definition of task start (2):
    # if we start a task in at least one earlier period, we need to assign a task
    for (a, v, t2), t1_list in l['t1_avt2'].items():
        avt2 = a, v, t2
        model += task.get(avt2, 0) >= \
                 pl.lpSum(start_T.get((a, v, t1), 0) for t1 in t1_list)

    # at the beginning of the planning horizon, we may have fixed assignments of tasks.
    # we need to fix the corresponding variable.
    for avt in l['at_mission_m']:
        model += task[avt] == 1

    # ##################################
    # Clusters
    # ##################################

    # minimum availability per cluster and period
    for (k, t), num in cluster_data['num'].items():
        model += \
            pl.lpSum(start_M[(a, _t)] for (a, _t) in l['at1_t2'][t]
                     if (a, _t) in l['at_start']
                     if a in c_candidates[k]) <= num

    # Each cluster has a minimum number of usage hours to have
    # at each period.
    for (k, t), hours in cluster_data['hours'].items():
        model += pl.lpSum(rut[a, t] for a in c_candidates[k] if (a, t) in l['at']) >= hours

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

    for a in l['resources']:
        model += rut[(a, l['period_0'])] == rut_init[a]

    # ##################################
    # Maintenances
    # ##################################

    # # we cannot do two maintenances too close one from the other:
    # for att in l['att_m']:
    #     a, t1, t2 = att
    #     model += start_M[a, t1] + start_M[a, t2] <= 1

    # we cannot do two maintenances too far apart one from the other:
    # (we need to be sure that t2_list includes the whole horizon to enforce it)
    for (a, t1), t2_list in l['t_at_M'].items():
        model += pl.lpSum(start_M[a, t2] for t2 in t2_list) >= start_M[a, t1]

    # if we have had a maintenance just before the planning horizon
    # we cant't have one at the beginning:
    # we can formulate this as constraining the combinations of maintenance variables.
    # (already done)
    # for at in l['at_m_ini']:
    #     model += start_M[at] == 0

    # TODO: this constraint is not working properly.
    # if we need a maintenance inside the horizon, we enforce it
    for a, t_list in l['t_a_M_ini'].items():
        model += pl.lpSum(start_M.get((a, t), 0) for t in t_list) >= 1

    # count the number of maintenances:
    model += num_maint == pl.lpSum(start_M[(a, _t)] for (a, _t) in l['at_start'])

    # max number of maintenances:
    for t in l['periods']:
        model += pl.lpSum(start_M[(a, _t)] for (a, _t) in l['at1_t2'][t] if (a, _t) in l['at_start']) + \
                 num_resource_maint[t] <= maint_capacity

    # ##################################
    # SOLVING
    # ##################################

    # SOLVING
    config = conf.Config(options)
    # model.writeMPS(filename='MFMP_3.mps')
    # return None
    result = config.solve_model(model)

    if result != 1:
        print("Model resulted in non-feasible status: {}".format(result))
        return None
    print('model solved correctly')

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
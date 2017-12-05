import pulp as pl
import package.aux as aux
import os
import package.data_input as di
import package.config as conf

######################################################


def solve_with_states(model_data, previous_states, solver="CBC"):
    """
    :param model_data: data to consruct and solve model. taken from get_model_data()
    :previous_states: past information that conditions the problem
    :return: solution of solved model
    """
    resources_data = model_data['resources']
    param_data = model_data['parameters']
    task_data = model_data['tasks']

    # HORIZON:
    first_period = param_data['start']
    last_period = param_data['end']

    # SETS:
    resources = list(resources_data.keys())  # a
    tasks = list(task_data.keys())  # v
    periods = aux.get_months(first_period, last_period)  # t
    states = ['M']  # s

    # TODO: this block is for testing exclusively:
    periods = periods[:50]
    tasks = [t for t in tasks if t != 'O8']  # there is something weird with this mission 08

    last_period = periods[-1]
    period_0 = aux.get_prev_month(first_period)
    periods_0 = [period_0] + periods  # periods with the previous one added at the start.

    # PARAMETERS:
    # maintenances:
    max_elapsed_time = param_data['max_elapsed_time']  # me. in periods
    max_used_time = param_data['max_used_time']  # mu. in hours of usage
    duration = param_data['maint_duration']  # md. in periods
    capacity = {t: param_data['maint_capacity'] for t in periods}  # c. in resources per period

    # tasks - resources
    start_time = aux.get_property_from_dic(task_data, 'start')  # not defined.
    end_time = aux.get_property_from_dic(task_data, 'end')  # not defined.
    candidates = aux.get_property_from_dic(task_data, 'candidates')  # cd. indexed set of resources.
    consumption = aux.get_property_from_dic(task_data, 'consumption')  # rh. hours per period.
    requirement = aux.get_property_from_dic(task_data, 'num_resource')  # rr. aircraft per period.

    # time:
    periods_pos = {periods[pos]: pos for pos in range(len(periods))}
    previous = {period: periods_0[periods_pos[period]] for period in periods}

    # initial state:
    ret_read = aux.get_property_from_dic(resources_data, 'initial_elapsed')
    rut_read = aux.get_property_from_dic(resources_data, 'initial_used')

    ret_init = {a: max_elapsed_time for a in resources}
    rut_init = {a: max_used_time for a in resources}

    ret_init.update(ret_read)
    rut_init.update(rut_read)

    ret_obj = sum(ret_init[a] for a in resources)
    rut_obj = sum(rut_init[a] for a in resources)
    # TODO: add minimum mission duration assignment

    # Here we calculate the initial periods were aircraft need maintenance
    planned_maint = aux.get_fixed_maintenances(previous_states, first_period, duration)

    # maximal bounds on continuous variables:
    ub = {
        'ret': max_elapsed_time,
        'rut': max_used_time,
        'used': max(consumption.values())
    }

    # DOMAINS:
    vt = [(v, t) for v in tasks for t in periods if start_time[v] <= t <= end_time[v]]
    avt = [(a, v, t) for a in resources for (v, t) in vt if a in candidates[v]]
    at = [(a, t) for a in resources for t in periods]
    ast = [(a, s, t) for (a, t) in at for s in states]

    # this is tricky: we  limit the possibilities of starting a maintenance:
    # at_start = [(a, t) for a, t in at if periods_pos[t] % 2 == 0]
    at_start = [(a, t) for a, t in at]
    at_start_not = [tup for tup in at if tup not in at_start]

    at0 = [(a, period_0) for a in resources] + at
    att = [(a, t1, t2) for (a, t1) in at for t2 in periods if
           periods_pos[t1] <= periods_pos[t2] <= periods_pos[t1] + duration - 1 and \
           (a, t1) in at_start]

    # a_t = {t: a for t in periods for a in resources if (a, t) in at}
    a_t = aux.tup_to_dict(at, result_col=0, is_list=True)
    a_vt = aux.tup_to_dict(avt, result_col=0, is_list=True)
    v_at = aux.tup_to_dict(avt, result_col=1, is_list=True)
    t1_at2 = aux.tup_to_dict(att, result_col=1, is_list=True)

    # number of resources in missions per month:
    num_resource_working = {t: 0 for t in periods}
    for (v, t) in vt:
        num_resource_working[t] += requirement[v]

    # VARIABLES:
    # binary:
    task = pl.LpVariable.dicts("task", avt, 0, 1, pl.LpInteger)
    start = pl.LpVariable.dicts("start", at_start, 0, 1, pl.LpInteger)
    state = pl.LpVariable.dicts("state", ast, 0, 1, pl.LpInteger)

    # numeric:
    ret = pl.LpVariable.dicts("ret", at0, 0, ub['ret'], pl.LpContinuous)
    rut = pl.LpVariable.dicts("rut", at0, 0, ub['rut'], pl.LpContinuous)
    used = pl.LpVariable.dicts("used", at, 0, ub['used'], pl.LpContinuous)

    # objective function:
    max_unavail = pl.LpVariable("max_unavail")
    max_maint = pl.LpVariable("max_maint")

    # MODEL
    model = pl.LpProblem("MFMP_v0001", pl.LpMinimize)

    # OBJECTIVE:
    model += max_unavail + max_maint

    # CONSTRAINTS:
    for t in periods:
        # objective: maintenance (commented because not in use)
        model += pl.lpSum(state[(a, 'M', t)] for a in a_t[t]) <= max_maint
        # objective: availability
        model += pl.lpSum(state[(a, 'M', t)] for a in a_t[t]) + num_resource_working[t] <= max_unavail

    # num resources:
    for (v, t) in a_vt:
        model += pl.lpSum(task[(a, v, t)] for a in a_vt[(v, t)]) >= requirement[v]

    # max one task per period or no-task state:
    for (a, t) in v_at:
        model += pl.lpSum(task[(a, v, t)] for v in v_at[(a, t)]) + \
                 pl.lpSum(state[(a, s, t)] for s in states) <= 1

    # used time, two options:
    # maybe set equal?
    # not sure which one is better, both?
    for (a, t) in v_at:
        model += used[(a, t)] >= pl.lpSum(task[(a, v, t)] * consumption[v] for v in v_at[(a, t)])
    for (a, v, t) in avt:
        model += used[(a, t)] >= task[(a, v, t)] * consumption[v]

    # remaining used time calculations:
    # remaining elapsed time calculations:
    # *maybe* reformulate this
    # We only increase the remainders if in that month we could start a maintenance
    for (a, t) in at_start:
        model += rut[(a, t)] <= rut[(a, previous[t])] - used[(a, t)] + max_used_time * start[(a, t)]
        model += ret[(a, t)] <= ret[(a, previous[t])] - 1 + max_elapsed_time * start[(a, t)]

    # if that month we know we're not starting a maintenance... it's just decreasing:
    for (a, t) in at_start_not:
        model += rut[(a, t)] <= rut[(a, previous[t])] - used[(a, t)]
        model += ret[(a, t)] <= ret[(a, previous[t])] - 1

    # the start period is given by parameters:
    for a in resources:
        model += rut[(a, period_0)] == min(ub['rut'], rut_init[a])
        model += ret[(a, period_0)] == min(ub['ret'], ret_init[a])

    # fixed periods with maintenance need to be fixed in state:
    for (a, t) in planned_maint:
        model += state[(a, 'M', t)] == 1

    # maintenance duration:
    for (a, t1, t2) in att:
        model += state[(a, 'M', t2)] >= start[(a, t1)]

    # only maintenance state if started
    # not sure if this constraint is necessary.
    for (a, t2) in at:
        model += pl.lpSum(start[(a, t1)] for t1 in t1_at2[(a, t2)]) >= state[(a, 'M', t2)]

    # While we decide how to fix the ending of the planning period,
    # we will try get at least the same amount of total rut and ret than
    # at the beginning.
    model += pl.lpSum(ret[(a, last_period)] for a in resources) >= ret_obj
    model += pl.lpSum(rut[(a, last_period)] for a in resources) >= rut_obj

    # SOLVING
    timeLimit = 3000
    gap = 0
    directory_path = \
        '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/{}/'.\
        format(aux.get_timestamp())

    if solver == "GUROBI":
        result = model.solve(pl.GUROBI_CMD(options=conf.config_gurobi(gap, timeLimit, directory_path)))
    elif solver == "CPLEX":
        result = model.solve(pl.CPLEX_CMD(options=conf.config_cplex(gap, timeLimit, directory_path)))
    else:
        result = model.solve(pl.PULP_CBC_CMD(options=conf.config_cbc(gap, timeLimit, directory_path)))

    if result != 1:
        print("Model resulted in non-feasible status")
        return

    _start = {_t: 1 for _t in start if start[_t].value()}

    _state = aux.tup_to_dict(aux.vars_to_tups(state), result_col=1, is_list=False)
    _task = aux.tup_to_dict(aux.vars_to_tups(task), result_col=1, is_list=False)
    _used = {t: used[t].value() for t in used}
    _rut = {t: rut[t].value() for t in rut}
    _ret = {t: ret[t].value() for t in ret}

    solution = {
        'state': _state,
        'task': _task,
        'used': _used,
        'rut': _rut,
        'ret': _ret
    }

    aux.export_solution(directory_path, solution, name="data_out")
    return solution


if __name__ == "__main__":
    model_data = di.get_model_data()
    codes = aux.get_property_from_dic(model_data['resources'], 'code')
    codes_inv = {value: key for key, value in codes.items()}
    historic_data = di.generate_solution_from_source()
    historic_data_n = {
        (codes_inv[code], month): value for (code, month), value in historic_data.items()\
        if code in codes_inv
    }
    previous_states = {key: 'M' for key, value in historic_data_n.items()
                       if int(str(value).startswith('V'))
                       }
    solve_with_states(model_data, previous_states, solver="CPLEX")

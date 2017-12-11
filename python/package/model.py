import pulp as pl
import package.aux as aux
import package.data_input as di
import package.config as conf
import numpy as np
import package.tests as test

######################################################
# TODO: add minimum mission duration assignment
# TODO: change solution format to json.
# TODO: export file with details such as configuration and comments

def model_no_states():
    return


def solve_with_states(model_data, options=None):
    """
    :param model_data: data to consruct and solve model. taken from get_model_data()
    :param options: dictionary with parameters such as solver, time, gap, etc.
    :return: solution of solved model
    """
    # resources_data = model_data['resources']
    # resources = l['resources']
    # periods = l['periods']
    # duration = param_data['maint_duration']
    # previous_states = aux.get_property_from_dic(resources_data, 'states')
    param_data = model_data['parameters']
    task_data = model_data['tasks']
    l = get_domains_sets(model_data)
    ub = get_bounds(model_data)
    last_period = param_data['end']
    consumption = aux.get_property_from_dic(task_data, 'consumption')  # rh. hours per period.
    requirement = aux.get_property_from_dic(task_data, 'num_resource')  # rr. aircraft per period.
    ret_init = get_initial_state(model_data, "elapsed")
    rut_init = get_initial_state(model_data, "used")
    ret_obj = sum(ret_init[a] for a in l['resources'])
    rut_obj = sum(rut_init[a] for a in l['resources'])

    # number of resources in missions per month:
    num_resource_working = {t: 0 for t in l['periods']}
    for (v, t) in l['vt']:
        num_resource_working[t] += requirement[v]

    # VARIABLES:
    # binary:
    task = pl.LpVariable.dicts("task", l['avt'], 0, 1, pl.LpInteger)
    start = pl.LpVariable.dicts("start", l['at_start'], 0, 1, pl.LpInteger)
    state = pl.LpVariable.dicts("state", l['ast'], 0, 1, pl.LpInteger)

    # numeric:
    ret = pl.LpVariable.dicts("ret", l['at0'], 0, ub['ret'], pl.LpContinuous)
    rut = pl.LpVariable.dicts("rut", l['at0'], 0, ub['rut'], pl.LpContinuous)
    used = pl.LpVariable.dicts("used", l['at'], 0, ub['used'], pl.LpContinuous)

    # objective function:
    max_unavail = pl.LpVariable("max_unavail")
    max_maint = pl.LpVariable("max_maint")

    # MODEL
    model = pl.LpProblem("MFMP_v0001", pl.LpMinimize)

    # OBJECTIVE:
    model += max_unavail + max_maint

    # CONSTRAINTS:
    for t in l['periods']:
        # objective: maintenance (commented because not in use)
        model += pl.lpSum(state[(a, 'M', t)] for a in l['a_t'][t]) <= max_maint
        # objective: availability
        model += pl.lpSum(state[(a, 'M', t)] for a in l['a_t'][t]) + num_resource_working[t] <= max_unavail

    # num resources:
    for (v, t) in l['a_vt']:
        model += pl.lpSum(task[(a, v, t)] for a in l['a_vt'][(v, t)]) >= requirement[v]

    # max one task per period or no-task state:
    for (a, t) in l['v_at']:
        model += pl.lpSum(task[(a, v, t)] for v in l['v_at'][(a, t)]) + \
                 pl.lpSum(state[(a, s, t)] for s in l['states']) <= 1

    # used time, two options:
    # maybe set equal?
    # not sure which one is better, both?
    for (a, t) in l['v_at']:
        model += used[(a, t)] >= pl.lpSum(task[(a, v, t)] * consumption[v] for v in l['v_at'][(a, t)])
    for (a, v, t) in l['avt']:
        model += used[(a, t)] >= task[(a, v, t)] * consumption[v]

    # remaining used time calculations:
    # remaining elapsed time calculations:
    for (a, t) in l['at']:
        if (a, t) in l['at_start']:
            # We only increase the remainders if in that month we could start a maintenance
            model += rut[(a, t)] <= rut[(a, l["previous"][t])] - used[(a, t)] + ub['rut'] * start[(a, t)]
            model += ret[(a, t)] <= ret[(a, l["previous"][t])] - 1 + ub['ret'] * start[(a, t)]
        else:
            # if that month we know we're not starting a maintenance... it's just decreasing:
            model += rut[(a, t)] <= rut[(a, l["previous"][t])] - used[(a, t)]
            model += ret[(a, t)] <= ret[(a, l["previous"][t])] - 1

    # the start period is given by parameters:
    for a in l['resources']:
        model += rut[(a, l['period_0'])] == rut_init[a]
        model += ret[(a, l['period_0'])] == ret_init[a]

    # fixed periods with maintenance need to be fixed in state:
    for (a, t) in l['at_maint']:
        model += state[(a, 'M', t)] == 1

    # maintenance duration:
    for (a, t1, t2) in l['att']:
        model += state[(a, 'M', t2)] >= start[(a, t1)]

    # only maintenance state if started
    for (a, t2), t1list in l['t1_at2'].items():
        if (a, t2) not in l['at_maint']:
            # Only constraint the periods where we do not fix the maintenance:
            model += pl.lpSum(start[(a, t1)] for t1 in t1list) >= state[(a, 'M', t2)]

    # While we decide how to fix the ending of the planning period,
    # we will try get at least the same amount of total rut and ret than
    # at the beginning.
    model += pl.lpSum(ret[(a, last_period)] for a in l['resources']) >= ret_obj
    model += pl.lpSum(rut[(a, last_period)] for a in l['resources']) >= rut_obj

    # SOLVING
    default_options = {
        'timeLimit': 300
        , 'gap': 0
        , 'solver': "GUROBI"
        , 'directory_path': \
            '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/{}/'. \
                format(aux.get_timestamp())
    }
    if options is None:
        options = {}

    options = default_options.update(options)
    config = conf.Config(options)

    if options['solver'] == "GUROBI":
        result = model.solve(pl.GUROBI_CMD(options=config.config_gurobi()))
    elif options['solver'] == "CPLEX":
        result = model.solve(pl.CPLEX_CMD(options=config.config_cplex()))
    else:
        result = model.solve(pl.PULP_CBC_CMD(options=config.config_cbc()))

    if result != 1:
        print("Model resulted in non-feasible status")
        return

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

    di.export_data(directory_path, model_data, name="data_in", file_type='pickle')
    di.export_data(directory_path, model_data, name="data_in", file_type='json')
    di.export_data(directory_path, solution, name="data_out")
    return solution


def get_bounds(model_data):
    param_data = model_data['parameters']
    task_data = model_data['tasks']

    # maximal bounds on continuous variables:
    max_elapsed_time = param_data['max_elapsed_time']  # me. in periods
    max_used_time = param_data['max_used_time']  # mu. in hours of usage
    consumption = aux.get_property_from_dic(task_data, 'consumption')  # rh. hours per period.

    return {
        'ret': max_elapsed_time,
        'rut': max_used_time,
        'used': max(consumption.values())
    }


def get_domains_sets(model_data):
    states = ['M']
    # dtype_at = [('V', '<U6'), ('D', 'U7')]

    param_data = model_data['parameters']

    # periods
    first_period, last_period = model_data['parameters']['start'], model_data['parameters']['end']
    periods = aux.get_months(first_period, last_period)
    period_0 = aux.get_prev_month(model_data['parameters']['start'])
    periods_0 = [period_0] + periods
    periods_pos = {periods[pos]: pos for pos in range(len(periods))}
    previous = {period: periods_0[periods_pos[period]] for period in periods}

    # tasks
    task_data = model_data['tasks']
    tasks = list(model_data['tasks'].keys())
    start_time = aux.get_property_from_dic(task_data, 'start')
    end_time = aux.get_property_from_dic(task_data, 'end')
    candidates = aux.get_property_from_dic(task_data, 'candidates')

    # resources
    resources_data = model_data['resources']
    resources = list(resources_data.keys())
    duration = param_data['maint_duration']
    previous_states = aux.get_property_from_dic(resources_data, 'states')

    """
    Indentation means "includes the following:".
    The elements represent a given combination resource-period.
    at0: all, including the previous period.
        at: all.                                                    => 'used'
            at_mission: a mission is assigned (fixed)               => 'assign'
            at_free: nothing is fixed                               => 'assign' and 'state'
                at_free_start: can start a maintenance              => 'start'
            at_maint: maintenance is assigned (fixed)               => 'state' 
                at_start: start of maintenance is assigned (fixed). => 'start'
    """

    # TODO: solve numpy arrays issues with lists. Use np.setdiff1d

    at = [(a, t) for a in resources for t in periods]
    at0 = [(a, period_0) for a in resources] + at
    at_mission = []  # to be implemented
    at_start = []  # to be implemented
    at_maint = get_fixed_maintenances(model_data)
    at_free = [(a, t) for (a, t) in at if (a, t) not in at_maint + at_mission]
    at_free_start = [(a, t) for (a, t) in at_free]

    vt = [(v, t) for v in tasks for t in periods if start_time[v] <= t <= end_time[v]]
    avt = [(a, v, t) for a in resources for (v, t) in vt
           if a in candidates[v]
           if (a, t) in at_free + at_mission]
    ast = [(a, s, t) for (a, t) in at_free + at_maint for s in states]
    att = [(a, t1, t2) for (a, t1) in at_start + at_free_start for t2 in periods if
           periods_pos[t1] <= periods_pos[t2] <= periods_pos[t1] + duration - 1]

    a_t = aux.tup_to_dict(at, result_col=0, is_list=True)
    a_vt = aux.tup_to_dict(avt, result_col=0, is_list=True)
    v_at = aux.tup_to_dict(avt, result_col=1, is_list=True)
    t1_at2 = aux.tup_to_dict(att, result_col=1, is_list=True)

    return {
     'periods'          :  periods
    ,'period_0'         :  period_0
    ,'periods_0'        :  periods_0
    ,'periods_pos'      :  periods_pos
    ,'previous'         :  previous
    ,'tasks'            :  tasks
    ,'candidates'       :  candidates
    ,'resources'        :  resources
    ,'planned_maint'    :  at_maint
    ,'states'           :  states
    ,'vt'               :  vt
    ,'avt'              :  avt
    ,'at'               :  at
    ,'at_maint'         :  at_maint
    ,'ast'              :  ast
    ,'at_start'         :  at_start + at_free_start
    ,'at0'              :  at0
    ,'att'              :  att
    ,'a_t'              :  a_t
    ,'a_vt'             :  a_vt
    ,'v_at'             :  v_at
    ,'t1_at2'           :  t1_at2
    }


def get_initial_state(model_data, time_type):
    if time_type not in ["elapsed", "used"]:
        raise KeyError("Wrong type in time_type parameter: elapsed or used only")

    key_initial = "initial_" + time_type
    key_max = "max_" + time_type + "_time"
    param_resources = model_data['resources']
    rt_max = model_data['parameters'][key_max]

    rt_read = aux.get_property_from_dic(param_resources, key_initial)

    # we also check if the resources is currently in maintenance.
    # If it is: we assign the rt_max (according to convention).
    res_in_maint = set([res for res, period in get_fixed_maintenances(model_data)])
    rt_fixed = {a: rt_max for a in param_resources if a in res_in_maint}

    rt_init = {a: rt_max for a in param_resources}
    rt_init.update(rt_read)
    rt_init.update(rt_fixed)

    rt_init = {k: min(rt_max, v) for k, v in rt_init.items()}

    return rt_init


def get_fixed_maintenances(model_data):
    previous_states = aux.get_property_from_dic(model_data['resources'], "states")
    first_period = model_data['parameters']['start']
    duration = model_data['parameters']['maint_duration']

    last_maint = {}
    planned_maint = []
    previous_states_n = {key: [key2 for key2 in value if value[key2] == 'M']
                         for key, value in previous_states.items()}

    # after initialization, we search for the scheduled maintenances that:
    # 1. do not continue the maintenance of the previous month
    # 2. happen in the last X months before the start of the planning period.
    for res in previous_states_n:
        _list = list(previous_states_n[res])
        _list_n = [period for period in _list if aux.get_prev_month(period) not in _list
                   if aux.shift_month(first_period, -duration) < period < first_period]
        if not len(_list_n):
            continue
        last_maint[res] = max(_list_n)
        finish_maint = aux.shift_month(last_maint[res], duration - 1)
        for period in aux.get_months(first_period, finish_maint):
            planned_maint.append((res, period))
    return planned_maint


if __name__ == "__main__":
    model_data = di.get_model_data()
    historic_data = di.generate_solution_from_source()
    model_data = di.combine_data_states(model_data, historic_data)

    # this is for testing purposes:
    num_max_periods = 20
    model_data['parameters']['end'] = \
        aux.shift_month(model_data['parameters']['start'], num_max_periods)
    forbidden_tasks = ['O8']
    model_data['tasks'] = \
        {k: v for k, v in model_data['tasks'].items() if k not in forbidden_tasks}
    # this was for testing purposes

    # solving part:
    solution = solve_with_states(model_data)

    testing = test.CheckModel(model_data, solution)

    result = testing.check_task_num_resources()  # this fails.
    #  testing...

    # import pprint
    # pp = pprint.PrettyPrinter()
    # pp.pprint({k: len(v) for k, v in l.items()})
    # {k:v for k,v in rut_init.items() if v<0}

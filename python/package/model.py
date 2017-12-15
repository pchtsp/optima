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
# TODO: test different weights on OF
# TODO: check problems with model without states


def model_no_states(instance, options=None):
    l = instance.get_domains_sets()
    ub = instance.get_bounds()
    last_period = instance.get_param('end')
    consumption = instance.get_tasks('consumption')
    requirement = instance.get_tasks('num_resource')
    ret_init = instance.get_initial_state("elapsed")
    rut_init = instance.get_initial_state("used")
    ret_obj = sum(ret_init[a] for a in l['resources'])
    rut_obj = sum(rut_init[a] for a in l['resources'])
    num_resource_working = instance.get_total_period_needs()
    num_resource_maint = aux.fill_dict_with_default(instance.get_total_fixed_maintenances(), l['periods'])

    # VARIABLES:
    # binary:
    task = pl.LpVariable.dicts("task", l['avt'], 0, 1, pl.LpInteger)
    start = pl.LpVariable.dicts("start", l['at_start'], 0, 1, pl.LpInteger)

    # numeric:
    ret = pl.LpVariable.dicts("ret", l['at0'], 0, ub['ret'], pl.LpContinuous)
    rut = pl.LpVariable.dicts("rut", l['at0'], 0, ub['rut'], pl.LpContinuous)

    # objective function:
    max_unavail = pl.LpVariable("max_unavail")
    max_maint = pl.LpVariable("max_maint")

    # MODEL
    model = pl.LpProblem("MFMP_v0001", pl.LpMinimize)

    # OBJECTIVE:
    model += max_unavail + max_maint

    # CONSTRAINTS:
    for t in l['periods']:
        # objective: maintenance
        model += pl.lpSum(start[(a, _t)] for (a, _t) in l['at1_t2'][t] if (a, _t) in l['at_start']) + \
                 num_resource_maint[t] <= max_maint
        # objective: availability
        model += pl.lpSum(start[(a, _t)] for (a, _t) in l['at1_t2'][t] if (a, _t) in l['at_start']) + \
                 num_resource_working[t] + \
                 num_resource_maint[t] <= max_unavail
    # num resources:
    for (v, t) in l['a_vt']:
        model += pl.lpSum(task[(a, v, t)] for a in l['a_vt'][(v, t)]) == requirement[v]
    # max one task per period or no-task state:
    for (a, t) in l['v_at']:
        model += pl.lpSum(task[(a, v, t)] for v in l['v_at'][(a, t)]) + \
                 pl.lpSum(start[(a, _t)] for _t in l['t1_at2'][(a, t)] if (a, _t) in l['at_start']) <= 1
    # remaining used time calculations:
    # remaining elapsed time calculations:
    for (a, t) in l['at']:
        if (a, t) in l['at_start']:
            # We only increase the remainders if in that month we could start a maintenance
            model += rut[(a, t)] <= rut[(a, l["previous"][t])] - \
                                    pl.lpSum(task[(a, v, t)] * consumption[v] for v in l['v_at'][(a, t)]) + \
                                    ub['rut'] * start[(a, t)]
            model += ret[(a, t)] <= ret[(a, l["previous"][t])] - 1 + \
                                    ub['ret'] * start[(a, t)]

            model += rut[(a, t)] >= ub['rut'] * start[(a, t)]
            model += ret[(a, t)] >= ub['ret'] * start[(a, t)]
        elif (a, t) in l['v_at']:
            # if that month we know we're not starting a maintenance... it's just decreasing:
            model += rut[(a, t)] <= rut[(a, l["previous"][t])] - \
                                    pl.lpSum(task[(a, v, t)] * consumption[v] for v in l['v_at'][(a, t)])
            model += ret[(a, t)] <= ret[(a, l["previous"][t])] - 1
        else:
            # if that month we know we're not making a mission...
            model += rut[(a, t)] <= rut[(a, l["previous"][t])]
            model += ret[(a, t)] <= ret[(a, l["previous"][t])] - 1

    # the start period is given by parameters:
    for a in l['resources']:
        model += rut[(a, l['period_0'])] == rut_init[a]
        model += ret[(a, l['period_0'])] == ret_init[a]

    # While we decide how to fix the ending of the planning period,
    # we will try get at least the same amount of total rut and ret than
    # at the beginning.
    model += pl.lpSum(ret[(a, last_period)] for a in l['resources']) >= ret_obj
    model += pl.lpSum(rut[(a, last_period)] for a in l['resources']) >= rut_obj

    # SOLVING
    if options is None:
        options = {}

    default_options = {
        'timeLimit': 300
        , 'gap': 0
        , 'solver': "GUROBI"
        , 'path':
            '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/{}/'.
                format(aux.get_timestamp())
    }

    # the following merges the two configurations (replace into):
    options = {**default_options, **options}
    config = conf.Config(options)

    if options['solver'] == "GUROBI":
        result = model.solve(pl.GUROBI_CMD(options=config.config_gurobi()))
    elif options['solver'] == "CPLEX":
        result = model.solve(pl.CPLEX_CMD(options=config.config_cplex()))
    else:
        result = model.solve(pl.PULP_CBC_CMD(options=config.config_cbc()))

    if result != 1:
        print("Model resulted in non-feasible status")
        return None

    # _state = aux.tup_to_dict(aux.vars_to_tups(state), result_col=1, is_list=False)
    _task = aux.tup_to_dict(aux.vars_to_tups(task), result_col=1, is_list=False)
    _start = aux.vars_to_tups(start)
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


def solve_with_states(instance, options=None):
    """
    :param instance: object with data to solve model. taken from instance.py
    :param options: dictionary with parameters such as solver, time, gap, etc.
    :return: solution object of solved model. taken from solution.py
    """
    # resources_data = model_data['resources']
    # resources = l['resources']
    # periods = l['periods']
    # duration = param_data['maint_duration']
    # previous_states = aux.get_property_from_dic(resources_data, 'states')
    l = instance.get_domains_sets()
    ub = instance.get_bounds()
    last_period = instance.get_param('end')
    consumption = instance.get_tasks('consumption')
    requirement = instance.get_tasks('num_resource')
    ret_init = instance.get_initial_state("elapsed")
    rut_init = instance.get_initial_state("used")
    ret_obj = sum(ret_init[a] for a in l['resources'])
    rut_obj = sum(rut_init[a] for a in l['resources'])
    num_resource_working = instance.get_total_period_needs()

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
        # objective: maintenance
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
    if options is None:
        options = {}

    default_options = {
        'timeLimit': 300
        , 'gap': 0
        , 'solver': "GUROBI"
        , 'path':
            '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/{}/'.
                format(aux.get_timestamp())
    }

    # the following merges the two configurations (replace into):
    options = {**default_options, **options}
    config = conf.Config(options)

    if options['solver'] == "GUROBI":
        result = model.solve(pl.GUROBI_CMD(options=config.config_gurobi()))
    elif options['solver'] == "CPLEX":
        result = model.solve(pl.CPLEX_CMD(options=config.config_cplex()))
    else:
        result = model.solve(pl.PULP_CBC_CMD(options=config.config_cbc()))

    if result != 1:
        print("Model resulted in non-feasible status")
        return None

    _state = aux.tup_to_dict(aux.vars_to_tups(state), result_col=1, is_list=False)
    _task = aux.tup_to_dict(aux.vars_to_tups(task), result_col=1, is_list=False)
    _used = {t: used[t].value() for t in used}
    _rut = {t: rut[t].value() for t in rut}
    _ret = {t: ret[t].value() for t in ret}
    _start = {k: 1 for k in aux.vars_to_tups(start)}

    solution_data_pre = {
        'state': _state,
        'task': _task,
        'aux': {
            'start': _start,
            'rut': _rut,
            'ret': _ret,
            'used': _used
        }
    }

    solution_data = {k: aux.dicttup_to_dictdict(v)
                     for k, v in solution_data_pre.items() if k != "aux"}
    solution_data['aux'] = {k: aux.dicttup_to_dictdict(v)
                            for k, v in solution_data_pre['aux'].items()}
    solution = sol.Solution(solution_data)

    return solution


if __name__ == "__main__":
    model_data = di.get_model_data()
    historic_data = di.generate_solution_from_source()
    model_data = di.combine_data_states(model_data, historic_data)

    # this is for testing purposes:
    num_max_periods = 20
    model_data['parameters']['end'] = \
        aux.shift_month(model_data['parameters']['start'], num_max_periods)
    forbidden_tasks = ['O8']  # this task has less candidates than what it asks.
    # forbidden_tasks = []
    model_data['tasks'] = \
        {k: v for k, v in model_data['tasks'].items() if k not in forbidden_tasks}
    # this was for testing purposes

    instance = inst.Instance(model_data)

    options = {
        'timeLimit': 300
        , 'gap': 0
        , 'solver': "CPLEX"
        , 'path':
            '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/{}/'.
                format(aux.get_timestamp())
        ,"model": "no_states"
    }

    # solving part:
    # solution = solve_with_states(instance, options)
    solution = model_no_states(instance, options)

    di.export_data(options['path'], instance.data, name="data_in", file_type='pickle')
    di.export_data(options['path'], instance.data, name="data_in", file_type='json')
    di.export_data(options['path'], solution.data, name="data_out", file_type='pickle')
    di.export_data(options['path'], solution.data, name="data_out", file_type='json')
    di.export_data(options['path'], options, name="options", file_type='json')


    # testing = test.CheckModel(instance, solution)
    # result = testing.check_task_num_resources()
    #  testing...

    # import pprint
    # pp = pprint.PrettyPrinter()
    # pp.pprint({k: len(v) for k, v in l.items()})
    # {k:v for k,v in rut_init.items() if v<0}




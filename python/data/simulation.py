# /usr/bin/python3

import data.dates as aux
import random as rn
import math
import numpy as np
import pytups.superdict as sd
import string as st

def empty_data():
    return {
        'parameters': {
            'maint_weight': 0
            , 'unavail_weight': 0
            , 'max_used_time': 0
            , 'max_elapsed_time': 0
            , 'elapsed_time_size': 0
            , 'maint_duration': 0
            , 'maint_capacity': 0
            , 'start': '2018-01'
            , 'end': '2018-01'
            , 'min_usage_period': 0
        },
        'tasks': {
            task: {
                'start': '2018-01'
                , 'end': '2018-01'
                , 'consumption': 0
                , 'num_resource': 0
                , 'type_resource': 0
                , 'matricule': ''
                , 'min_assign': 1
                , 'capacities': []
            } for task in range(1)
        },
        'resources': {
            resource: {
                'initial_used': 0
                , 'initial_elapsed': 0
                , 'code': ''
                , 'type': ''
                , 'capacities': []
                , 'states': {t: '' for t in range(0)}
            } for resource in range(1)
        }
    }


def create_dataset(options):
    sim_data = options['simulation']

    seed = sim_data.get('seed', None)
    if not seed:
        seed = math.ceil(rn.random() * 100000)
        sim_data['seed'] = seed
    rn.seed(seed)
    np.random.seed(seed)

    data_input = {}
    data_input['parameters'] = generate_parameters(sim_data, options)
    data_input['tasks'] = generate_tasks(data_input)
    data_input['resources'] = generate_resources(data_input)

    return data_input

def generate_parameters(sim_data, options):

    d_param = {**sim_data}
    d_param['start'] = options['start']
    d_param['num_period'] = options['num_period']
    num_resources = d_param['num_resources']
    d_param['maint_capacity'] = math.ceil(num_resources * d_param['perc_capacity'])
    d_param['end'] = aux.shift_month(d_param['start'], options['num_period'] - 1)

    return d_param

def generate_resources(data_input):

    d_param = data_input['parameters']
    num_resources = d_param['num_resources']
    resources = [str(r) for r in range(num_resources)]

    res_types = generate_resource_types(d_param, data_input)
    res_capacities = generate_resource_capacites(data_input, res_types)
    res_in_maint, res_maint_init = generate_resources_in_maint(d_param)
    res_task_init = gerenate_initial_assignments(d_param, data_input, res_in_maint, res_capacities)
    initial_elapsed, initial_used = generate_initial_status(d_param, res_in_maint, res_task_init)

    _dict = {
        str(res): {
            'initial_elapsed': initial_elapsed[res]
            , 'initial_used': initial_used[res]
            , 'code': ''  # this is aesthetic
            , 'type': res_types[res]
            , 'capacities': list(res_capacities[res])
            , 'states': {**res_task_init[res], **res_maint_init[res]}
        } for res in resources
    }
    return sd.SuperDict.from_dict(_dict)

def generate_resources_in_maint(d_param):
    # Here we simulate the initial state of resources.
    # First of all we decide which resources are in maintenance
    # and then the periods they have been under maintenance

    start_period = d_param['start']
    maint_duration = d_param['maint_duration']
    num_resources = d_param['num_resources']
    resources = [str(r) for r in range(num_resources)]
    perc_in_maint = d_param['perc_in_maint']

    res_in_maint = np.random.choice(resources,
                                    math.floor(num_resources*perc_in_maint),
                                    replace=False)
    res_maint_init = {
        r: {aux.shift_month(start_period, - n - 1): 'M'
            for n in range(rn.randrange(maint_duration) + 1)}
        for r in res_in_maint
    }
    res_maint_init = sd.SuperDict(res_maint_init).\
        fill_with_default(resources, default={})
    return res_in_maint, res_maint_init


def generate_resource_types(d_param, data_input):

    start_period = d_param['start']
    num_parallel_tasks = d_param['num_parallel_tasks']
    d_tasks = data_input['tasks']
    last_period = d_param['end']
    num_resources = d_param['num_resources']
    resources = [str(r) for r in range(num_resources)]

    period_type_num_resource = {
        t: {p: 0 for p in aux.get_months(start_period, last_period)}
        for t in range(num_parallel_tasks)
    }
    for k, v in d_tasks.items():
        t = v['type_resource']
        for p in aux.get_months(v['start'], v['end']):
            period_type_num_resource[t][p] += v['num_resource']

    # we want at least as many resources as the max requirement of any given month
    # for the rest, we randomly assign a type weighted by the needs
    max_res_need_type = sd.SuperDict({k: max(v.values()) for k, v in period_type_num_resource.items()})
    res_types = [t for t, q in max_res_need_type.items() for r in range(q)]
    res_types.extend(
        rn.choices(max_res_need_type.keys_l(),
                   k=len(resources) - len(res_types),
                   weights=max_res_need_type.values_l())
    )
    res_types = sd.SuperDict({k: res_types[i] for i, k in enumerate(resources)})

    return res_types

def generate_resource_capacites(data_input, res_types):

    d_tasks = data_input['tasks']
    # we initialize the capacities with the types of resources
    res_capacities = sd.SuperDict({k: {str(v)} for k, v in res_types.items()})
    # we want to add the special capacities to only some resources.
    # we do this by iterating over tasks and trying to complete the partially able resources.
    # We will start with the tasks that demand the most capacities
    task_content = sorted(d_tasks.items(), key=lambda x: - len(x[1]['capacities']))
    for _task, contents in task_content:
        _capacities = set(contents['capacities'])
        _t_type = str(contents['type_resource'])
        _resources = contents['num_resource']
        # we filter resources that have the task's type
        _res_possible = res_capacities.clean(func=lambda x: _t_type in x)
        # then we see if we already have matching resources:
        _res_ready = _res_possible.clean(func=lambda x: not _capacities.difference(x))
        new_resources = _resources*2 - len(_res_ready)
        if new_resources > 0:
            # we take new_resources random resources from _res_possible.
            # we try to upgrade the ones with the least capacities
            _res_to_upgrade = \
                rn.choices(_res_possible.keys_l(),
                           k=new_resources,
                           weights=[1/len(v) for v in _res_possible.values_l()])
            for k in _res_to_upgrade:
                for cap in _capacities:
                    res_capacities[k].add(cap)
    return res_capacities


def generate_initial_status(d_param, res_in_maint, res_task_init):

    max_used_time = d_param['max_used_time']
    max_elapsed_time = d_param['max_elapsed_time']
    initial_unbalance = d_param['initial_unbalance']
    num_resources = d_param['num_resources']
    resources = [str(r) for r in range(num_resources)]

    # We fill the states according to the initial values already calculated:
    # with a little random unbalance (initial_unbalance)

    # if a resource is doing a mission, we need to give it a little more hours
    init_elapsed = {r: 3 if not len(res_task_init[r]) else max_elapsed_time//2 for r in resources}
    initial_elapsed = {r: rn.randrange(init_elapsed[r], max_elapsed_time) for r in resources}

    initial_elapsed_adj = {k: v + rn.randint(*initial_unbalance) for k, v in initial_elapsed.items()}
    initial_elapsed_adj = {k: min(max(v, 0), max_elapsed_time) for k, v in initial_elapsed_adj.items()}
    initial_used = {k: math.ceil(v / max_elapsed_time * max_used_time) for k, v in initial_elapsed_adj.items()}

    # if resource is in maintenance: we have the status at the max.
    for res in res_in_maint:
        initial_used[res] = max_used_time
        initial_elapsed[res] = max_elapsed_time

    return initial_elapsed, initial_used

def gerenate_initial_assignments(d_param, data_input, res_in_maint, res_capacities):

    num_resources = d_param['num_resources']
    resources = [str(r) for r in range(num_resources)]
    start_period = d_param['start']
    d_tasks = data_input['tasks']
    # Secondly, for the remaining resources, we assign the previous tasks
    # at the beginning of the planning horizon
    _res = [r for r in resources if r not in res_in_maint]
    res_task_init = {}
    starting_tasks = [k for k, v in d_tasks.items() if v['start'] == start_period]

    for j in starting_tasks:
        capacities = set(d_tasks[j]['capacities'])
        # first we choose a number of resources equivalent to the number
        # of resources needed by the task
        # these resources need to be able to do the task
        _res_task = [r for r in _res if not capacities.difference(res_capacities[r])]
        res_to_assign = \
            np.random.choice(_res_task,
                             # min(rn.randrange(d_tasks[j]['num_resource'] + 1), len(_res)),
                             min(d_tasks[j]['num_resource'], len(_res)),
                             replace=False)
        # we take them out of the list of available resources:
        _res = [r for r in _res if r not in res_to_assign]
        # now we decide when was the task assigned:
        min_assign = d_tasks[j]['min_assign']
        for r in res_to_assign:
            # we assumed the resource could have had the task
            # for a time = double the minimum time
            months_in_task = rn.randrange(min_assign)*2
            # if the resource started less than min_assin ago,
            # we fix the task for the following periods
            res_task_init[r] = {
                aux.shift_month(start_period, -n - 1): j
                for n in range(months_in_task)
            }
        if not _res:
            break
    res_task_init = sd.SuperDict(res_task_init).fill_with_default(resources, default={})
    return res_task_init


def generate_tasks(data_input):

    d_param = data_input['parameters']
    last_period = d_param['end']
    num_parallel_tasks = d_param['num_parallel_tasks']
    start_period = d_param['start']
    t_min_assign = d_param['t_min_assign']
    t_required_hours = d_param['t_required_hours']
    t_num_resource = d_param['t_num_resource']
    t_duration = d_param['t_duration']
    perc_add_capacity = d_param['perc_add_capacity']

    # Here we simulate the tasks along the planning horizon.
    # we need to guarantee there are num_parallel_tasks active task
    # at each time.
    task = 0
    t_start = {}
    t_end = {}
    t_type = {}
    for t in range(num_parallel_tasks):
        date = start_period
        while date <= last_period:
            t_type[task] = t
            t_start[task] = date
            date = aux.shift_month(date, rn.randint(*t_duration))
            if date > last_period:
                date = aux.get_next_month(last_period)
            t_end[task] = aux.get_prev_month(date)
            task += 1

    t_capacites = {k: {str(v)} for k, v in t_type.items()}
    optionals = list(st.ascii_uppercase)[::-1]
    for _task in t_capacites:
        if rn.random() < perc_add_capacity:
            t_capacites[_task].add(optionals.pop())


    return sd.SuperDict({
            str(t): {
                'start': t_start[t]
                , 'end': t_end[t]
                , 'consumption': math.floor(np.random.triangular(*t_required_hours))
                , 'num_resource': rn.randint(*t_num_resource)
                , 'type_resource': t_type[t]
                , 'matricule': ''  # this is aesthetic
                , 'min_assign': rn.choice(t_min_assign)
                , 'capacities': list(t_capacites[t])
            } for t in t_start
        })

if __name__ == "__main__":
    import package.params as params
    create_dataset(params.OPTIONS)
    pass
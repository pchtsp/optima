# /usr/bin/python3

# This simulation complies to the format returned in the function
# get_model_data in the file data_input.py

import package.auxiliar as aux
import random as rn
import math
import numpy as np
import package.superdict as sd


def empty_data():
    return {
        'parameters': {
            'maint_weight': 0
            , 'unavail_weight': 0
            , 'max_used_time': 0
            , 'max_elapsed_time': 0
            , 'min_elapsed_time': 0
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
    # TODO: clusters, capacities, types, etc.
    # param_list = ['start', 'num_period']
    # options = sd.SuperDict.from_dict(options)
    sim_data = options['simulation']

    data_input = {}
    d_param = data_input['parameters'] = {
        **sim_data,
    }
    d_param['start'] = options['start']
    d_param['num_period'] = options['num_period']
    seed = sim_data.get('seed', None)
    if not seed:
        seed = math.ceil(rn.random() * 100000)
        sim_data['seed'] = seed
    rn.seed(seed)
    np.random.seed(seed)

    num_parallel_tasks = d_param['num_parallel_tasks']
    start_period = d_param['start']
    num_resources = d_param['num_resources']
    maint_duration = d_param['maint_duration']
    max_used_time = d_param['max_used_time']
    max_elapsed_time = d_param['max_elapsed_time']
    resources = [str(r) for r in range(num_resources)]

    # The following are fixed options, not arguments for the scenario:
    t_min_assign = d_param['t_min_assign']
    t_required_hours = d_param['t_required_hours']
    t_num_resource = d_param['t_num_resource']
    t_duration = d_param['t_duration']
    perc_in_maint = d_param['perc_in_maint']

    d_param['min_elapsed_time'] = max_elapsed_time - d_param['elapsed_time_size']
    d_param['maint_capacity'] = math.ceil(num_resources * d_param['perc_capacity'])
    last_period = d_param['end'] = aux.shift_month(d_param['start'], options['num_period'] - 1)

    # Here we simulate the tasks along the planning horizon.
    # we need to guarantee there are num_parallel_tasks active task
    # at each time.
    task = 0
    t_start = {}
    t_end = {}
    starting_tasks = []
    for t in range(num_parallel_tasks):
        date = start_period
        starting_tasks.append(str(task))
        while date < last_period:
            t_start[task] = date
            date = aux.shift_month(date, rn.randint(*t_duration))
            if date > last_period:
                date = last_period
            t_end[task] = aux.get_prev_month(date)
            task += 1

    d_tasks = data_input['tasks'] = {
            str(task): {
                'start': t_start[task]
                , 'end': t_end[task]
                , 'consumption': rn.choice(t_required_hours)
                , 'num_resource': rn.randint(*t_num_resource)
                , 'type_resource': ''  # TODO: capacities
                , 'matricule': ''  # this is aesthetic
                , 'min_assign': rn.choice(t_min_assign)
                , 'capacities': []  # TODO: capacities
            } for task in range(task)
        }

    # Here we simulate the initial state of resources.
    # First of all we decide which resources are in maintenance
    # and then the periods they have been under maintenance
    res_in_maint = np.random.choice(resources,
                                    math.ceil(num_resources*perc_in_maint),
                                    replace=False)
    res_maint_init = {
        r: {aux.shift_month(start_period, - n - 1): 'M'
            for n in range(rn.randrange(maint_duration) + 1)}
        for r in res_in_maint
    }
    res_maint_init = sd.SuperDict(res_maint_init).\
        fill_with_default(resources, default={})

    # Secondly, for the remaining resources, we assign tasks
    _res = [r for r in resources if r not in res_in_maint]
    res_task_init = {}
    for j in starting_tasks:
        # first we choose a number of resources equivalent to the number
        # of resources needed by the task
        res_to_assign = \
            np.random.choice(_res,
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
    res_task_init = sd.SuperDict(res_task_init). \
        fill_with_default(resources, default={})
    # We fill the states according to the initial values already calculated:

    # TODO: this adjustment of +- 6 months is arbitrary. It should exist in the configuration in some way.
    initial_elapsed = {r: rn.randrange(0, max_elapsed_time) for r in resources}
    initial_elapsed_adj = {k: v + rn.randint(-6, 6) for k, v in initial_elapsed.items()}
    initial_elapsed_adj = {k: min(max(v, 0), max_elapsed_time) for k, v in initial_elapsed_adj.items()}
    data_input['resources'] = {
        str(res): {
            'initial_elapsed':
                initial_elapsed[res]
                if res not in res_in_maint else max_elapsed_time
            , 'initial_used':
                math.ceil(initial_elapsed_adj[res] / max_elapsed_time * max_used_time)
                if res not in res_in_maint else max_used_time
            , 'code': ''  # this is aesthetic
            , 'type': ''  # TODO: capacities
            , 'capacities': []  # TODO: capacities
            , 'states': {**res_task_init[res], **res_maint_init[res]}
        } for res in resources
    }

    return data_input


if __name__ == "__main__":
    import pprint as pp
    import package.params as params
    create_dataset(params)
    pass
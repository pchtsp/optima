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


def create_dataset():
    # TODO: clusters, capacities, types, etc.
    # TODO: the following params should be arguments:
    num_resources = 50
    num_parallel_tasks = 4
    num_periods = 60
    start_period = '2019-01'
    last_period = aux.shift_month(start_period, num_periods-1)
    maint_duration = 4
    max_usage_hours = 800
    max_elapsed_periods = 40
    min_elapsed_periods = max_elapsed_periods - 20
    min_flight_hours_period = 15
    perc_capacity = 0.2

    # The following are fixed options, not arguments for the scenario:
    t_min_assign = [2, 3, 6]
    t_required_hours = [50, 60, 70, 80]
    t_num_resource = (2, 5)
    t_duration = (12, 36)
    perc_in_maint = 0.1

    data_input = {}

    data_input['parameters'] = {
        'maint_weight': 0
        , 'unavail_weight': 0
        , 'max_used_time': max_usage_hours
        , 'max_elapsed_time': max_elapsed_periods
        , 'min_elapsed_time': min_elapsed_periods
        , 'maint_duration': maint_duration
        , 'maint_capacity': math.ceil(num_resources * perc_capacity)
        , 'start': start_period
        , 'end': aux.shift_month(start_period, num_periods - 1)
        , 'min_usage_period': min_flight_hours_period
    }

    # Here we simulate the tasks along the planning horizon.
    # we need to guarantee there are num_parallel_tasks active task
    # at each time.
    task = 0
    t_start = {}
    t_end = {}
    starting_tasks = []
    for t in range(num_parallel_tasks):
        date = start_period
        starting_tasks.append(task)
        while date < last_period:
            t_start[task] = date
            date = aux.shift_month(date, rn.randint(*t_duration))
            if date > last_period:
                date = last_period
            t_end[task] = aux.get_prev_month(date)
            task += 1

    d_tasks = data_input['tasks'] = {
            task: {
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
    res_in_maint = np.random.choice(range(num_resources),
                                    math.ceil(num_resources*perc_in_maint),
                                    replace=False)
    res_maint_init = {
        r: {aux.shift_month(start_period, - n - 1): 'M'
            for n in range(rn.randrange(maint_duration) + 1)}
        for r in res_in_maint
    }
    res_maint_init = sd.SuperDict(res_maint_init).\
        fill_with_default(range(num_resources), default={})

    # Secondly, for the remaining resources, we assign tasks
    _res = [r for r in range(num_resources) if r not in res_in_maint]
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
        fill_with_default(range(num_resources), default={})
    # TODO: adapt states instead of using fixed_maintenances and fixed_tasks
    # We fill the states according to the initial values already calculated:
    data_input['resources'] = {
        res: {
            'initial_used':
                rn.randrange(0, max_usage_hours)
                if res not in res_in_maint else max_usage_hours
            , 'initial_elapsed':
                rn.randrange(0, max_elapsed_periods)
                if res not in res_in_maint else max_elapsed_periods
            , 'code': ''  # this is aesthetic
            , 'type': ''  # TODO: capacities
            , 'capacities': []  # TODO: capacities
            , 'states': {**res_task_init[res], **res_maint_init[res]}
        } for res in range(num_resources)
    }

    return data_input


if __name__ == "__main__":
    import pprint as pp
    create_dataset()
    pass
# /usr/bin/python3

# This simulation complies to the format returned in the function
# get_model_data in the file data_input.py

import package.auxiliar as aux
import random as rn
import math
import numpy as np


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
                , 'capacities': []
                , 'fixed_maintenances': []
                , 'fixed_tasks': [()]
            } for resource in range(1)
        }
    }


def create_dataset():
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
                , 'matricule': ''
                , 'min_assign': rn.choice(t_min_assign)
                , 'capacities': []  # TODO: capacities
            } for task in range(task)
        }

    # Here we simulate the initial state of resources.
    # First of all we decide which resources are in maintenance
    # and the amount of periods they are still in maintenance
    res_in_maint = np.random.choice(range(num_resources),
                                    math.ceil(num_resources*perc_in_maint),
                                    replace=False)
    res_maint_init = {
        r: [aux.shift_month(start_period, n - 1)
            for n in range(rn.randrange(maint_duration) + 1)]
        for r in res_in_maint
    }

    # Secondly, for the remaining resources, we assign tasks
    _res = [r for r in range(num_resources) if r not in res_in_maint]
    res_task_init = {}
    for j in starting_tasks:
        res_to_assign = \
            np.random.choice(_res,
                             min(rn.randrange(d_tasks[j]['num_resource'] + 1), len(_res)),
                             replace=False)
        _res = [r for r in _res if r not in res_to_assign]
        for r in res_to_assign:
            res_task_init[r] = [(j, aux.shift_month(start_period, n - 1))
                            for n in range(rn.randrange(d_tasks[j]['min_assign'])+1)
                            ]
        if not _res:
            break

    # We fill the states according to the initial values already calculated:
    data_input['resources'] = {
        res: {
            'initial_used':
                rn.randrange(0, max_usage_hours)
                if res not in res_in_maint else max_usage_hours
            , 'initial_elapsed':
                rn.randrange(0, max_elapsed_periods)
                if res not in res_in_maint else max_elapsed_periods
            , 'code': ''
            , 'capacities': []  # TODO: capacities
            , 'fixed_maintenances': res_maint_init[res] if res in res_in_maint else []
            , 'fixed_tasks': res_task_init[res] if res in res_task_init else []
        } for res in range(num_resources)
    }

    return data_input

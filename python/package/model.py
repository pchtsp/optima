import pulp as pl
from package.data_input import generate_data_from_source
from package.aux import get_months, get_prev_month, clean_dict, tup_to_dict, vars_to_tups
import re
import pandas as pd
import package.aux as aux
import os
import package.data_input as di

######################################################

# arguments:
model_data = di.get_model_data()
start = 1
end = 1

# SETS:
resources_data = model_data['resources']
param_data = model_data['parameters']
task_data = model_data['tasks']

resources = list(resources_data['initial_elapsed'].keys())  # a
tasks = list(task_data['start'].keys())  # v
periods = aux.get_months(start, end)  # t
states = ['M', 'V', 'N', 'A']  # s

# TODO: this is for testing exclusively:

# resources = resources[:30]
periods = periods[:30]
tasks = tasks[:-1]
# print(tasks[-1])

states_noV = [s for s in states if s != 'V']
periods_0 = [aux.get_prev_month(start)] + periods  # periods with the previous one added at the start.


# PARAMETERS:

# maintenances:
max_elapsed_time = param_data['max_elapsed_time']  # me. in periods
max_used_time = param_data['used']  # mu. in hours of usage
duration = param_data['maint_duration']  # md. in periods
capacity = {t: param_data['maint_capacity'] for t in periods}  # c. in resources per period

# tasks - resources
start_time = ['start']  # not defined.
end_time = task_data['start']  # not defined.
candidates = task_data['candidates']  # cd. indexed set of resources.
consumption = task_data['consumption']  # rh. hours per period.
requirement = task_data['num_resource']  # rr. aircraft per period.

# time:
periods_pos = {periods[pos]: pos for pos in range(len(periods))}
previous = {period: periods_0[periods_pos[period]] for period in periods}
last_period = periods[-1]
first_period = periods[0]

# initial state:
ret_read = resources_data['initial_elapsed']
rut_read = resources_data['initial_used']

ret_init = {a: max_elapsed_time for a in resources}
rut_init = {a: max_used_time for a in resources}

ret_init.update(ret_read)
rut_init.update(rut_read)

ret_obj = sum(ret_init[a] for a in resources)
rut_obj = sum(rut_init[a] for a in resources)

# TODO: i'm still missing the fixed states (maintenances)
# TODO: add minimum mission assignment

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
at_start = [(a, t) for a, t in at if periods_pos[t] % 2 == 0]
at_start_not = [tup for tup in at if tup not in at_start]

at0 = [(a, t) for a in resources for t in periods_0]
att = [(a, t1, t2) for (a, t1) in at for t2 in periods if
       periods_pos[t1] <= periods_pos[t2] <= periods_pos[t1] + duration - 1 and \
       (a, t1) in at_start]

# a_t = {t: a for t in periods for a in resources if (a, t) in at}
a_t = tup_to_dict(at, result_col=0, is_list=True)
a_vt = tup_to_dict(avt, result_col=0, is_list=True)
v_at = tup_to_dict(avt, result_col=1, is_list=True)
t1_at2 = tup_to_dict(att, result_col=1, is_list=True)

# VARIABLES:

# binary:
task = pl.LpVariable.dicts("task", avt, 0, 1, pl.LpInteger)
start = pl.LpVariable.dicts("start", at_start, 0, 1, pl.LpInteger)
state = pl.LpVariable.dicts("state", ast, 0, 1, pl.LpInteger)

# numeric:
ret = pl.LpVariable.dicts("ret", at, 0, ub['ret'], pl.LpContinuous)
rut = pl.LpVariable.dicts("rut", at, 0, ub['rut'], pl.LpContinuous)
used = pl.LpVariable.dicts("used", at, 0, ub['used'], pl.LpContinuous)

# objective function:
min_avail = pl.LpVariable("min_avail", upBound=len(resources))
max_maint = pl.LpVariable("min_avail", lowBound=0)

# MODEL

model = pl.LpProblem("MFMP_v0001", pl.LpMaximize)

# OBJECTIVE:

model += min_avail

# CONSTRAINTS:

# capacity:
for t in periods:
    model += pl.lpSum(state[(a, 'M', t)] for a in a_t[t]) <= capacity[t]

# num resources:
for (v, t) in a_vt:
    model += pl.lpSum(task[(a, v, t)] for a in a_vt[(v, t)]) >= requirement[v]

# max one task per period:
for (a, t) in v_at:
    model += pl.lpSum(task[(a, v, t)] for v in v_at[(a, t)]) <= 1

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
    if t != first_period:
        model += rut[(a, t)] <= rut[(a, previous[t])] - used[(a, t)] + max_used_time * start[(a, t)]
        model += ret[(a, t)] <= ret[(a, previous[t])] - 1 + max_elapsed_time * start[(a, t)]
    else:
        model += rut[(a, t)] <= rut_init[a] - used[(a, t)] + max_used_time * start[(a, t)]
        model += ret[(a, t)] <= ret_init[a] - 1 + max_elapsed_time * start[(a, t)]

# if that month we know we're not starting a maintenance... it's just decreasing:
for (a, t) in at_start_not:
    if t != first_period:
        model += rut[(a, t)] <= rut[(a, previous[t])] - used[(a, t)]
        model += ret[(a, t)] <= ret[(a, previous[t])] - 1
    else:
        model += rut[(a, t)] <= rut_init[a] - used[(a, t)]
        model += ret[(a, t)] <= ret_init[a] - 1

# maintenance duration:
for (a, t1, t2) in att:
    model += state[(a, 'M', t2)] >= start[(a, t1)]

# only maintenance state if started
# not sure if this constraint is necessary.
for (a, t2) in at:
    model += pl.lpSum(start[(a, t1)] for t1 in t1_at2[(a, t2)]) >= state[(a, 'M', t2)]

# not sure which one is better, both?
for (a, v, t) in avt:
    model += state[(a, 'V', t)] >= task[(a, v, t)]
    for s in states_noV:
        model += task[(a, v, t)] + state[(a, s, t)] <= 1
for (a, t) in v_at:
    model += state[(a, 'V', t)] >= pl.lpSum(task[(a, v, t)] for v in v_at[(a, t)])

# only one state per period:
for (a, t) in at:
    model += pl.lpSum(state[(a, s, t)] for s in states) == 1

for (t) in periods:
    # objective: availability
    model += min_avail <= pl.lpSum(state[(a, 'A', t)] for a in a_t[t])
    # objective: maintenance (commented because not in use)
    # model += max_maint >= pl.lpSum(state[(a, 'M', t)])

# While we decide how to fix the ending of the planning period,
# we will try get at least the same amount of total rut and ret than
# at the beginning.
model += pl.lpSum(ret[(a, last_period)] for a in resources) >= ret_obj
model += pl.lpSum(rut[(a, last_period)] for a in resources) >= rut_obj

# model += min_avail <= len(resources)

# SOLVING
# model.solve(pl.PULP_CBC_CMD(maxSeconds=99, msg=True, fracGap=0, cuts=True, presolve=True))
directory_path = '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/{}/'.format(aux.get_timestamp())
os.mkdir(directory_path)
result_path = directory_path + 'gurobi.sol'.format()
log_path = directory_path + 'gurobi.log'
gurobi_options = [('TimeLimit', 6000), ('ResultFile', result_path), ('LogFile', log_path)]
model.solve(pl.GUROBI_CMD(options=gurobi_options))

_start = {_t: 1 for _t in start if start[_t].value()}

_state = tup_to_dict(vars_to_tups(state), result_col=1, is_list=False)
_task = tup_to_dict(vars_to_tups(task), result_col=1, is_list=False)
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
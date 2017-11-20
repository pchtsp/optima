import pulp as pl
import arrow
from package.data_input import generate_data_from_source

# arrow.now('US/Pacific').shift(months=1).format("YYYY-MM")


def get_months(start, end):
    periods = []
    current = start
    while current <= end:
        periods.append(current.format("YYYY-MM"))
        current = current.shift(months=1)
    return periods

# we import the data set.
table = generate_data_from_source()

# df3 = df3.assign(velo=df3.dist / df3.duration*3600/1000)  # for km/h

tasks_data = table['Missions']
tasks_data = \
    tasks_data.assign(start=tasks_data.AnneeDeDebut.apply(str) + '-' +
                            tasks_data.MoisDeDebut.apply(lambda x: str(x).zfill(2)),
                      end=tasks_data.AnneeDeFin.apply(str) + '-' +
                          tasks_data.MoisDeFin.apply(lambda x: str(x).zfill(2)))

tasks_data.set_index('IdMission', inplace= True)

start = arrow.get(tasks_data.start.values.min() + "-01")
end = arrow.get(tasks_data.end.values.max() + "-01")

prev = start.shift(months=-1).format("YYYY-MM")


params = table['Parametres']
params = params[~params.Unnamed9.isna()].rename(
    columns={'Unnamed9': 'name', 'Unnamed10': 'value'})[['name', 'value']]

params = params.set_index('name').to_dict()['value']

maint = table['DefinitionMaintenances']

# SETS:

resources = table['Avions_Capacite'].IdAvion.values  # a
tasks = table['Missions'].IdMission.values  # v
periods = get_months(start, end)  # t
periods_0 = [prev] + periods  # periods with the previous one added at the start.
states = ['M', 'V', 'N', 'A']  # s

# PARAMETERS:

# maintenances:
max_elapsed_time = \
    maint.GainPotentielCalendaire_mois.values.min()  # me. in periods
max_used_time = \
    maint.GainPotentielHoraire_heures.values.min() # mu. in hours of usage
duration = \
    maint.DureeMaintenance_mois.values.max()  # md. in periods
capacity = \
    {t: params['Maintenance max par mois'] for t in periods}  # c. in resources per period

# tasks - resources
start_time = tasks_data.start.to_dict()  # not defined.
end_time = tasks_data.end.to_dict()  # not defined.
candidates = {}  # cd. indexed set of resources TODO.
consumption = tasks_data['MaxPu/avion/mois'].to_dict()  # rh. hours per period.
requirement = tasks_data.nombreRequisA1.to_dict()  # rr. aircraft per period.

# time:
periods_pos = {periods[pos]: pos for pos in range(len(periods))}
previous = {period: periods_0[periods_pos[period]] for period in periods}

# fixed values:

# TODO: fill fixed

# DOMAINS:

vt = [(v, t) for v in tasks for t in periods if start_time[v] <= t <= end_time[v]]
avt = [(a, v, t) for a in resources for (v, t) in vt if a in candidates[v]]
at = [(a, t) for a in resources for t in periods]  # if periods_pos[t] % 2 == 0?
ast = [(a, s, t) for (a, t) in at for s in states]

at0 = [(a, t) for a in resources for t in periods_0]
att = [(a, t1, t2) for (a, t1) in at for t2 in periods if
       periods_pos[t1] <= periods_pos[t2] <= periods_pos[t1] + duration - 1]

a_vt = {(v, t): [a for a in resources if (a, v, t) in avt] for (v, t) in vt}
v_at = {(a, t): [v for v in tasks if (a, v, t) in avt] for (a, t) in at}
t1_at2 = {(a, t2): [t1 for t1 in periods if (a, t1, t2) in att] for (a, t2) in at}

# VARIABLES:

# binary:
task = pl.LpVariable.dicts("task", avt, 0, 1, pl.LpInteger)
start = pl.LpVariable.dicts("start", at, 0, 1, pl.LpInteger)
state = pl.LpVariable.dicts("state", ast, 0, 1, pl.LpInteger)

# numeric:
# TODO: calculate bounds for continous vars. # ub['var'] instead of 1
ret = pl.LpVariable.dicts("ret", at0, 0, 1, pl.LpContinuous)
rut = pl.LpVariable.dicts("rut", at0, 0, 1, pl.LpContinuous)
used = pl.LpVariable.dicts("used", at, 0, 1, pl.LpContinuous)

# objective function:
min_avail = pl.LpVariable("min_avail", upBound=len(resources))
max_maint = pl.LpVariable("min_avail", lowBound=0)

# MODEL

model = pl.LpProblem("MFMP_v0001", pl.LpMinimize)

# OBJECTIVE:

model += min_avail

# CONSTRAINTS:

# capacity:
for t in periods:
    model += pl.lpSum(state[_ast] for _ast in ast if _ast[1] == 'M') <= capacity[t]

# num resources:
for (v, t) in vt:
    model += pl.lpSum(task[(a, v, t)] for a in a_vt[(v, t)]) >= requirement[(v, t)]

# max one task per period:
for (a, t) in at:
    model += pl.lpSum(task[(a, v, t)] for v in v_at[(a, t)]) <= 1

# used time, two options:
# TODO: maybe set equal?
# TODO: not sure which one is better, both?
for (a, t) in at:
    model += used[(a, t)] >= pl.lpSum(task[(a, v, t)] * consumption[(v, t)] for v in v_at[(a, t)])
for (a, v, t) in avt:
    model += used[(a, t)] >= task[(a, v, t)] * consumption[(v, t)]

# remaining used time calculations:
for (a, t) in at:
    model += rut[(a, t)] <= rut[(a, previous[t])] - used[(a, t)] + max_used_time * start[(a, t)]

# remaining elapsed time calculations:
# TODO: *maybe* reformulate this
for (a, t) in at:
    model += ret[(a, t)] <= ret[(a, previous[t])] - 1 + max_elapsed_time * start[(a, t)]

# maintenance duration:
for (a, t1, t2) in att:
    model += state[(a, 'M', t2)] >= start[(a, t1)]

# only maintenance state if started
# TODO: not sure if this constraint is necessary.
for (a, t2) in at:
    model += pl.lpSum(start[(a, t1)] for t1 in t1_at2[(a, t2)]) >= state[(a, 'M', t2)]

# TODO: not sure which one is better, both?
for (a, v, t) in avt:
    model += state[(a, 'V', t)] >= task[(a, v, t)]
for (a, t) in at:
    model += state[(a, 'V', t)] >= pl.lpSum(task[(a, v, t)] for v in tasks if (a, v, t) in avt)

# only one state per period:
for (a, t) in at:
    model += pl.lpSum(state[(a, s, t)] == 1 for s in states)

for (a, t) in at:
    # objective: availability
    model += min_avail <= pl.lpSum(state[(a, 'A', t)])
    # objective: maintenance
    model += max_maint >= pl.lpSum(state[(a, 'M', t)])

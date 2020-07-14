import data.simulation as sim
import package.params as params
import package.instance as inst
import solvers.model as model

options = params.OPTIONS
num_tasks = 1
options['num_period'] = 20
options['simulation']['num_period'] = 20
options['simulation']['maint_duration'] = 2
options['simulation']['max_used_time'] = 300
options['simulation']['max_elapsed_time'] = 15
# options['simulation']['maintenances'] = dict(M=options['simulation']['maintenances']['M'])
options['simulation']['t_required_hours'] = (10, 30, 40)
options['simulation']['t_duration'] = (3, 4)
options['simulation']['t_min_assign'] = (2, 3)
options['simulation']['t_num_resource'] = (1 , 3)
options['simulation']['num_resources'] = num_tasks*5
options['simulation']['num_parallel_tasks'] = num_tasks
options['solver'] = 'CPLEX'

data = sim.create_dataset(options)
instance = inst.Instance(data)

# tasks:
rename = dict(min_assign='$MT^{min}_{j}$', start='$Start_j$', end='$End_j$',
                 consumption='$H_{j}$', num_resource='$R_{j}$')
a = instance.get_tasks().to_df(orient='index')
pos = instance.get_period_positions()
a.start = a.start.map(pos) + 1
a.end = a.end.map(pos) + 1
a = a.rename(columns=rename).filter(rename.values(), axis=1)
a.index.name = 'j'
print(a.to_latex(longtable=False, escape=False))

# resources:
rename = dict(elapsed="$Rct^{Init}_i$", used="$Rft^{Init}_i$")
b = instance.\
    get_resources('initial').\
    get_property('M').\
    to_df(orient='index').\
    rename(columns=rename).filter(rename.values(), axis=1)
b.index.name = 'i'
print(b.to_latex(longtable=False, escape=False))

# clusters:
cluster_info = instance.get_cluster_constraints()
cluster_info['num'].to_dictdict().vapply(lambda v: max(v.values()))
cluster_info['hours'].to_dictdict().vapply(lambda v: max(v.values()))
# 1,4
# 300, 750

mobj = model.Model(instance)
solution = mobj.solve(options)
mobj.to_dir(options['path'])
mobj.check_solution()

import reports.gantt as gantt

gantt.make_gantt_from_experiment(mobj)


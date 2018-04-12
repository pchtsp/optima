import package.tests as exp
import package.solution as sol
import pprint as pp
from package.params import PATHS
import package.auxiliar as aux
import pandas as pd

# '201801141334' vs '201802061201 vs '201802061434'

experiments = ['201801141334', '201802061201', '201802061434']

for e in experiments:
    experiment = exp.Experiment.from_dir(PATHS['experiments']+e)
    # experiment.solution.print_solution("/home/pchtsp/Downloads/calendar_temp1.html")
    experiment.check_solution()
    print(e)
    pp.pprint(experiment.get_kpis())

e = '201804111538'
experiment = exp.Experiment.from_dir(PATHS['experiments'] + e)
# experiment.solution.print_solution("/home/pchtsp/Downloads/calendar_temp1.html")
checks = experiment.check_solution()
pp.pprint(checks)
pp.pprint(experiment.get_kpis())

# experiment.get_objective_function()
# kpis1 = experiment.get_kpis()
# experiment.solution.get_in_maintenance()
#
# experiment2 = exp.Experiment.from_dir(PATHS['experiments']+'201801141334')
# kpis2 = experiment2.get_kpis()
# experiment2.get_objective_function()
# experiment2.solution.get_in_maintenance()

path_abs = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/"
path_states = path_abs + "201712190002/"
path_nostates = path_abs + "201712182321/"

sol_states = exp.Experiment.from_dir(path_states)
sol_nostates = exp.Experiment.from_dir(path_nostates)

t = aux.tup_to_dict(aux.dict_to_tup(sol_nostates.solution.get_tasks()),
                    result_col=[0], is_list=True, indeces=[2, 1]).items()
# {k: len(v) for k, v in t}
sol_nostates.instance.get_tasks('num_resource')

path = path_abs + "201801091412/"
sol_states = exp.Experiment.from_dir(path)
check = sol_states.check_solution()
# check['resources']


rut_old = sol_states.solution.data["aux"]['rut']
sol_states.set_remaining_usage_time()
rut_new = sol_states.solution.data['aux']['rut']
pd.DataFrame.from_dict(rut_new, orient='index').reset_index().melt(id_vars='index'). \
    sort_values(['index', 'variable']).to_csv('/home/pchtsp/Downloads/TEMP_new.csv', index=False)
pd.DataFrame.from_dict(rut_old, orient='index').reset_index().melt(id_vars='index'). \
    sort_values(['index', 'variable']).apply('round').to_csv('/home/pchtsp/Downloads/TEMP_old.csv', index=False)

# sol_nostates.check_solution()
# sol_states.check_solution()

# exp = exp.Experiment(inst.Instance(model_data), sol.Solution(solution))
# exp = exp.Experiment.from_dir(path)
# results = exp.check_solution()

# [k for (k, v) in sol_nostates.check_solution().items() if len(v) > 0]
# [k for (k, v) in sol_states.check_solution().items() if len(v) > 0]

print(sol_states.get_objective_function())
print(sol_nostates.get_objective_function())
sol_nostates.check_task_num_resources()

# sol_states.solution.get_state()[('A2', '2017-03')]
l = sol_states.instance.get_domains_sets()
# l['v_at']
checks = sol_states.check_solution()

sol_nostates.check_solution()
# sol_states.solution.print_solution("/home/pchtsp/Documents/projects/OPTIMA/img/calendar.html")
sol_nostates.solution.print_solution("/home/pchtsp/Documents/projects/OPTIMA/img/calendar.html")
#
# [k for (k, v) in results["duration"].items() if v < 6]
# results["usage"]
#
#
# consum = exp.get_consumption()
# aux.dicttup_to_dictdict(exp.get_remaining_usage_time())['A46']
# aux.dicttup_to_dictdict(consum)["A46"]
#
# results["resources"]
#
# aux.dicttup_to_dictdict(exp.solution.get_task_num_resources())['O10']['2017-09']
# exp.instance.get_tasks("num_resource")

# sol.Solution(solution).get_schedule()
# check.check_resource_consumption()
# check.check_resource_state()

import package.tests as exp
import package.solution as sol
import pprint as pp
from package.params import PATHS

# '201801141334' vs '201802061201 vs '201802061434'

experiments = ['201801141334', '201802061201', '201802061434']

for e in experiments:
    experiment = exp.Experiment.from_dir(PATHS['experiments']+e)
    # experiment.solution.print_solution("/home/pchtsp/Downloads/calendar_temp1.html")
    experiment.check_solution()
    print(e)
    pp.pprint(experiment.get_kpis())


# experiment.get_objective_function()
# kpis1 = experiment.get_kpis()
# experiment.solution.get_in_maintenance()
#
# experiment2 = exp.Experiment.from_dir(PATHS['experiments']+'201801141334')
# kpis2 = experiment2.get_kpis()
# experiment2.get_objective_function()
# experiment2.solution.get_in_maintenance()
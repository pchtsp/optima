import package.data_input as di
import package.instance as inst
import package.tests as exp
import package.solution as sol
import pprint as pp
import package.auxiliar as aux

if __name__ == "__main__":
    path_solution = "/home/pchtsp/Documents/projects/PIE/glouton/solution.csv"
    path_input = "/home/pchtsp/Documents/projects/PIE/glouton/parametres_DGA_final.xlsm"

    input_data = di.get_model_data(path_input)
    solution_data = di.import_pie_solution(path_solution, path_input)

    instance = inst.Instance(input_data)
    solution = sol.Solution(solution_data)
    experiment = exp.Experiment(instance, solution)

    checks = experiment.check_solution()
    schedule = solution.get_schedule()
    pp.pprint(schedule)

    tups = aux.dict_to_tup(checks['resources'])
    dicts = aux.tup_to_dict(tups, result_col=[1, 2])
    mission = {k: sum(int(v2[1]) for v2 in v) for k, v in dicts.items()}
    sum(checks['resources'].values())
    sum(instance.get_task_period_needs().values())

    pp.pprint(checks)
    pp.pprint(instance.get_tasks('num_resource'))

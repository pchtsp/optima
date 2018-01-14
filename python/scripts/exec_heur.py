import package.solution as sol
import package.heuristics as heur
import package.data_input as di
import package.instance as inst


if __name__ == "__main__":

    path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201801141334/"
    model_data = di.load_data(path + "data_in.json")
    # this was for testing purposes

    instance = inst.Instance(model_data)
    solution = sol.Solution({'state': {}, 'task': {}})
    heur = heur.Greedy(instance, solution)
    # heur.instance.get_tasks()
    # heur.expand_resource_period(heur.solution.data['task'], 'A100', '2017-01')
    tasks = list(heur.instance.get_tasks().keys())
    # tasks = ['O5']
    for task in tasks:
        heur.fill_mission(task)
    rut_old = dict(heur.solution.data['aux']['rut'])
    # heur.solution.print_solution("/home/pchtsp/Downloads/calendar_temp1.html")
    checks = heur.check_solution()
    rut_new = heur.solution.data['aux']['rut']
    print([k for k, v in checks.items() if len(v) > 0])
    heur.get_objective_function()
    # checks['usage']
    # {k: v for k, v in checks.items() if len(v) > 0}.keys()
    # checks.keys()
    # path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712190002/"
    # self = Greedy.from_dir(path)
    #
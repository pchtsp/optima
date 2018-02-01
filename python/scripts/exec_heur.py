import package.solution as sol
import package.heuristics as heur
import package.data_input as di
import package.instance as inst
import pprint as pp
import package.tests as exp


if __name__ == "__main__":

    path_abs = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/"
    path = path_abs + "201801131817/"
    path = path_abs + "201801141331/"
    path = path_abs + "201801141646/"
    experiment = exp.Experiment.from_dir(path)
    heur_obj = heur.Greedy(experiment.instance)
    heur_obj.solve()
    check = heur_obj.check_solution()
    pp.pprint(check)
    # heur.solution.print_solution("/home/pchtsp/Downloads/calendar_temp1.html")

import package.experiment as exp
import package.solution as sol
import pprint as pp
from package.params import PATHS
import package.auxiliar as aux
import package.data_input as di
import pandas as pd
import package.superdict as sd
import package.model as md

# '201801141334' vs '201802061201 vs '201802061434'

def test1():
    experiments = ['201801141334', '201802061201', '201802061434']

    for e in experiments:
        experiment = exp.Experiment.from_dir(PATHS['experiments']+e)
        # experiment.solution.print_solution("/home/pchtsp/Downloads/calendar_temp1.html")
        experiment.check_solution()
        print(e)
        pp.pprint(experiment.get_kpis())


def test2():
    # e = '201809121711'
    e = '201809121815'
    experiment = exp.Experiment.from_dir(PATHS['experiments'] + e)
    experiment.set_remaining_usage_time('ret')
    experiment.get_status('19')
    # experiment.set_remaining_usage_time('ret')
    # pp.pprint(experiment.solution.data['aux']['start_T'])
    # experiment.solution.print_solution("/home/pchtsp/Downloads/calendar_temp1.html")
    checks = experiment.check_solution()
    checks.keys()
    pp.pprint(checks['resources'])
    pp.pprint(checks['elapsed'])
    pp.pprint(experiment.get_kpis())
    # experiment.


def test555():
    e = '201809121056'
    experiment = exp.Experiment.from_dir(PATHS['experiments'] + e)
    instance = experiment.instance
    options = di.load_data(PATHS['experiments'] + e + '/options.json')
    solution = md.solve_model(instance, options)
    # experiment.solution.print_solution("/home/pchtsp/Downloads/calendar_temp1.html")
    # checks = experiment.check_solution()
    # checks.keys()


def test4():
    case1 = '201805232311'
    case2 = '201805232322'
    case3 = '201805241036'
    heuristiques = [case3]
    path_abs = PATHS['experiments']
    options_e = exp.list_options(path_abs)
    heuristiques = [o for o, v in options_e.items() if v['solver'] == 'HEUR']
    experiments_info = exp.list_experiments(path=path_abs, exp_list=heuristiques, get_log_info=False)
    experiments = {e: exp.Experiment.from_dir(path_abs + e) for e in heuristiques}
    experiments = sd.SuperDict(experiments)
    experiments = experiments.clean(default_value=None)
    checks = {e: v.check_solution() for e, v in experiments.items()}
    checks_len = {e: {'infeasible': sum(len(vv) for vv in v.values())}
              for e, v in checks.items()}

    dict_join = {e: {**experiments_info.get(e),
                     **checks_len.get(e),
                     **options_e[e],
                     } for e in experiments}

    pd.DataFrame.from_dict(dict_join, orient='index')\
        [['infeasible', 'periods', 'tasks']]. \
        sort_values(['tasks', 'periods'])

    pass
    # backup = copy.deepcopy(heur_obj.solution.data['aux']['ret'])

    # check = heur_obj.check_solution()
    # pp.pprint(check)
    # check.keys()
    # res_problems = set([tup[0] for tup in check['elapsed']])

    # pp.pprint(heur_obj.solution.data['aux']['rut']['A97'])
    # pp.pprint(backup)
    # [res for res in res_problems if res in heur_obj.solution.data['task']]
    # heur_obj.solution.data['task']['A97']

    # heur.solution.print_solution("/home/pchtsp/Downloads/calendar_temp1.html")


def test3():
    path_abs = PATHS['experiments']
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
    rut_old = sol_states.solution.data["aux"]['rut']
    sol_states.set_remaining_usage_time()
    rut_new = sol_states.solution.data['aux']['rut']
    pd.DataFrame.from_dict(rut_new, orient='index').reset_index().melt(id_vars='index'). \
        sort_values(['index', 'variable']).to_csv('/home/pchtsp/Downloads/TEMP_new.csv', index=False)
    pd.DataFrame.from_dict(rut_old, orient='index').reset_index().melt(id_vars='index'). \
        sort_values(['index', 'variable']).apply('round').to_csv('/home/pchtsp/Downloads/TEMP_old.csv', index=False)

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


if __name__ == '__main__':
    # test2()
    pass

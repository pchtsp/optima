import package.experiment as exp
import pprint as pp
from package.params import PATHS
import package.auxiliar as aux
import package.data_input as di
import pandas as pd
import package.superdict as sd
import scripts.exec as exec
import os
import package.model as md
import package.rpy_graphs as rg

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
    e = '201810041046'
    experiment = exp.Experiment.from_dir(PATHS['experiments'] + e)
    experiment.set_remaining_usage_time('ret')
    # experiment.get_status('0')
    # pp.pprint(experiment.solution.data['aux']['start_T'])
    # experiment.solution.print_solution("/home/pchtsp/Downloads/calendar_temp1.html")

    checks = experiment.check_solution()
    checks.keys()
    pp.pprint(checks['resources'])
    pp.pprint(checks['elapsed'])
    pp.pprint(checks['usage'])
    pp.pprint(experiment.get_kpis())
    print(experiment.get_status('16').to_string())
    # experiment.


def test555():
    e = 'simulated_data/1_task_types/201810051247/'
    experiment = exp.Experiment.from_dir(PATHS['results'] + e)
    instance = experiment.instance
    options = di.load_data(PATHS['results'] + e + 'options.json')
    checks = experiment.check_solution()
    solution = experiment.solution
    # solution = md.solve_model(instance, options)
    di.export_data(options['path'], solution.data, name="data_out", file_type='json')
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

def test444():
    path = PATHS['experiments'] + "201902011249/"
    path = PATHS['data'] + 'examples/201811092041/'
    path = PATHS['results'] + 'clust_params1_cplex/base/201811092041_1//'
    path = PATHS['results'] + 'clust_params2_cplex/numparalleltasks_2/201811220958/'
    path = PATHS['results'] + 'clust_params1_cplex/minusageperiod_15/201811240019/'
    experiment = exp.Experiment.from_dir(path)
    rg.gantt_experiment(path)

    experiment.check_solution()

    input_data = di.load_data(PATHS['experiments'] + "201810051701/data_in.json")

    [r for r in input_data['resources'].values() if 'C' in r['capacities']]


def test111():
    path = PATHS['results'] + 'simulated_data/1_task_types_capa_slack/201810090924/'
    new_options = {'gap': 1}
    new_options =  {'slack_vars': 'No', 'gap': 1}
    new_options = None
    exec.re_execute_instance(path, new_options)

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


def check_over_assignments():
    experiment = 'clust1_20181107'
    path_results = PATHS['results']
    path_exps = path_results + experiment

    def listdir_fullpath(d):
        return [os.path.join(d, f) for f in os.listdir(d)]

    exps = {os.path.basename(e): exp.Experiment.from_dir(e+'/') for p in listdir_fullpath(path_exps) for e in listdir_fullpath(p)}
    exp_scenario = {os.path.basename(e): os.path.basename(p) for p in listdir_fullpath(path_exps) for e in             listdir_fullpath(p)}
    checks = {e: v.check_task_num_resources(strict=True) for e, v in exps.items() if v is not None}


    keys = sd.SuperDict(checks).to_lendict().clean().keys_l()
    set(sd.SuperDict(exp_scenario).filter(keys).values_l())
    set(sd.SuperDict(exp_scenario).values_l())

    scenario_over_assigned = sd.SuperDict(exp_scenario).filter(keys).values_l()
    num = pd.Series(scenario_over_assigned).value_counts()

    num = sd.SuperDict(checks).to_lendict().values_l()
    result = pd.Series.from_array(num).value_counts()
    # exp.Experiment.
    result.sort_index(inplace=True)
    return result


if __name__ == '__main__':
    check_over_assignments()
    pass

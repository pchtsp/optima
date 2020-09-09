import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import package.experiment as exp
from package.params import PATHS, OPTIONS

import data.data_input as di

import solvers.heuristics_maintfirst as heur
import solvers.heuristic_graph as hgr

import pprint as pp
import pandas as pd
import pytups.superdict as sd
import pytups.tuplist as tl
import execution.exec as exec
import multiprocessing as multi

# Windows workaround for python 3.7 (sigh...)
# import _winapi
# import multiprocessing.spawn
# multiprocessing.spawn.set_executable(_winapi.GetModuleFileName(0))
#################

def load_batch():
    import package.batch as ba
    import stochastic.solution_stats as sol_stats

    case1 = '201805232311'
    case2 = '201805232322'
    case3 = '201805241036'
    heuristiques = [case3]
    # path_abs = r'/home/pchtsp/Documents/projects/optima_results_old/experiments/'
    path_abs = PATHS['results'] + 'IT000125_20190915.zip'
    batch = ba.ZipBatch(path_abs, scenarios=['numparalleltasks_1'])
    dataset = batch.generate_JR_dataset()
    dataset.to_csv('test.csv', header=False, index=False)
    options_e = batch.get_options()
    heuristiques = [o for o, v in options_e.items() if v['solver'] == 'HEUR']
    experiments_info = batch.list_experiments(get_log_info=False, exp_list=heuristiques)
    # exp.list_experiments(path=path_abs, exp_list=heuristiques, get_log_info=False)
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
def load():
    # e = '201809121711'
    e = '/home/pchtsp/Documents/projects/optima_results_old/experiments/202004142329/'
    experiment = hgr.GraphOriented.from_dir(e)
    experiment.get_objective_function()
    experiment.check_solution()
    experiment.instance.get_tasks('num_resource').values_l()
    experiment.instance.get_tasks('start')
    experiment = hgr.GraphOriented.from_dir(PATHS['experiments'] + e)
    experiment.check_solution()
    experiment.get_objective_function()
    experiment.set_remaining_usage_time('ret')

    experiment2 = exp.Experiment.from_dir('../data/cases/202003231502')
    experiment2 = hgr.GraphOriented.from_dir('../data/cases/202003231502')
    experiment2.check_solution()
    experiment2.get_objective_function()
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



def graph_check():
    # path = PATHS['experiments'] + "201902061522/"
    # path = PATHS['experiments'] + "201903251641/"
    # path = PATHS['data'] + 'examples/201811231417/'
    path = PATHS['results'] + '../optima_results_old/experiments/202004142329/'
    # path = PATHS['results'] + 'clust_params2_cplex/numparalleltasks_2/201811220958/'
    # path = PATHS['results'] + 'clust_params1_cplex/minusageperiod_15/201811240019/'
    experiment = exp.Experiment.from_dir(path)
    experiment.check_solution()
    experiment.check_min_distance_maints()
    status = experiment.get_status('9')
    status.reset_index(inplace=True)
    status.columns = ['period', 'rut', 'ret', 'state', 'task']
    status.query("period>'2023-08' and period < '2023-12'")
    # status >> dp.filter_by(X.period > "2023-08", X.period < "2023-12")


    # rg.gantt_experiment(path, './../../R/functions/import_results.R')
    # experiment.check_maints_size()
    experiment.check_solution()

    # input_data = di.load_data(PATHS['experiments'] + "201810051701/data_in.json")


def graph():
    import reports.gantt as gantt
    path = PATHS['results'] + '../optima_results_old/experiments/202004142333/'
    gantt.make_gantt_from_experiment(path=path)


def run_rexecute():
    e = 'examples/201811230508/'
    path = PATHS['data'] + e
    path = '../data/cases/202003231502/'
    # path = "/home/pchtsp/Documents/projects/optima_results/experiments/201902141810/"

    new_options = OPTIONS
    new_options = di.load_data(path+'options.json')
    add_options = {'solver': 'CPLEX', 'path': path,
                        'mip_start': True, 'fix_start': False, 'writeLP': False}
    add_options = {'solver': 'HEUR_mf', 'mip_start': False}
    add_options = dict(max_iters_initial=0, big_window=True, num_max=10000,
                    max_patterns_initial=10000,
                    max_iters=0, timeLimit_cycle=1000, solver='HEUR_Graph.CPLEX_PY')
    new_options.update(add_options)
    # new_options =  {'slack_vars': 'No', 'gap': 1}
    # new_options = None
    exec.re_execute_instance(path, new_options)

def run_rexecute_many(exp_origin, exp_dest, num_proc=2, max_instances=None, new_options=None, **kwargs):

    pool = multi.Pool(processes=num_proc)
    origin_path = os.path.join(PATHS['results'], exp_origin)
    destination_path = os.path.join(PATHS['results'], exp_dest)
    remakes_path = os.path.join(destination_path, 'index.txt')
    with open(remakes_path, 'r') as f:
        instances = f.readlines()

    instances = tl.TupList(instances).vapply(lambda x: x.strip())
    path_ins = instances.vapply(lambda x: os.path.join(origin_path, x))
    path_outs = instances.vapply(lambda x: os.path.join(destination_path, x))
    if new_options is None:
        new_options = {}

    if max_instances is None:
        max_instances = len(path_ins)

    results = {}
    for pos, path_in in enumerate(path_ins):
        path_out = path_outs[pos]
        args = {'directory': path_in, 'new_options': {**new_options, **{'path': path_out}}, **kwargs}
        results[pos] = pool.apply_async(exec.re_execute_instance_errors, kwds=args)
        if pos + 1 >= max_instances:
            break

    timelimit = new_options.get('timeLimit', 3600)
    for pos, result in results.items():
        try:
            result.get(timeout=timelimit+600)
        except multi.TimeoutError:
            print('Multiprocessing TimeoutError happened.')
            pass


def check_over_assignments():
    experiment = 'clust1_20181107'
    path_results = PATHS['results']
    path_exps = path_results + experiment

    def listdir_fullpath(d):
        return [os.path.join(d, f) for f in os.listdir(d)]

    exps = {os.path.basename(e): exp.Experiment.from_dir(e+'/')
            for p in listdir_fullpath(path_exps)
            for e in listdir_fullpath(p)}
    exp_scenario = {os.path.basename(e): os.path.basename(p)
                    for p in listdir_fullpath(path_exps)
                    for e in listdir_fullpath(p)}
    checks = {e: v.check_task_num_resources(deficit_only=False) for e, v in exps.items() if v is not None}


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

def check_template_data():
    import data.template_data as td
    import package.instance as inst
    import package.solution as sol
    import data.data_input as di

    _dir = PATHS['data'] + r'template/dassault20190821_3/'
    path_in = _dir + r'template_in.xlsx'
    path_sol = _dir + r'template_out.xlsx'
    path_err = _dir + r'errors.json'
    model_data = td.import_input_template(path_in)
    instance = inst.Instance(model_data)
    sol_data = td.import_output_template(path_sol)
    solution = sol.Solution(sol_data)
    experiment = heur.MaintenanceFirst(instance, solution)
    errors_old = di.load_data(path_err)
    errors = experiment.check_solution().to_dictdict().to_dictup()

    len(sd.SuperDict.from_dict(errors_old).to_dictup())
    experiment2 = exp.Experiment.from_dir(_dir)
    experiment2.solution.data['state_m']
    errors_real = experiment2.check_solution().to_dictdict().to_dictup()
    len(errors_real)
    len(errors)
    # errors.filter(errors_real.keys_l())
    errors_real.keys() - errors.keys()
    errors.keys() - errors_real.keys()
    experiment2.get_status('B3')
    experiment.get_status('B3')
    # errors_real.filter()
    aux1 =  experiment.solution.data['aux'].to_dictup().to_tuplist()
    aux2 = experiment2.solution.data['aux'].to_dictup().to_tuplist()
    set(aux2) - set(aux1)
    # import unittest
    # unittest.TestCase().assertDictEqual(errors, errors_real)
    # unittest.

    # gantt.make_gantt_from_experiment(experiment)

def solve_template():
    import data.template_data as td
    import package.instance as inst

    path_in = PATHS['data'] + r'template/Lot5 (3000s)/template_in.xlsx'
    model_data = td.import_input_template(path_in)
    instance = inst.Instance(model_data)
    experiment = heur.MaintenanceFirst(instance)
    options = OPTIONS
    options['path'] = os.path.dirname(path_in)
    experiment.solve(options)
    td.export_output_template('test.xlsx', model_data, experiment.solution.data)

    pass

def solve_dataset():
    import package.instance as inst
    import solvers.heuristics_maintfirst as heur
    from data import test_data
    def ttt():
        from importlib import reload
        reload(test_data)
    model_data = test_data.dataset2()
    instance = inst.Instance(model_data)
    experiment = heur.MaintenanceFirst(instance)
    solution = experiment.solve(dict(timeLimit=10, path=".",
                                     prob_free_aircraft=0.1,
                                     prob_free_periods=0.5,
                                     prob_delete_maint=0.1))
    solution.data['state_m']


def check_rem_calculation(experiment):
    path_exp = PATHS['experiments'] + experiment
    self = exp.Experiment.from_dir(path_exp)
    acc_consumption = self.get_acc_consumption()
    cycles = tl.TupList(acc_consumption).to_dict(result_col=[1, 2])
    acc_consumption = acc_consumption.clean(default_value=0)
    # add one pos to cycles that start a maint_duration just at the start
    # (because it means they have a previous 0 length period)
    _shift = self.instance.shift_period
    first = self.instance.get_param('start')
    duration = self.instance.get_param('maint_duration')
    extra_pos = cycles.apply(lambda k, v: (_shift(first, duration) == v[0][0])+0)
    cycles_pos = cycles.apply(lambda k, v: sd.SuperDict({str(kk+extra_pos[k]): vv for kk, vv in enumerate(v)})).to_dictup()

    _range = self.instance.get_dist_periods

    # I need to clean the auxiliary cycle info to be able to compare it.
    acc_consumption_aux = \
        sd.SuperDict.from_dict(self.solution.data['aux']['rem']).\
        to_dictup(). \
        clean(default_value=0)

    # key exchange:
    acc_consumption_aux2 = {(k1, *cycles_pos[k1, k2]): v for (k1, k2), v in acc_consumption_aux.items()}
    acc_consumption_aux2 = \
        sd.SuperDict.from_dict(acc_consumption_aux2).\
            kvapply(lambda k, v: (_range(k[1], k[2])+1) * v)
    acc_consumption.apply(lambda k, v: v - acc_consumption_aux2[k]).clean(func=lambda v: abs(v) > 0.1)
    acc_consumption_aux2.kvapply(lambda k, v: v - acc_consumption[k]).clean(func=lambda v: abs(v) > 0.1)

    rut = self.set_remaining_usage_time('rut')

    def dist(t1, t2):
        # t_2 - t_1 + 1
        return self.instance.get_dist_periods(t1, t2) + 1

    def acc_dist(t_1, t_2, tp):
        # sum_{t = t_1 -1}^{t_2} tp - t
        # return (t_2 - t_1 + 1) * (2 * tp - t_1 - t_2) / 2
        return dist(t_1, t_2) * (dist(t_1, tp) + dist(t_2, tp)) / 2

    # conclusion: some deviations with fixed assignments at the beginning of the planning period.
    # that are not taken into account in the model to calculate the average accumulated consumption.
    pass


def export_import_pulp():
    import data.simulation as sim
    import solvers.model as mdl
    import package.instance as inst
    import pulp
    import solvers.config as config

    options = OPTIONS
    options['do_not_solve'] = True
    options['solver'] = 'CPLEX'
    model_data = sim.create_dataset(options)
    instance = inst.Instance(model_data)
    experiment = mdl.Model(instance, solution=None)
    solution = experiment.solve(options)
    model_dict = experiment.model.to_dict()
    variables, another_model = pulp.LpProblem.from_dict(model_dict)

    c = config.Config(options)
    result = c.solve_model(another_model)
    result = c.solve_model(experiment.model)

    exec.config_and_solve(options)

    pass



if __name__ == '__main__':
    # load()
    # check_rem_calculation('201904181142')
    # test_rexecute()
    # graph()
    # load_batch()
    # options = sd.SuperDict(timeLimit=3600, mip_start=True, exclude_aux=False, threads=1)
    # options.update(StochCuts= {'active': False}, reduce_2M_window={'active': True, 'window_size': 10})
    # new_input = sd.SuperDict.from_dict({'parameters': {'elapsed_time_size_2M': 10, 'max_elapsed_time_2M': 40}})
    # kwargs = {'exp_origin': 'dell_20190515_all/base',
    #           'exp_dest': 'dell_20190515_remakes/base',
    #           'num_proc': 7,
    #           'new_options': options,
    #           'max_instances': 2
    #           }
    # test_rexecute_many(**kwargs)
    # check_over_assignments()
    # test_rexecute()
    # path = r'C:\Users\pchtsp\Documents\borrar\experiments\201903121106/'
    # graph_check(path)
    # solve_template()
    # solve_template()

    pass


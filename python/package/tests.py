# /usr/bin/python3
import package.aux as aux
import numpy as np
import pandas as pd
import package.data_input as di
import package.solution as sol
import package.instance as inst
import os
import shutil
import pprint as pp
import re
import package.logFiles as log


# TODO: make Experiment subclass of Solution.
# TODO: make Solution subclass of Instance?
# TODO: create period objects with proper date methods based on arrow
# TODO: create listtuple and dictionary objects with proper methods


class Experiment(object):
    """
    These objects represent the unification of both input data and solution.
    Each is represented by their proper objects.
    Methods are especially checks on faisability.
    In the future I may make this object be a sublass of Solution.
    """

    def __init__(self, instance, solution):
        self.instance = instance
        self.solution = solution

    @classmethod
    def from_dir(cls, path, format='json', prefix="data_"):
        files = [os.path.join(path, prefix + f + "." + format) for f in ['in', 'out']]
        if not np.all([os.path.exists(f) for f in files]):
            return None
        instance = di.load_data(files[0])
        solution = di.load_data(files[1])
        return cls(inst.Instance(instance), sol.Solution(solution))

    @staticmethod
    def expand_resource_period(data, resource, period):
        if resource not in data:
            data[resource] = {}
        if period not in data[resource]:
            data[resource][period] = {}
        return True

    @staticmethod
    def label_rt(time):
        if time == "rut":
            return 'used'
        elif time == 'ret':
            return 'elapsed'
        else:
            raise ValueError("time needs to be rut or ret")

    def check_solution(self):
        func_list = {
            'duration':     self.check_maintenance_duration
            ,'candidates':  self.check_resource_in_candidates
            ,'state':       self.check_resource_state
            ,'resources':   self.check_task_num_resources
            ,'usage':       self.check_resource_consumption
            ,'capacity':    self.check_maintenance_capacity
        }
        result = {k: v() for k, v in func_list.items()}
        return {k: v for k, v in result.items() if len(v) > 0}

    def check_maintenance_capacity(self):
        maints = self.solution.get_in_maintenance()
        return {k: v for k, v in maints.items() if v > self.instance.get_param('maint_capacity')}

    def check_task_num_resources(self):
        task_reqs = self.instance.get_tasks('num_resource')

        task_assigned = \
            aux.fill_dict_with_default(
                self.solution.get_task_num_resources(),
                self.instance.get_task_period_list()
            )
        task_under_assigned = {
            (task, period): task_reqs[task] - task_assigned[(task, period)]
            for (task, period) in task_assigned
            if task_reqs[task] - task_assigned[(task, period)] != 0
        }

        return task_under_assigned

    def check_resource_in_candidates(self):
        # task_data = self.instance.get_tasks()
        task_solution = self.solution.get_tasks()

        task_candidates = self.instance.get_task_candidates()
        # aux.get_property_from_dic(task_data, 'candidates')

        bad_assignment = {
            (resource, period): task
            for (resource, period), task in task_solution.items()
            if resource not in task_candidates[task]
        }
        return bad_assignment

    def get_consumption(self):
        hours = self.instance.get_tasks("consumption")
        return {k: hours[v] for k, v in self.solution.get_tasks().items()}

    def set_remainingtime(self, resource, period, time, value):
        self.expand_resource_period(self.solution.data['aux'][time], resource, period)
        self.solution.data['aux'][time][resource][period] = value
        return True

    def update_time_usage(self, resource, periods, previous_value=None, time='rut'):
        if previous_value is None:
            previous_value = self.solution.data['aux'][time][resource]\
                [aux.get_prev_month(periods[0])]
        for period in periods:
            value = previous_value - self.get_consumption_individual(resource, period, time)
            self.set_remainingtime(resource, period, time, value)
            previous_value = value
        return True

    def get_consumption_individual(self, resource, period, time='rut'):
        if time == 'ret':
            return 1
        task = self.solution.data['task'].get(resource, {}).get(period, '')
        if task == '':
            return 0
        return self.instance.data['tasks'].get(task, {}).get('consumption', 0)

    def get_non_maintenance_periods(self):
        first, last = self.instance.get_param('start'), self.instance.get_param('end')
        maintenances = aux.tup_to_dict(self.solution.get_maintenance_periods(), result_col=[1, 2])
        nonmaintenances = []
        resources_nomaint = [r for r in self.instance.get_resources() if r not in maintenances]
        for resource in resources_nomaint:
            nonmaintenances.append((resource, first, last))
        for resource in maintenances:
            maints = sorted(maintenances[resource], key=lambda x: x[0])
            first_maint_start = maints[0][0]
            last_maint_end = maints[-1][1]
            if first_maint_start != first:
                nonmaintenances.append((resource, first, aux.get_prev_month(first_maint_start)))
            for maint1, maint2 in zip(maints, maints[1:]):
                nonmaintenances.append(
                    (resource, aux.get_next_month(maint1[1]), aux.get_prev_month(maint2[0]))
                                       )
            if last_maint_end != last:
                nonmaintenances.append((resource, aux.get_next_month(last_maint_end), last))
        return nonmaintenances

    def set_remaining_usage_time(self, time="rut"):
        if 'aux' not in self.solution.data:
            self.solution.data['aux'] = {'ret': {}, 'rut': {}}
        else:
            self.solution.data['aux'][time] = {}

        label = 'initial_' + self.label_rt(time)
        prev_month = aux.get_prev_month(self.instance.get_param('start'))
        initial = self.instance.get_resources(label)

        label = 'max_' + self.label_rt(time) + '_time'
        max_rut = self.instance.get_param(label)
        for resource in initial:
            self.set_remainingtime(resource, prev_month, time, min(initial[resource], max_rut))

        maintenances = self.solution.get_maintenance_periods()
        for resource, start, end in maintenances:
            for period in aux.get_months(start, end):
                self.set_remainingtime(resource, period, time, max_rut)

        non_maintenances = self.get_non_maintenance_periods()
        for resource, start, end in non_maintenances:
            # print(resource, start, end)
            self.update_time_usage(resource, aux.get_months(start, end), time=time)

        return self.solution.data['aux'][time]

    def check_resource_consumption(self):
        rut = self.set_remaining_usage_time()
        rut_tup = aux.dictdict_to_dictup(rut)
        return {k: v for k, v in rut_tup.items() if v < 0}

    def check_resource_state(self):
        task_solution = self.solution.get_tasks()
        state_solution = self.solution.get_state()

        task_solution_k = np.fromiter(task_solution.keys(),
                                      dtype=[('A', '<U6'), ('T', 'U7')])
        state_solution_k = np.fromiter(state_solution.keys(),
                                      dtype=[('A', '<U6'), ('T', 'U7')])
        duplicated_states = \
            np.intersect1d(task_solution_k, state_solution_k)

        return [tuple(item) for item in duplicated_states]

    def check_maintenance_duration(self):
        maintenances = self.solution.get_maintenance_periods()
        first_period = self.instance.get_param('start')
        last_period = self.instance.get_param('end')
        duration = self.instance.get_param('maint_duration')

        maintenance_duration_incorrect = {}
        for (resource, start, finish) in maintenances:
            size_period = len(aux.get_months(start, finish))
            if size_period > duration:
                maintenance_duration_incorrect[(resource, start)] = size_period
            if size_period < duration and start != first_period and \
                            finish != last_period:
                maintenance_duration_incorrect[(resource, start)] = size_period
        return maintenance_duration_incorrect

    def get_objective_function(self):
        weight1 = self.instance.get_param("maint_weight")
        weight2 = self.instance.get_param("unavail_weight")
        unavailable = max(self.solution.get_unavailable().values())
        in_maint = max(self.solution.get_in_maintenance().values())
        return in_maint * weight1 + unavailable * weight2

    def get_kpis(self):
        rut = self.set_remaining_usage_time(time='rut')
        ret = self.set_remaining_usage_time(time='ret')
        end = self.instance.get_param('end')
        return {
            'unavail': max(self.solution.get_unavailable().values())
            , 'maint': max(self.solution.get_in_maintenance().values())
            , 'rut_end': sum(v[end] for v in rut.values())
            , 'ret_end': sum(v[end] for v in ret.values())
        }


def clean_experiments(path, clean=True, regex=""):
    """
    loads and cleans all experiments that are incomplete
    :param path: path to experiments
    :param clean: if set to false it only does not delete them
    :return: deleted experiments
    """
    exps_paths = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isdir(os.path.join(path, f))
                  if re.search(regex, f)
                  ]
    to_delete = []
    for e in exps_paths:
        exp = Experiment.from_dir(e, format="json")
        if exp is None:
            exp = Experiment.from_dir(e, format="pickle")
        to_delete.append(exp is None)
    exps_to_delete = np.array(exps_paths)[to_delete]
    if clean:
        for ed in exps_to_delete:
            shutil.rmtree(ed)
    return exps_to_delete


def exp_get_info(path):
    exp = Experiment.from_dir(path, format="json")
    if exp is None:
        exp = Experiment.from_dir(path, format="pickle")
    if exp is None:
        return None
    options_path = os.path.join(path, "options.json")
    options = di.load_data(options_path)
    if not options:
        return None
    log_path = os.path.join(path, "results.log")
    log_info = {}
    if os.path.exists(log_path):
        if options['solver'] == 'CPLEX':
            log_results = log.LogFile(log_path)
            log_info = log_results.get_log_info_cplex()
        elif options['solver'] == 'GUROBI':
            log_results = log.LogFile(log_path)
            log_info = log_results.get_log_info_gurobi()
    parameters = exp.instance.get_param()
    inst_info = exp.instance.get_info()
    return {**parameters, **options, **log_info, **inst_info}


def list_experiments(path):
    exps_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    experiments = {}
    for e in exps_paths:
        # print(e)
        info = exp_get_info(e)
        if info is None:
            continue
        directory = os.path.basename(e)
        experiments[directory] = info
    return experiments


if __name__ == "__main__":
    path_abs = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/"
    path_states = path_abs + "201712190002/"
    path_nostates = path_abs + "201712182321/"

    sol_states = Experiment.from_dir(path_states)
    sol_nostates = Experiment.from_dir(path_nostates)

    t = aux.tup_to_dict(aux.dict_to_tup(sol_nostates.solution.get_tasks()),
                    result_col= [0], is_list=True, indeces=[2, 1]).items()
    # {k: len(v) for k, v in t}
    sol_nostates.instance.get_tasks('num_resource')

    path = path_abs + "201801091412/"
    sol_states = Experiment.from_dir(path)
    check = sol_states.check_solution()
    # check['resources']


    rut_old = sol_states.solution.data["aux"]['rut']
    sol_states.set_remaining_usage_time()
    rut_new = sol_states.solution.data['aux']['rut']
    pd.DataFrame.from_dict(rut_new, orient='index').reset_index().melt(id_vars='index').\
        sort_values(['index', 'variable']).to_csv('/home/pchtsp/Downloads/TEMP_new.csv', index=False)
    pd.DataFrame.from_dict(rut_old, orient='index').reset_index().melt(id_vars='index').\
        sort_values(['index','variable']).apply('round').to_csv('/home/pchtsp/Downloads/TEMP_old.csv', index=False)

    # sol_nostates.check_solution()
    # sol_states.check_solution()

    # exp = Experiment(inst.Instance(model_data), sol.Solution(solution))
    # exp = Experiment.from_dir(path)
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

    path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments"
    exps = clean_experiments(path)
    # len(exps)

    # sol.Solution(solution).get_schedule()
    # check.check_resource_consumption()
    # check.check_resource_state()

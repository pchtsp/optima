# /usr/bin/python3
import package.auxiliar as aux
import numpy as np
import pandas as pd
import package.data_input as di
import package.solution as sol
import package.instance as inst
import package.tuplist as tl
import package.superdict as sd
import os
import shutil
import pprint as pp
import re
import package.logFiles as log


# TODO: create period objects with proper date methods based on arrow

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
            'candidates':  self.check_resource_in_candidates
            ,'state':       self.check_resource_state
            ,'resources':   self.check_task_num_resources
            ,'usage':       self.check_usage_consumption
            ,'elapsed':     self.check_elapsed_consumption
            ,'capacity':    self.check_maintenance_capacity
            ,'min_assign':  self.check_min_assignment
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
        """
        :param resource:
        :param period:
        :param time: ret or rut
        :param value: remaining time
        :return: True
        This procedure *updates* the remaining time in the aux property of the solution.
        """
        self.expand_resource_period(self.solution.data['aux'][time], resource, period)
        self.solution.data['aux'][time][resource][period] = value
        return True

    def update_time_usage(self, resource, periods, previous_value=None, time='rut'):
        """
        :param resource: a resource to update
        :param periods: a list of consecutive periods to update. ordered.
        :param previous_value: optional value for the remaining time before the first period
        :param time: rut or ret depending if it's usage time or elapsed time
        :return: True
        This procedure *updates* the time of each period using set_remainingtime.
        It assumes all periods do not have a maintenance.
        So the periods should be filled with a task or nothing.
        """
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
            return self.instance.get_param('min_usage_period')
        return self.instance.data['tasks'].get(task, {}).get('consumption', 0)

    def get_non_maintenance_periods(self):
        """
        :return: a dictionary with the following structure:
        resource: [(start_period1, end_period1), (start_period2, end_period2), ..., (start_periodN, end_periodN)]
        two consecutive periods being separated by a maintenance operation.
        It's built using the information of the maintenance operations.
        """
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
        max_rem = self.instance.get_param(label)
        for resource in initial:
            self.set_remainingtime(resource, prev_month, time, min(initial[resource], max_rem))

        maintenances = self.solution.get_maintenance_periods()
        for resource, start, end in maintenances:
            for period in aux.get_months(start, end):
                self.set_remainingtime(resource, period, time, max_rem)

        non_maintenances = self.get_non_maintenance_periods()
        for resource, start, end in non_maintenances:
            # print(resource, start, end)
            self.update_time_usage(resource, aux.get_months(start, end), time=time)

        return self.solution.data['aux'][time]

    def check_usage_consumption(self):
        return self.check_resource_consumption(time='rut')

    def check_elapsed_consumption(self):
        return self.check_resource_consumption(time='ret')

    def check_resource_consumption(self, time='rut'):
        rt = self.set_remaining_usage_time(time)
        rt_tup = aux.dictdict_to_dictup(rt)
        return {k: v for k, v in rt_tup.items() if v < 0}

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

    def check_min_assignment(self):
        """
        :return: periods were the min assignment (including maintenance)
        is not respected
        """
        tasks = sd.SuperDict(self.solution.get_tasks()).to_tuplist()
        maints = sd.SuperDict.from_dict(self.solution.get_state()).to_tuplist()
        previous = sd.SuperDict.from_dict(self.instance.get_resources("states")).\
            to_dictup().to_tuplist()
        min_assign = self.instance.get_min_assign()

        all_states = maints + tasks + previous
        all_states_periods = tl.TupList(all_states).tup_to_start_finish()

        # tasks_periods = self.solution.get_task_periods()
        # task_min_assign = self.instance.get_tasks('min_assign')
        first_period = self.instance.get_param('start')
        last_period = self.instance.get_param('end')

        incorrect = {}
        for (resource, start, state, finish) in all_states_periods:
            # periods that finish before the horizon
            # or after at the end
            # are not checked
            if finish < first_period or finish == last_period:
                continue
            size_period = len(aux.get_months(start, finish))
            if size_period < min_assign.get(state, 1):
                incorrect[(resource, start)] = size_period
        return incorrect

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
        starts = self.solution.get_maintenance_periods()
        return {
            'unavail': max(self.solution.get_unavailable().values())
            , 'maint': max(self.solution.get_in_maintenance().values())
            , 'rut_end': sum(v[end] for v in rut.values())
            , 'ret_end': sum(v[end] for v in ret.values())
            , 'maintenances': len(starts)
        }

    def export_solution(self, path, sheet_name='solution'):
        # path = "../data/parametres_DGA_final_test.xlsm"
        tasks = aux.dict_to_tup(self.solution.get_tasks())
        hours = self.instance.get_tasks('consumption')
        tasks_edited = [(t[0], t[1], '{} ({}h)'.format(t[2], hours[t[2]]))
                        for t in tasks]
        statesMissions = aux.dict_to_tup(self.solution.get_state()) + tasks_edited
        table = pd.DataFrame(statesMissions, columns=['resource', 'period', 'state'])
        table = table.pivot(index='resource', columns='period', values='state')
        table.to_excel(path, sheet_name=sheet_name)
        return table

    def get_status(self, candidate):
        """
        This function is great for debugging
        :param candidate: a resource
        :return: dataframe with everything that's going on with the resource
        """
        rut = pd.DataFrame.from_dict(self.solution.data['aux']['rut'].get(candidate, {}), orient='index')
        ret = pd.DataFrame.from_dict(self.solution.data['aux']['ret'].get(candidate, {}), orient='index')
        start = pd.DataFrame.from_dict(self.solution.data['aux']['start'].get(candidate, {}), orient='index')
        state = pd.DataFrame.from_dict(self.solution.data['state'].get(candidate, {}), orient='index')
        task = pd.DataFrame.from_dict(self.solution.data['task'].get(candidate, {}), orient='index')
        args = {'left_index': True, 'right_index': True, 'how': 'left'}
        table = rut.merge(ret, **args).merge(state, **args).merge(task, **args).merge(start, **args).sort_index()
        # table.columns = ['rut', 'ret', 'state', 'task']
        return table


def clean_experiments(path, clean=True, regex=""):
    """
    loads and cleans all experiments that are incomplete
    :param path: path to experiments
    :param clean: if set to false it only shows the files instead of deleting them
    :param regex: optional regex filter
    :return: deleted experiments
    """
    exps_paths = [os.path.join(path, f) for f in os.listdir(path)
                  if os.path.isdir(os.path.join(path, f))
                  if re.search(regex, f)
                  ]
    to_delete = []
    for e in exps_paths:
        exp = Experiment.from_dir(e, format="json")
        to_delete.append(exp is None)
    exps_to_delete = sorted(np.array(exps_paths)[to_delete])
    if clean:
        for ed in exps_to_delete:
            shutil.rmtree(ed)
    return exps_to_delete


def exp_get_info(path, get_log_info=True):
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
    if os.path.exists(log_path) and get_log_info:
        if options['solver'] == 'CPLEX':
            log_results = log.LogFile(log_path)
            log_info = log_results.get_log_info_cplex()
        elif options['solver'] == 'GUROBI':
            log_results = log.LogFile(log_path)
            log_info = log_results.get_log_info_gurobi()
    parameters = exp.instance.get_param()
    inst_info = exp.instance.get_info()
    return {**parameters, **options, **log_info, **inst_info}


def list_experiments(path, exp_list=None, get_log_info=True):
    if exp_list is None:
        exps_paths = [os.path.join(path, f) for f in os.listdir(path)
                      if os.path.isdir(os.path.join(path, f))]
    else:
        exps_paths = [os.path.join(path, f) for f in exp_list
                      if os.path.isdir(os.path.join(path, f))]
    experiments = {}
    for e in exps_paths:
        info = exp_get_info(e, get_log_info)
        if info is None:
            continue
        directory = os.path.basename(e)
        experiments[directory] = info
    return experiments


def list_options(path, exp_list=None):
    if exp_list is None:
        exps_paths = [os.path.join(path, f) for f in os.listdir(path)
                      if os.path.isdir(os.path.join(path, f))]
    else:
        exps_paths = [os.path.join(path, f) for f in exp_list
                      if os.path.isdir(os.path.join(path, f))]
    experiments = {}
    for e in exps_paths:
        options_path = os.path.join(e, "options.json")
        options = di.load_data(options_path)
        if not options:
            continue
        directory = os.path.basename(e)
        experiments[directory] = options
    return experiments


if __name__ == "__main__":
    import package.params as pm

    clean_experiments(pm.PATHS['experiments'], clean=True, regex=r'')
    pass
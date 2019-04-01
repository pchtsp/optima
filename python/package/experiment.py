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
import re
import orloge as ol

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

    def check_solution_count(self, **params):
        return self.check_solution(**params).to_lendict()

    def check_solution(self, **params):
        func_list = {
            'candidates':  self.check_resource_in_candidates
            ,'state':       self.check_resource_state
            ,'resources':   self.check_task_num_resources
            ,'usage':       self.check_usage_consumption
            ,'elapsed':     self.check_elapsed_consumption
            ,'capacity':    self.check_maintenance_capacity
            ,'min_assign':  self.check_min_max_assignment
            ,'available':   self.check_min_available
            ,'hours':       self.check_min_flight_hours
            ,'start_periods': self.check_fixed_assignments
            ,'dist_maints': self.check_min_distance_maints
        }
        result = {k: v(**params) for k, v in func_list.items()}
        return sd.SuperDict.from_dict({k: v for k, v in result.items() if len(v) > 0})

    def check_maintenance_capacity(self, **params):
        maints = self.solution.get_in_maintenance()
        return sd.SuperDict({(k, ): v for k, v in maints.items()
                                       if v > self.instance.get_param('maint_capacity')})

    def check_task_num_resources(self, strict=False, **params):
        task_reqs = self.instance.get_tasks('num_resource')

        task_assigned = \
            aux.fill_dict_with_default(
                self.solution.get_task_num_resources(),
                self.instance.get_task_period_list()
            )
        task_under_assigned = {
            (task, period): task_reqs[task] - task_assigned[task, period]
            for (task, period) in task_assigned
        }
        if strict:
            return sd.SuperDict(task_under_assigned).clean(func=lambda x: x != 0)
        return sd.SuperDict(task_under_assigned).clean(func=lambda x: x > 0)

    def check_resource_in_candidates(self, **params):
        task_solution = self.solution.get_tasks()

        task_candidates = self.instance.get_task_candidates()

        bad_assignment = {
            (resource, period): task
            for (resource, period), task in task_solution.items()
            if resource not in task_candidates[task]
        }
        return sd.SuperDict.from_dict(bad_assignment)

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

    def update_resource_all(self, resource):
        periods_to_update = self.instance.get_periods()
        for t in ['rut', 'ret']:
            self.update_time_usage(resource, periods_to_update, time=t)

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
                [self.instance.get_prev_period(periods[0])]
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

    def get_non_maintenance_periods(self, resource=None):
        """
        :return: a tuplist with the following structure:
        resource: [(resource, start_period1, end_period1), (resource, start_period2, end_period2), ..., (resource, start_periodN, end_periodN)]
        two consecutive periods being separated by a maintenance operation.
        It's built using the information of the maintenance operations.
        :param resource: if not None, we filter to only provide this resource's info
        """
        # a = self.get_status(resource)
        # a[a.period>='2019-02'][:8]
        first, last = self.instance.get_param('start'), self.instance.get_param('end')
        maintenances = aux.tup_to_dict(
            self.get_maintenance_periods(resource)
            , result_col=[1, 2])
        if resource is None:
            resources = self.instance.get_resources()
        else:
            resources = [resource]
        # we initialize nomaint periods for resources that do not have a single maintenance:
        nonmaintenances = [(r, first, last) for r in resources if r not in maintenances]
        # now, we iterate over all maintenances to add the before and the after
        for res in maintenances:
            maints = sorted(maintenances[res], key=lambda x: x[0])
            first_maint_start = maints[0][0]
            last_maint_end = maints[-1][1]
            if first_maint_start > first:
                nonmaintenances.append((res, first, self.instance.get_prev_period(first_maint_start)))
            for maint1, maint2 in zip(maints, maints[1:]):
                start = self.instance.get_next_period(maint1[1])
                end = self.instance.get_prev_period(maint2[0])
                nonmaintenances.append((res, start, end))
            if last_maint_end != last:
                start = self.instance.get_next_period(last_maint_end)
                nonmaintenances.append((res, start, last))
        return tl.TupList(nonmaintenances)

    def set_start_periods(self):
        """
        This function remakes the start tasks assignments and states (maintenances)
        It edits the aux part of the solution data
        :return: the dictionary that it assigns
        """
        tasks_start = self.get_task_periods()
        states_start = self.get_state_periods()
        all_starts = tasks_start + states_start
        starts = {(r, t): v for (r, t, v, _) in all_starts}

        if 'aux' not in self.solution.data:
            self.solution.data['aux'] = {'start': {}}
        else:
            self.solution.data['aux']['start'] = {}
        self.solution.data['aux']['start'] = starts

        return starts

    def set_remaining_usage_time(self, time="rut"):
        """
        This function remakes the rut and ret times for all resources.
        It assumes nothing of state or task.
        It edits the aux part of the solution data
        :param time: ret or rut
        :return: the dictionary that's assigned
        """
        if 'aux' not in self.solution.data:
            self.solution.data['aux'] = {'ret': {}, 'rut': {}}
        else:
            self.solution.data['aux'][time] = {}

        label = 'initial_' + self.label_rt(time)
        prev_month = self.instance.get_prev_period(self.instance.get_param('start'))
        initial = self.instance.get_resources(label)

        label = 'max_' + self.label_rt(time) + '_time'
        max_rem = self.instance.get_param(label)
        for resource in initial:
            self.set_remainingtime(resource, prev_month, time, min(initial[resource], max_rem))

        maintenances = self.get_maintenance_periods()
        for resource, start, end in maintenances:
            for period in self.instance.get_periods_range(start, end):
                self.set_remainingtime(resource, period, time, max_rem)

        non_maintenances = self.get_non_maintenance_periods()
        for resource, start, end in non_maintenances:
            # print(resource, start, end)
            self.update_time_usage(resource, self.instance.get_periods_range(start, end), time=time)

        return self.solution.data['aux'][time]

    def check_usage_consumption(self, **params):
        return self.check_resource_consumption(time='rut', **params)

    def check_elapsed_consumption(self, **params):
        return self.check_resource_consumption(time='ret', **params)

    def check_resource_consumption(self, time='rut', recalculate=True, **params):
        if recalculate:
            rt = self.set_remaining_usage_time(time)
        else:
            rt = self.solution.data['aux'][time]

        rt_tup = aux.dictdict_to_dictup(rt)
        return sd.SuperDict({k: v for k, v in rt_tup.items() if v < 0})

    def check_resource_state(self, **params):
        task_solution = self.solution.get_tasks()
        state_solution = self.solution.get_state()

        task_solution_k = np.fromiter(task_solution.keys(),
                                      dtype=[('A', '<U6'), ('T', 'U7')])
        state_solution_k = np.fromiter(state_solution.keys(),
                                       dtype=[('A', '<U6'), ('T', 'U7')])
        duplicated_states = \
            np.intersect1d(task_solution_k, state_solution_k)

        return sd.SuperDict({tuple(item): 1 for item in duplicated_states})

    def check_min_max_assignment(self, **params):
        """
        :return: periods were the min assignment (including maintenance)
        in format: (resource, start, end): error.
        if error negative: bigger than max. Otherwise: less than min
        is not respected
        """
        # TODO: do it with self.solution.get_schedule()
        tasks = self.solution.get_tasks().to_tuplist()
        maints = self.solution.get_state().to_tuplist()
        previous = sd.SuperDict.from_dict(self.instance.get_resources("states")).\
            to_dictup().to_tuplist()
        min_assign = self.instance.get_min_assign()
        max_assign = self.instance.get_max_assign()

        num_periods = self.instance.get_param('num_period')
        ct = self.instance.compare_tups
        all_states = maints + tasks + previous
        all_states_periods = tl.TupList(all_states).tup_to_start_finish(compare_tups=ct)

        first_period = self.instance.get_param('start')
        last_period = self.instance.get_param('end')

        incorrect = {}
        for (resource, start, state, finish) in all_states_periods:
            # periods that finish before the horizon
            # or at the end are not checked
            if finish < first_period or finish == last_period:
                continue
            size_period = len(self.instance.get_periods_range(start, finish))
            if size_period < min_assign.get(state, 1):
                incorrect[resource, start, finish] = min_assign[state] - size_period
            elif size_period > max_assign.get(state, num_periods):
                incorrect[resource, start, finish] = max_assign[state] - size_period
        return sd.SuperDict(incorrect)

    def check_fixed_assignments(self, **params):
        first_period = self.instance.get_param('start')
        last_period = self.instance.get_param('end')
        state_tasks = self.solution.get_state_tasks()
        fixed_states = self.instance.get_fixed_states()
        fixed_states_h = fixed_states.filter_list_f(lambda x: first_period <= x[2] <= last_period)
        state_tasks_tab = pd.DataFrame.from_records(list(state_tasks),
                                                    columns=['resource', 'period', 'state'])
        fixed_states_tab = pd.DataFrame.from_records(list(fixed_states_h),
                                                     columns=['resource', 'state', 'period'])
        result = pd.merge(fixed_states_tab, state_tasks_tab, how='left', on=['resource', 'period'])
        return sd.SuperDict({tuple(x): 1 for x in
                             result[result.state_x != result.state_y].to_records(index=False)})


    def check_min_available(self, **params):
        """
        :return: periods where the min availability is not guaranteed.
        """
        resources = self.instance.get_resources().keys()
        c_candidates = self.instance.get_cluster_candidates()
        cluster_data = self.instance.get_cluster_constraints()
        maint_periods = tl.TupList(self.get_maintenance_periods()).\
            to_dict(result_col=[1, 2]).fill_with_default(keys=resources, default=[])
        max_candidates = cluster_data['num']
        num_maintenances = sd.SuperDict().fill_with_default(max_candidates.keys())
        for cluster, candidates in c_candidates.items():
            for candidate in candidates:
                for maint_period in maint_periods[candidate]:
                    for period in self.instance.get_periods_range(*maint_period):
                        if (cluster, period) in num_maintenances:
                            num_maintenances[cluster, period] += 1
        over_assigned = sd.SuperDict({k: max_candidates[k] - v for k, v in num_maintenances.items()})
        return over_assigned.clean(func=lambda x: x < 0)
        # return over_assigned
        # cluster_data.keys()

    def check_min_flight_hours(self, recalculate=True, **params):
        """
        :return: periods where the min flight hours is not guaranteed.
        """
        c_candidates = self.instance.get_cluster_candidates()
        cluster_data = self.instance.get_cluster_constraints()
        min_hours = cluster_data['hours']
        if recalculate:
            ruts = self.set_remaining_usage_time('rut')
        else:
            ruts = self.solution.data['aux']['rut']
        cluster_hours = sd.SuperDict().fill_with_default(min_hours.keys())
        for cluster, candidates in c_candidates.items():
            for candidate in candidates:
                for period, hours in ruts[candidate].items():
                    if period >= self.instance.get_param('start'):
                        cluster_hours[cluster, period] += hours

        hours_deficit = sd.SuperDict({k: v - min_hours[k] for k, v in cluster_hours.items()})
        return hours_deficit.clean(func=lambda x: x < 0)

    def check_min_distance_maints(self, **params):
        """
        checks if maintenances have the correct distance between them
        :return:
        """
        maints = self.get_maintenance_starts()
        maints_res = tl.TupList(maints).to_dict(result_col=1)
        errors = {}
        duration = self.instance.get_param('maint_duration')
        max_dist = self.instance.get_param('max_elapsed_time') + duration
        size = self.instance.get_param('elapsed_time_size')
        min_dist = max_dist - size

        for res, periods in maints_res.items():
            periods.sort()
            for period1, period2 in zip(periods, periods[1:]):
                dist = self.instance.get_dist_periods(period1, period2)
                if dist < min_dist:
                    errors[res, period1, period2] = dist - min_dist
                elif dist > max_dist:
                    errors[res, period1, period2] = dist - max_dist

        return sd.SuperDict.from_dict(errors)

    def check_maint_size(self, **params):
        tups_func = self.instance.compare_tups
        maint_periods = self.get_maintenance_periods()
        # we check if the size of the maintenance is equal to its duration

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
        starts = self.get_maintenance_periods()
        return {
            'unavail': max(self.solution.get_unavailable().values())
            , 'maint': max(self.solution.get_in_maintenance().values())
            , 'rut_end': sum(v[end] for v in rut.values())
            , 'ret_end': sum(v[end] for v in ret.values())
            , 'maintenances': len(starts)
        }

    def export_solution(self, path, sheet_name='solution'):
        tasks = aux.dict_to_tup(self.solution.get_tasks())
        hours = self.instance.get_tasks('consumption')
        tasks_edited = [(t[0], t[1], '{} ({}h)'.format(t[2], hours[t[2]]))
                        for t in tasks]
        statesMissions = aux.dict_to_tup(self.solution.get_state()) + tasks_edited
        table = pd.DataFrame(statesMissions, columns=['resource', 'period', 'state'])
        table = table.pivot(index='resource', columns='period', values='state')
        table.to_excel(path, sheet_name=sheet_name)
        return table

    def get_state_periods(self, resource=None):
        # TODO: we're filtering too much here. Not all states are maints
        # (although for now this is the case)
        previous_states = self.instance.get_prev_states(resource).\
            filter_list_f(lambda x: x[2]=='M')
        states = self.solution.get_state(resource).to_tuplist()
        previous_states.extend(states)

        ct = self.instance.compare_tups
        return previous_states.unique2().tup_to_start_finish(compare_tups=ct)

    def get_maintenance_periods(self, resource=None):
        result = self.get_state_periods(resource)
        return [(k[0], k[1], k[3]) for k in result if k[2] == 'M']

    def get_task_periods(self, resource=None):
        tasks = set(self.instance.get_tasks().keys())
        previous_tasks = self.instance.get_prev_states(resource). \
            filter_list_f(lambda x: x[2] in tasks)
        ct = self.instance.compare_tups
        tasks_assigned = self.solution.get_tasks().to_tuplist()
        previous_tasks.extend(tasks_assigned)
        return previous_tasks.tup_to_start_finish(compare_tups=ct)

    def get_maintenance_starts(self):
        maintenances = self.get_maintenance_periods()
        return [(r, s) for (r, s, e) in maintenances]

    def get_status(self, candidate):
        """
        This function is great for debugging
        :param candidate: a resource
        :return: dataframe with everything that's going on with the resource
        """
        data = self.solution.data
        if 'aux' not in data:
            data['aux'] = {}
        for t in ['rut', 'ret']:
            if t not in data['aux']:
                self.set_remaining_usage_time(t)
        if 'start' not in data['aux']:
            self.set_start_periods()
        rut = pd.DataFrame.from_dict(data['aux']['rut'].get(candidate, {}), orient='index')
        ret = pd.DataFrame.from_dict(data['aux']['ret'].get(candidate, {}), orient='index')
        start = pd.DataFrame.from_dict(data['aux']['start'].get(candidate, {}), orient='index')
        state = pd.DataFrame.from_dict(data['state'].get(candidate, {}), orient='index')
        task = pd.DataFrame.from_dict(data['task'].get(candidate, {}), orient='index')
        args = {'left_index': True, 'right_index': True, 'how': 'left'}
        table = rut.merge(ret, **args).merge(state, **args).merge(task, **args).merge(start, **args).sort_index()
        return table.reset_index().rename(columns={'index': 'period'})
        # table.columns = ['rut', 'ret', 'state', 'task']
        # return table


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


def exp_get_info(path, get_log_info=True, get_exp_info=True):
    # print(path)
    parameters = log_info = inst_info = {}
    if get_exp_info:
        exp = Experiment.from_dir(path, format="json")
        if exp is None:
            return None
        parameters = exp.instance.get_param()
        inst_info = exp.instance.get_info()
    options_path = os.path.join(path, "options.json")
    options = di.load_data(options_path)
    if not options:
        return None
    log_path = os.path.join(path, "results.log")
    if os.path.exists(log_path) and get_log_info:
        log_info = ol.get_info_solver(log_path, options['solver'])
    return {**parameters, **options, **log_info, **inst_info}


def list_experiments(path, exp_list=None, **kwags):
    if exp_list is None:
        exp_list = os.listdir(path)
    exps_paths = [os.path.join(path, f) for f in exp_list
                  if os.path.isdir(os.path.join(path, f))]
    experiments = {}
    for e in exps_paths:
        print(e)
        info = exp_get_info(e, **kwags)
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
# /usr/bin/python3

import numpy as np
import package.auxiliar as aux
import package.data_input as di
import pandas as pd
import math
import pytups.superdict as sd
import pytups.tuplist as tl
import arrow


class Instance(object):
    """
    This represents the input data set used to solve.
    It doesn't include the solution.
    The methods help get useful information from the dataset.
    The data is stored in .data at it consists on a dictionary of dictionaries.
    The structure will vary in the future but right now is:
        * parameters: adimentional data
        * tasks: data related to tasks
        * resources: data related to resources
        * aux: cached data (months, for example)
    """

    def __init__(self, model_data):
        self.data = model_data
        params = {
            'maint_weight': 1
            , 'unavail_weight': 1
            , 'min_usage_period': 0
            , 'min_avail_percent': 0.1
            , 'min_avail_value': 1
            , 'min_hours_perc': 0.2
        }
        params.update(self.data['parameters'])
        self.data['parameters'] = params

        start = self.get_param('start')
        num_periods = self.get_param('num_period')
        self.data['aux'] = {}

        begin = aux.month_to_arrow(start).shift(months=-50)
        end = aux.month_to_arrow(start).shift(months=num_periods+50)
        self.data['aux']['period_e'] = {
            k- 50:  r.format("YYYY-MM") for k, r in enumerate(arrow.Arrow.range('month', begin, end))
        }
        self.data['aux']['period_i'] = {
            v: k for k, v in self.data['aux']['period_e'].items()
        }

    def get_param(self, param=None):
        params = self.data['parameters']
        if param is not None:
            try:
                return params[param]
            except:
                raise ValueError("param named {} does not exist in parameters".format(param))
        return params

    def get_categories(self):
        result = {}
        for category, value in self.data.items():
            # if type(value) is dict:
            elem = list(value.keys())[0]
            if isinstance(value[elem], dict):
                result[category] = list(value[elem].keys())
            else:
                result[category] = list(value.keys())
        return result

    def get_category(self, category, param=None, default_dict=None):
        assert category in self.data
        data = self.data[category]
        if default_dict is not None:
            data = {k: {**default_dict, **v} for k, v in data.items()}
        if param is None:
            return data
        if param in list(data.values())[0]:
            return aux.get_property_from_dic(data, param)
        raise IndexError("param {} is not present in the category {}".format(param, category))

    def get_tasks(self, param=None):
        default_tasks = {'min_assign': 1, 'min_usage_period':0}
        return self.get_category('tasks', param, default_tasks)

    def get_resources(self, param=None):
        default_resources = {'states': {}}
        return self.get_category('resources', param, default_resources)

    def get_initial_state(self, time_type, resource=None):
        """
        Returns the correct initial states for resources.
        It corrects it using the max and whether it is in maintenance.
        :param time_type: elapsed or used
        :param resource: optional value to filter only one resource
        :return:
        """
        first_period = self.get_param('start')
        prev_first_period = self.get_prev_period(first_period)
        if time_type not in ["elapsed", "used"]:
            raise KeyError("Wrong type in time_type parameter: elapsed or used only")

        key_initial = "initial_" + time_type
        key_max = "max_" + time_type + "_time"
        param_resources = sd.SuperDict(self.get_resources())
        if resource is not None:
            param_resources.filter(resource)
        rt_max = self.get_param(key_max)

        rt_read = aux.get_property_from_dic(param_resources, key_initial)

        # we also check if the resources is currently in maintenance.
        # If it is: we assign the rt_max (according to convention).
        res_maints = \
            self.get_fixed_maintenances(resource=resource).\
            filter_list_f(lambda x: x[1] >= prev_first_period).\
            to_dict(result_col=1).\
            to_lendict()

        if time_type == 'elapsed':
            rt_fixed = {k: rt_max + v - 1 for k, v in res_maints.items()}
        else:
            rt_fixed = {k: rt_max for k, v in res_maints.items()}

        rt_init = {a: rt_max for a in param_resources}
        rt_init.update(rt_read)
        rt_init.update(rt_fixed)

        # rt_init = {k: min(rt_max, v) for k, v in rt_init.items()}

        return rt_init

    def get_min_assign(self):
        min_assign = dict(self.get_tasks('min_assign'))
        min_assign['M'] = self.get_param('maint_duration')
        return min_assign

    def get_max_assign(self):
        max_assign = dict(M = self.get_param('maint_duration'))
        return sd.SuperDict(max_assign)

    def compare_tups(self, tup1, tup2, pp):
        for n, (v1, v2) in enumerate(zip(tup1, tup2)):
            if n == pp:
                if v1 != self.get_next_period(v2):
                    return True
            else:
                if v1 != v2:
                    return True
        return False

    def get_prev_states(self, resource=None):
        previous_states = sd.SuperDict.from_dict(self.get_resources("states"))
        if resource is not None:
            previous_states = previous_states.filter(resource)
        return previous_states.to_dictup().to_tuplist()

    def get_fixed_states(self, resource=None, filter_horizon=False):
        """
        This function returns the fixed states in the beginning of the planning period
        They can be maintenances or mission assignments
        They include previous states too
        :param resource: if given filters only for that resource
        :return:
        """
        previous_states = self.get_prev_states(resource)
        first_period = self.get_param('start')
        period_0 = self.get_prev_period(first_period)
        min_assign = self.get_min_assign()
        # we get the states into a tuple list,
        # we turn them into a start-finish tuple
        # we filter it so we only take the start-finish periods that end before the horizon
        assignments = \
            previous_states.to_start_finish(compare_tups=self.compare_tups).\
                filter_list_f(lambda x: x[3] == period_0)

        fixed_assignments_q = \
            [(a[0], a[2], min_assign.get(a[2], 0) - len(self.get_periods_range(a[1], a[3])))
             for a in assignments if len(self.get_periods_range(a[1], a[3])) < min_assign.get(a[2], 0)]

        fixed_future = tl.TupList(
            [(f_assign[0], f_assign[1], self.shift_period(first_period, t))
             for f_assign in fixed_assignments_q for t in range(f_assign[2])]
            )
        previous_n = tl.TupList((r, s, t) for r, t, s in previous_states)
        if not filter_horizon:
            fixed_future.extend(previous_n)
        return fixed_future

    def get_fixed_maintenances(self, dict_key=None, resource=None):
        fixed_states = self.get_fixed_states(resource)
        fixed_maints = tl.TupList([(a, t) for (a, s, t) in fixed_states if s == 'M'])
        if dict_key is None:
            return fixed_maints
        if dict_key == 'resource':
            return fixed_maints.to_dict(result_col=1)
        if dict_key == 'period':
            return fixed_maints.to_dict(result_col=0)

    def get_fixed_tasks(self):
        tasks = self.get_tasks()
        states = self.get_fixed_states()
        return tl.TupList([(a, s, t) for (a, s, t) in states if s in tasks])

    def get_fixed_periods(self):
        states = self.get_fixed_states()
        return states.filter([0, 2])
        # return tl.TupList([(a, t) for (a, s, t) in states]).unique()

    def get_task_period_list(self, in_dict=False):

        task_periods = {task:
            np.intersect1d(
                self.get_periods_range(self.get_tasks('start')[task], self.get_tasks('end')[task]),
                self.get_periods()
            ) for task in self.get_tasks()
        }
        if in_dict:
            return task_periods
        return [(task, period) for task in self.get_tasks() for period in task_periods[task]]

    def get_periods(self):
        return self.get_periods_range(self.get_param('start'), self.get_param('end'))

    def get_periods_range(self, start, end):
        pos_period = self.data['aux']['period_i']
        period_pos = self.data['aux']['period_e']
        return [period_pos[t] for t in range(pos_period[start], pos_period[end]+1)]

    def get_dist_periods(self, start, end):
        pos_period = self.data['aux']['period_i']
        return pos_period[end] - pos_period[start]

    def shift_period(self, period, num=1):
        pos_period = self.data['aux']['period_i']
        period_pos = self.data['aux']['period_e']
        return period_pos[pos_period[period] + num]

    def get_next_period(self, period):
        return self.shift_period(period, 1)

    def get_prev_period(self, period):
        return self.shift_period(period, -1)

    def get_next_periods(self, period, num=1, previous=False):
        """
        get num next periods. The include the reference as the first one.
        Or last one in the case of previous.
        :param period:
        :param num:
        :param previous: boolean to change the sense of the search
        :return:
        """
        if num < 0:
            return []
        pos_period = self.data['aux']['period_i']
        period_pos = self.data['aux']['period_e']
        if not previous:
            start = pos_period[period]
            end = start + num
        else:
            end = pos_period[period] + 1
            start = end - num
        return [period_pos[t] for t in range(start, end)]

    def get_task_period_needs(self):
        requirement = self.get_tasks('num_resource')
        needs = {(v, t): requirement[v] for (v, t) in self.get_task_period_list()}
        return needs

    def get_total_period_needs(self):
        num_resource_working = {t: 0 for t in self.get_periods()}
        task_period_needs = self.get_task_period_needs()
        for (v, t), req in task_period_needs.items():
            num_resource_working[t] += req
        return num_resource_working

    def get_total_fixed_maintenances(self):
        return self.get_fixed_maintenances().to_dict(result_col=0, is_list=True).to_lendict()

    def check_enough_candidates(self):
        task_num_resources = self.get_tasks('num_resource')
        task_num_candidates = {task: len(candidates) for task, candidates in self.get_task_candidates().items()}
        task_slack = {task: task_num_candidates[task] - task_num_resources[task] for task in task_num_resources}

        return {k: (v, v / task_num_resources[k]) for k, v in task_slack.items()}

    def get_info(self):
        task_period = sd.SuperDict.from_dict(self.get_task_period_list(True))
        assign = \
            sum(v * self.data['tasks'][k]['num_resource']
                for k, v in task_period.to_lendict().items())

        return {
            'periods': len(self.get_periods()),
            'tasks': len(self.get_tasks()),
            'assignments': assign
        }

    def get_clusters(self):
        present_group = 1
        if 'capacities' not in self.get_categories()['resources']:
            # if we're in the old instances: we do not form clusters
            return {t: present_group for t in self.get_tasks()}
        # initialized group.
        group = {}
        capacities = self.get_tasks('capacities')
        for task1, cap1 in capacities.items():
            if task1 in group:
                continue
            group[task1] = str(present_group)
            present_group += 1
            for task2, cap2 in capacities.items():
                if task2 in group:
                    continue
                if len(set(cap1).symmetric_difference(set(cap2))) == 0:
                    group[task2] = group[task1]
        return group

    def get_cluster_needs(self):
        cluster = self.get_clusters()
        task_needs = self.get_task_period_needs()
        cluster_needs = {(c, period): 0
                         for c in cluster.values()
                         for period in self.get_periods()}
        for (task, period), value in task_needs.items():
            cluster_needs[(cluster[task], period)] += value
        return cluster_needs

    def get_cluster_constraints(self):
        min_percent = self.get_param('min_avail_percent')
        min_value = self.get_param('min_avail_value')
        hour_perc = self.get_param('min_hours_perc')
        # num_periods = len(self.get_periods())

        # cluster availability will now mean:
        # resources that are not under maintenance
        cluster = self.get_clusters()
        kt = [(c, period) for c in cluster.values() for period in self.get_periods()]
        num_res_maint = \
            sd.SuperDict(self.get_fixed_maintenances_cluster()).\
                to_lendict().\
                fill_with_default(keys=kt)
        c_num_candidates = sd.SuperDict(self.get_cluster_candidates()).to_lendict()
        c_slack = {tup: c_num_candidates[tup[0]] - num_res_maint[tup]
                   for tup in kt}
        c_min = {(k, t): min(
            c_slack[(k, t)],
            max(
                math.ceil(c_num_candidates[k] * min_percent),
                min_value)
            ) for (k, t) in c_slack
        }
        #
        c_needs_num = {(k, t): c_num_candidates[k] - c_min[k, t] - num_res_maint[k, t]
                       for k, t in kt
                       }
        c_needs_hours = {(k, t): v * self.get_param('max_used_time') * hour_perc
                         for k, v in c_num_candidates.items() for t in self.get_periods()}

        return {'num': c_needs_num, 'hours': c_needs_hours}

    def get_task_candidates(self, recalculate=True, task=None):

        # we check if these are old instances:
        if 'capacities' not in self.get_categories()['resources']:
            t_candidates = self.get_tasks('candidates')
            if task is not None:
                return t_candidates[task]
            return t_candidates

        # if they're the 'newer' instances:
        r_cap = self.get_resources('capacities')
        t_cap = self.get_tasks('capacities')

        t_candidates = {t: [] for t in t_cap}
        for t, task_caps in t_cap.items():
            for res, res_caps in r_cap.items():
                if len(set(task_caps) - set(res_caps)) == 0:
                    t_candidates[t].append(res)

        if task is not None:
            return t_candidates[task]
        return t_candidates

    def get_cluster_candidates(self):
        # Since clusters are strict, their candidates are the same as the tasks.
        c_candidates = {}
        t_candidates = self.get_task_candidates()
        cluster = self.get_clusters()
        for k, v in t_candidates.items():
            c_candidates[cluster[k]] = v
        return c_candidates

    def get_fixed_maintenances_cluster(self):
        fixed_per_period = self.get_fixed_maintenances(dict_key='period')
        candidates_per_cluster = self.get_cluster_candidates()
        fixed_per_period_cluster = {}
        for period, resources in fixed_per_period.items():
            for cluster, candidates in candidates_per_cluster.items():
                fixed_per_period_cluster[(cluster, period)] =\
                    np.intersect1d(resources, candidates)
        return fixed_per_period_cluster

    def get_types(self):
        values = self.get_tasks('type_resource').values()
        return tl.TupList(values).unique2()

    def get_stats(self):
        """
        These stats are useful to characterize an instance and to get better bounds
        :return:
        """
        param_data = self.get_param()
        first_period, last_period = param_data['start'], param_data['end']
        periods = self.get_periods()
        periods_pos = self.data['aux']['period_i']
        resources = sd.SuperDict.from_dict(self.get_resources())
        tasks = sd.SuperDict.from_dict(self.get_tasks())
        ret_init = resources.get_property("initial_elapsed")
        duration = param_data['maint_duration']
        max_elapsed = param_data['max_elapsed_time'] + duration
        ret_init_adjusted = ret_init.apply(lambda _, v: v - max_elapsed + min_elapsed)

        first_maint_range = tl.TupList((a, t) for a in resources for t in periods
                              if ret_init[a] <= len(periods)
                              if ret_init_adjusted[a] <= periods_pos[t] <= ret_init[a]
                              ).\
            to_dict(result_col=1).\
            apply(lambda _, v: (v[0], v[-1]))

        second_maint_range = first_maint_range.\
            apply(lambda _, v: (self.shift_period(v[0], min_elapsed),
                                self.shift_period(v[1], max_elapsed))
        )
        mandatory_two_maints = second_maint_range.clean(func=lambda x: x[1] <= last_period)
        mandatory_one_maint = second_maint_range.clean(func=lambda x: x[0] > last_period)
        one_or_two_maints = {k: v for k, v in second_maint_range.items() if k not in mandatory_two_maints and
                             k not in mandatory_one_maint}

        type_tasks = tasks.get_property('type_resource').to_tuplist().to_dict(result_col=0)
        task_consumption = tasks.apply(lambda _, v: self.get_dist_periods(v['start'], v['end'])*v['consumption'])
        type_consumption = type_tasks.apply(lambda _, v: sum(task_consumption[t] for t in v))

        type_resource = resources.get_property('type').to_tuplist().to_dict(result_col=0)
        res_remaining = resources.get_property('initial_used')
        type_initial = type_resource.apply(lambda _, v: sum(res_remaining[r] for r in v))

        type_maints = []


        tasks.get_property('consumption')



if __name__ == "__main__":
    # path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712191655/"
    # model_data = di.load_data(path + "data_in.json")
    import package.simulation as sim
    import package.params as params

    options = params.OPTIONS
    model_data = sim.create_dataset(options)
    instance = Instance(model_data)
    instance.get_stats()
    # instance.get_categories()
    # result = instance.get_total_fixed_maintenances()

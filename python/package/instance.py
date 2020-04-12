# /usr/bin/python3

import numpy as np

import math
import pytups.superdict as sd
import pytups.tuplist as tl
import ujson as json


class Instance(object):
    """
    This represents the input data set used to solve.
    It doesn't include the solution.
    The methods help get useful information from the dataset.
    The data is stored in .data at it consists on a dictionary of dictionaries.
    The structure is:

    - parameters: 1-dimension data
    - tasks: data related to tasks
    - resources: data related to resources
    - maintenances: data related to maintenances (smaller for now)
    - maint_types: data related to capacity or maintenance type
    - aux: cached data (months, for example)
    """

    def __init__(self, model_data):
        self.data = sd.SuperDict.from_dict(model_data)
        self.set_date_params()
        self.set_default_params()
        self.set_default_resources()
        for time_type in ['used', 'elapsed']:
            self.correct_initial_state(time_type=time_type)
        self.set_default_maintenances()
        self.prepare_maint_params()
        self.set_default_maint_types()

    @classmethod
    def from_instance(cls, instance):
        data = instance.data
        data_copy = json.loads(json.dumps(data))
        new_instance = cls(sd.SuperDict.from_dict(data_copy))
        return new_instance

    def set_default_params(self):
        # TODO: take out deprecated params
        params = {
            'maint_weight': 1
            , 'unavail_weight': 1
            , 'min_usage_period': 0
            , 'min_avail_percent': 0.1
            , 'min_avail_value': 1
            , 'min_hours_perc': 0.2
            , 'default_type2_capacity': 66
        }

        params = sd.SuperDict.from_dict(params)
        params.update(self.data['parameters'])
        self.data['parameters'] = params

    def set_default_maintenances(self):
        params = self.data['parameters']
        if 'maint_duration' not in params:
            # we're using the new parameters, no need for defaults
            return
        resources = self.data['resources']

        maints = {
            'M': {
                'duration_periods': params['maint_duration']
                , 'capacity_usage': 1
                , 'max_used_time': params['max_used_time']
                , 'max_elapsed_time': params['max_elapsed_time']
                , 'elapsed_time_size': params['elapsed_time_size']
                , 'used_time_size': params['max_used_time']
                , 'type': '1'
                , 'capacity': params['maint_capacity']
                , 'depends_on': []
            }
        }
        if 'maintenances' in self.data:
            maints = sd.SuperDict.from_dict(maints)
            maints.update(self.data['maintenances'])
        self.data['maintenances'] = sd.SuperDict.from_dict(maints)

        if 'initial' not in resources.values_l()[0]:
            data_resources = self.data['resources']
            rut_init = data_resources.get_property('initial_used')
            ret_init = data_resources.get_property('initial_elapsed')
            for r in resources:
                self.data['resources'][r]['initial'] = \
                sd.SuperDict(M=sd.SuperDict(elapsed=ret_init[r], used=rut_init[r]))
        return

    def set_default_maint_types(self):
        types = self.get_maintenances('type').values()
        tups = [('maint_types', t, 'capacity') for t in set(types)]
        for tup in tups:
            if self.data.get_m(*tup) is not None:
                continue
            self.data.set_m(*tup, value={})
        return

    def set_default_resources(self):
        params = self.data['parameters']
        resources = sd.SuperDict.from_dict(self.data['resources'])
        default_keys = ('min_usage_period', 'default')
        default_value = params['min_usage_period']
        for r in resources:
            if resources[r].get_m(*default_keys) is None:
                resources[r].set_m(*default_keys, value=default_value)
        self.data['resources'] = resources
        return

    def prepare_maint_params(self):
        maints = self.data['maintenances']
        depends_on = sd.SuperDict(maints).get_property('depends_on')
        for m in depends_on:
            if m not in maints[m]['depends_on']:
                maints[m]['depends_on'].append(m)
        affects = depends_on.list_reverse()
        for m, v in affects.items():
            maints[m]['affects'] = v
        return maints

    def set_date_params(self):
        """
        This makes a cache of periods to later access it to move
        between periods inside the planning horizon
        :return:
        """
        start = self.get_param('start')
        num_periods = self.get_param('num_period')
        self.data['aux'] = sd.SuperDict()

        begin_year = int(start[:4]) - 5
        end_year = int(start[:4]) + 5 + num_periods//12
        many_months = ['{}-{:02.0f}'.format(_a, _b)
                       for _a in range(begin_year, end_year)
                       for _b in range(1, 13)]
        pos = many_months.index(start)
        self.data['aux']['period_e'] = {
            k- pos: r for k, r in enumerate(many_months)
        }
        self.data['aux']['period_i'] = {
            v: k for k, v in self.data['aux']['period_e'].items()
        }
        # we need to guarantee consistency between start, num_periods and end
        self.data['parameters']['end'] = self.shift_period(start, num_periods-1)

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
            if not len(value):
                result[category] = []
                continue
            elem = list(value.keys())[0]
            if isinstance(value[elem], dict):
                result[category] = list(value[elem].keys())
            else:
                result[category] = list(value.keys())
        return result

    def get_category(self, category, param=None, default_dict=None):
        assert category in self.data
        data = self.data[category]
        if not data:
            return sd.SuperDict()
        if default_dict is not None:
            default_dict = sd.SuperDict.from_dict(default_dict)
            data = data.vapply(lambda v: sd.SuperDict({**default_dict, **v}))
        if param is None:
            return data
        if param in list(data.values())[0]:
            return data.get_property(param)

        raise IndexError("param {} is not present in the category {}".format(param, category))

    def get_tasks(self, param=None):
        default_tasks = {'min_assign': 1, 'min_usage_period':0}
        return self.get_category('tasks', param, default_tasks)

    def get_resources(self, param=None):
        default_resources = {'states': {}}
        return self.get_category('resources', param, default_resources)

    def get_maintenances(self, param=None):
        default = {'priority': 1}
        return self.get_category('maintenances', param, default)

    def correct_initial_state(self, time_type):
        """
        Returns the correct initial states for resources.
        It corrects it using the max and whether it is in maintenance.
        :param time_type:
        :return:
        """
        first_period = self.get_param('start')
        prev_first_period = self.get_prev_period(first_period)

        if time_type not in ["elapsed", "used"]:
            raise KeyError("Wrong type in time_type parameter: elapsed or used only")

        key_initial = "initial_" + time_type
        key_max = "max_" + time_type + "_time"
        resources = sd.SuperDict(self.get_resources())
        res1 = resources.keys_l()[0]
        if key_initial not in res1:
            # this means we're already using the good nomenclature
            return
        rt_read = resources.get_property(key_initial)
        rt_max = self.get_param(key_max)


        # here, we calculate the number of fixed maintenances.
        res_maints = \
            self.get_fixed_maintenances().\
            vfilter(lambda x: x[1] >= prev_first_period).\
            to_dict(result_col=1).\
            to_lendict()

        if time_type == 'elapsed':
            # this extra is only for ret, not for rut
            rt_fixed = {k: rt_max + v - 1 for k, v in res_maints.items()}
        else:
            rt_fixed = {k: rt_max for k, v in res_maints.items()}

        rt_init = dict(rt_read)
        rt_init.update(rt_fixed)
        for r in rt_init:
            self.data['resources'][r]['M'][key_initial] = rt_init[r]
        return rt_init


    def get_initial_state(self, time_type, resource=None, maint='M'):
        """
        :param time_type: elapsed or used
        :param resource: optional value to filter only one resource
        :return:
        """
        if time_type not in ["elapsed", "used"]:
            raise KeyError("Wrong type in time_type parameter: elapsed or used only")

        initials = sd.SuperDict(self.get_resources('initial'))
        if resource is not None:
            initials = initials.filter(resource, check=False)
        return sd.SuperDict({k: v[maint][time_type] for k, v in initials.items()})

    def get_min_assign(self):
        min_assign = dict(self.get_tasks('min_assign'))
        min_assign.update(self.get_maintenances('duration_periods'))
        return min_assign

    def get_max_assign(self):

        # max_assign = dict(M = self.get_param('maint_duration'))
        return sd.SuperDict(self.get_maintenances('duration_periods'))

    def get_max_remaining_time(self, time, maint):
        time_label = 'max_' + self.label_rt(time) + '_time'
        return self.data['maintenances'][maint][time_label]

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
        previous_states = self.get_resources("states")
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
            previous_states.to_start_finish(self.compare_tups).\
                vfilter(lambda x: x[3] == period_0)

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

    def get_fixed_maintenances(self, dict_key=None, *args, **kwargs):
        fixed_states = self.get_fixed_states(*args, **kwargs)
        fixed_maints = tl.TupList([(a, t) for (a, s, t) in fixed_states if s == 'M'])
        if dict_key is None:
            return fixed_maints
        if dict_key == 'resource':
            return fixed_maints.to_dict(result_col=1)
        if dict_key == 'period':
            return fixed_maints.to_dict(result_col=0)

    def get_fixed_tasks(self, *args, **kwargs):
        tasks = self.get_tasks()
        states = self.get_fixed_states(*args, **kwargs)
        return tl.TupList([(a, s, t) for (a, s, t) in states if s in tasks])

    def get_fixed_periods(self):
        states = self.get_fixed_states()
        return states.take([0, 2])
        # return tl.TupList([(a, t) for (a, s, t) in states]).unique()

    def get_task_period_list(self, in_dict=False):

        tasks = self.get_tasks()
        _range = self.get_periods_range
        task_periods = tasks.\
            vapply(lambda v: _range(v['start'], v['end'])).\
            vapply(tl.TupList).\
            vapply(lambda v: v.intersect(self.get_periods()))
        if in_dict:
            return task_periods
        return task_periods.to_tuplist()

    def get_first_last_period(self):
        return self.get_param('start'), self.get_param('end')

    def get_periods(self):
        return self.get_periods_range(*self.get_first_last_period())

    def get_start_end(self):
        return (self.get_param(p) for p in ['start', 'end'])

    def get_period_positions(self):
        return self.data['aux']['period_i']

    def get_periods_by_position(self):
        return self.data['aux']['period_e']

    def get_periods_range(self, start, end):
        pos_period = self.get_period_positions()
        period_pos = self.get_periods_by_position()
        return tl.TupList(period_pos[t] for t in range(pos_period[start], pos_period[end]+1))

    def get_dist_periods(self, start, end):
        pos_period = self.get_period_positions()
        return pos_period[end] - pos_period[start]

    def shift_period(self, period, num=1):
        pos_period = self.get_period_positions()
        period_pos = self.get_periods_by_position()
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
        pos_period = self.get_period_positions()
        period_pos = self.get_periods_by_position()
        if not previous:
            start = pos_period[period]
            end = start + num
        else:
            end = pos_period[period] + 1
            start = end - num
        return [period_pos[t] for t in range(start, end)]

    def get_task_period_needs(self):
        requirement = self.get_tasks('num_resource')
        return \
            self.get_task_period_list().\
            to_dict(None).\
            kapply(lambda k: requirement[k[0]])

    def get_total_period_needs(self):
        num_resource_working = {t: 0 for t in self.get_periods()}
        task_period_needs = self.get_task_period_needs()
        for (v, t), req in task_period_needs.items():
            num_resource_working[t] += req
        return sd.SuperDict.from_dict(num_resource_working)

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
        # TODO: this should depend on the maintenance (we assumme 'M')
        limit = self.get_maintenances('max_used_time')['M']
        c_needs_hours = {(k, t): v * limit * hour_perc
                         for k, v in c_num_candidates.items() for t in self.get_periods()}
        return sd.SuperDict.from_dict({'num': c_needs_num, 'hours': c_needs_hours})

    def get_task_candidates(self, recalculate=True, task=None, resource=None):

        # we check if these are old instances:
        if 'capacities' not in self.get_categories()['resources']:
            t_candidates = self.get_tasks('candidates')
            if task is not None:
                return t_candidates[task]
            return t_candidates

        # if they're the 'newer' instances:
        r_cap = self.get_resources('capacities')
        t_cap = self.get_tasks('capacities')
        if task is not None:
            t_cap = t_cap.filter(task)
        if resource is not None:
            r_cap = r_cap.filter(resource)

        t_candidates = t_cap.vapply(lambda v: [])
        for t, task_caps in t_cap.items():
            for res, res_caps in r_cap.items():
                if not len(set(task_caps) - set(res_caps)):
                    t_candidates[t].append(res)

        return t_candidates

    def get_cluster_candidates(self):
        # Since clusters are strict, their candidates are the same as the tasks.
        c_candidates = sd.SuperDict()
        t_candidates = self.get_task_candidates()
        cluster = self.get_clusters()
        for k, v in t_candidates.items():
            c_candidates[cluster[k]] = v
        return c_candidates

    def get_fixed_maintenances_cluster(self):
        fixed_per_period = self.get_fixed_maintenances(dict_key='period')
        candidates_per_cluster = self.get_cluster_candidates()
        fixed_per_period_cluster = sd.SuperDict()
        for period, resources in fixed_per_period.items():
            for cluster, candidates in candidates_per_cluster.items():
                fixed_per_period_cluster[(cluster, period)] =\
                    np.intersect1d(resources, candidates)
        return fixed_per_period_cluster

    def get_capacity_calendar(self, periods=None):
        """
        :param periods: optional filter for horizon periods
        :return: capacity for each maintenance type and period indexed by tuples.
        """
        # This should be replaced by reference to the correct maintenance data
        def_working_days = self.get_param('default_type2_capacity')
        def_capacity = self.get_param('maint_capacity')
        base_capacity = {'1': def_capacity, '2': def_working_days}
        capacity_period = self.data['maint_types'].get_property('capacity')
        base_capacity = sd.SuperDict.from_dict(base_capacity).filter(capacity_period)
        if periods is None:
            periods = self.get_periods()
        caps = {
            (t, p): capacity_period[t].get(p, base) for p in periods
            for t, base in base_capacity.items()
        }
        return sd.SuperDict.from_dict(caps)

    def get_default_consumption(self, resource, period):
        min_use = self.data['resources'][resource]['min_usage_period']
        return min_use.get(period, min_use['default'])

    def get_maint_rt(self, maint):
        maint_data = self.data['maintenances'][maint]
        return [t for t in ['ret', 'rut'] if
                  maint_data['max_' + self.label_rt(t) + '_time'] is not None]

    def get_rt_maints(self, rt):
        maint_data = self.data['maintenances']
        return [m for m, v in maint_data.items() if
                v['max_' + self.label_rt(rt) + '_time'] is not None]

    @staticmethod
    def label_rt(time):
        if time == "rut":
            return 'used'
        elif time == 'ret':
            return 'elapsed'
        else:
            raise ValueError("time needs to be rut or ret")

    def dassault_remaining_capacity(self, periods=None):
        # this function returns the net capacity without taking into account
        # the capacity that goes away with planes that go in mission.
        capacities = self.get_capacity_calendar(periods).to_dictdict()
        capacities2 = capacities.get('2', {})
        resources = self.get_resources()
        fleet_size = len(resources)
        aircraft_needs = self.get_total_period_needs()
        capacities['2'] =\
            capacities2.\
            apply(lambda k, v: (fleet_size-aircraft_needs[k])/fleet_size*v).\
            vapply(lambda v: math.ceil(v))
        return capacities.to_dictup()

    def get_types(self):
        return self.get_tasks('type_resource').values_tl().unique2()

    def get_maint_types(self):
        return self.get_maintenances('type').values_tl().unique2()

if __name__ == "__main__":
    import data.simulation as sim
    import package.params as params

    options = params.OPTIONS
    model_data = sim.create_dataset(options)
    instance = Instance(model_data)

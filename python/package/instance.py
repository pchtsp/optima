# /usr/bin/python3

import numpy as np
import package.auxiliar as aux
import package.data_input as di
import pandas as pd
import math
import package.superdict as sd
import package.tuplist as tl


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
    """

    def __init__(self, model_data):
        self.data = model_data

    def get_param(self, param=None):
        default_params = {
            'maint_weight': 1
            , 'unavail_weight': 1
            , 'min_elapsed_time': 0
            , 'min_usage_period': 0
            , 'min_avail_percent': 0.1
            , 'min_avail_value': 1
            , 'min_hours_perc': 0.2
        }
        params = {**default_params, **self.data['parameters']}
        if param is not None:
            if param not in params:
                raise ValueError("param named {} does not exist in parameters".format(param))
            return params[param]
        return params

    def get_categories(self):
        result = {}
        for category, value in self.data.items():
            # if type(value) is dict:
            elem = list(value.keys())[0]
            if type(value[elem]) is dict:
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
        default_tasks = {'min_assign': 1}
        return self.get_category('tasks', param, default_tasks)

    def get_resources(self, param=None):
        default_resources = {'states': {}}
        return self.get_category('resources', param, default_resources)

    def get_bounds(self):
        param_data = self.get_param()
        task_data = self.get_tasks()

        # maximal bounds on continuous variables:
        max_used_time = param_data['max_used_time']  # mu. in hours of usage
        maint_duration = param_data['maint_duration']
        max_elapsed_time = param_data['max_elapsed_time'] + maint_duration # me. in periods
        consumption = aux.get_property_from_dic(task_data, 'consumption')  # rh. hours per period.
        num_resources = len(self.get_resources())
        num_periods = len(self.get_periods())
        max_num_maint = num_resources*num_periods/maint_duration
        ret_total = max_elapsed_time*num_resources
        rut_total = max_used_time*num_resources

        return {
            'ret': math.floor(max_elapsed_time),
            'rut': math.floor(max_used_time),
            'used_max': math.ceil(max(consumption.values())),
            'num_maint': math.floor(max_num_maint),
            'ret_end': math.ceil(ret_total),
            'rut_end': math.ceil(rut_total)
            # 'used_min': math.ceil(min_usage)
        }

    def get_domains_sets(self):
        states = ['M']

        param_data = self.get_param()

        # periods
        first_period, last_period = param_data['start'], param_data['end']
        periods = aux.get_months(first_period, last_period)
        period_0 = aux.get_prev_month(param_data['start'])
        periods_0 = [period_0] + periods
        periods_pos = {periods[pos]: pos for pos in range(len(periods))}
        previous = {period: periods_0[periods_pos[period]] for period in periods}

        # tasks
        task_data = self.get_tasks()
        tasks = list(task_data.keys())
        start_time = self.get_tasks('start')
        end_time = self.get_tasks('end')
        min_assign = self.get_tasks('min_assign')
        candidates = self.get_task_candidates()
        # candidates = aux.get_property_from_dic(task_data, 'candidates')

        # resources
        resources_data = self.get_resources()
        resources = list(resources_data.keys())
        duration = param_data['maint_duration']
        max_elapsed = param_data['max_elapsed_time'] + duration
        min_elapsed = param_data['min_elapsed_time'] + duration
        # previous_states = \
        #     sd.SuperDict.from_dict(aux.get_property_from_dic(resources_data, 'states')).\
        #         to_dictup().to_tuplist().tup_to_start_finish()
        ret_init = self.get_initial_state("elapsed")
        ret_init_adjusted = {k: v - max_elapsed + min_elapsed for k, v in ret_init.items()}
        kt = sd.SuperDict(self.get_cluster_constraints()['num']).keys_l()

        """
        Indentation means "includes the following:".
        The elements represent a given combination resource-period.
        at0: all, including the previous period.
            at: all.                                                    => 'used'
                at_mission: a mission is assigned (fixed)               => 'assign'
                    at_mission_m: specific mission is assigned (fixed)  => 'assign'
                at_free: nothing is fixed                               => 'assign' and 'state'
                    at_free_start: can start a maintenance              => 'start_M'
                at_maint: maintenance is assigned (fixed)               => 'state' 
                    at_start: start of maintenance is assigned (fixed). => 'start_M'
        """

        at = tl.TupList((a, t) for a in resources for t in periods)
        at0 = tl.TupList([(a, period_0) for a in resources] + at)
        at_mission_m = self.get_fixed_tasks()
        at_mission = tl.TupList((a, t) for (a, s, t) in at_mission_m)  # Fixed mission assignments.
        at_start = []  # Fixed maintenances starts
        at_maint = self.get_fixed_maintenances()  # Fixed maintenance assignments.
        at_free = tl.TupList((a, t) for (a, t) in at if (a, t) not in list(at_maint + at_mission))

        # we update the possibilities of starting a maintenance
        # depending on the rule of minimal time between maintenance
        # an the initial "ret" state of the resource
        at_free_start = [(a, t) for (a, t) in at_free]
        # at_free_start = [(a, t) for (a, t) in at_free if periods_pos[t] % 3 == 0]
        at_m_ini = [(a, t) for (a, t) in at_free_start
                    if periods_pos[t] <= ret_init_adjusted[a]
                    ]
        at_m_ini_s = set(at_m_ini)

        at_free_start = tl.TupList(i for i in at_free_start if i not in at_m_ini_s)

        vt = tl.TupList((v, t) for v in tasks for t in periods if start_time[v] <= t <= end_time[v])
        avt = tl.TupList([(a, v, t) for a in resources for (v, t) in vt
               if a in candidates[v]
               if (a, t) in at_free] + \
              at_mission_m)
        ast = tl.TupList((a, s, t) for (a, t) in list(at_free + at_maint) for s in states)
        att = tl.TupList([(a, t1, t2) for (a, t1) in list(at_start + at_free_start) for t2 in periods if
               periods_pos[t1] <= periods_pos[t2] < periods_pos[t1] + duration])
        avtt = tl.TupList([(a, v, t1, t2) for (a, v, t1) in avt for t2 in periods if
                periods_pos[t1] <= periods_pos[t2] < periods_pos[t1] + min_assign[v]
                ])
        att_m = tl.TupList([(a, t1, t2) for (a, t1) in at_free_start for t2 in periods
                 if periods_pos[t1] < periods_pos[t2] < periods_pos[t1] + min_elapsed
                 ])
        att_M = tl.TupList([(a, t1, t2) for (a, t1) in at_free_start for t2 in periods
                 if periods_pos[t1] + max_elapsed < len(periods)
                 if periods_pos[t1] + min_elapsed <= periods_pos[t2] < periods_pos[t1] + max_elapsed
                 ])
        at_M_ini = tl.TupList([(a, t) for (a, t) in at_free_start
                    if ret_init[a] <= len(periods)
                    if ret_init_adjusted[a] < periods_pos[t] <= ret_init[a]
                    ])

        a_t = at.to_dict(result_col=0, is_list=True)
        a_vt = avt.to_dict(result_col=0, is_list=True)
        v_at = avt.to_dict(result_col=1, is_list=True).fill_with_default(at, [])
        at1_t2 = att.to_dict(result_col=[0, 1], is_list=True)
        t1_at2 = att.to_dict(result_col=1, is_list=True).fill_with_default(at, [])
        t2_at1 = att.to_dict(result_col=2, is_list=True)
        t2_avt1 = avtt.to_dict(result_col=3, is_list=True)
        t1_avt2 = avtt.to_dict(result_col=2, is_list=True)
        t_at_M = att_M.to_dict(result_col=2, is_list=True)
        t_a_M_ini = at_M_ini.to_dict(result_col=1, is_list=True)

        return {
         'periods'          :  periods
        ,'period_0'         :  period_0
        ,'periods_0'        :  periods_0
        ,'periods_pos'      :  periods_pos
        ,'previous'         :  previous
        ,'tasks'            :  tasks
        ,'candidates'       :  candidates
        ,'resources'        :  resources
        ,'states'           :  states
        ,'vt'               :  vt
        ,'avt'              :  avt
        ,'at'               :  at
        ,'at_maint'         :  at_maint
        ,'at_mission_m'    : at_mission_m
        ,'ast'              :  ast
        ,'at_start'         :  list(at_start + at_free_start)
        ,'at0'              :  at0
        ,'att'              :  att
        ,'a_t'              :  a_t
        ,'a_vt'             :  a_vt
        ,'v_at'             :  v_at
        ,'t1_at2'           :  t1_at2
        ,'at1_t2'           :  at1_t2
        ,'t2_at1'           :  t2_at1
        ,'at_avail'         : list(at_free + at_mission)
        ,'t2_avt1'          : t2_avt1
        ,'t1_avt2'          : t1_avt2
        ,'avtt'             : avtt
        , 'att_m'           : att_m
        , 't_at_M'          : t_at_M
        , 'at_m_ini'        : at_m_ini
        , 't_a_M_ini'       : t_a_M_ini
        , 'kt'              : kt
        }

    def get_initial_state(self, time_type, resource=None):
        """
        Returns the correct initial states for resources.
        It corrects it using the max and whether it is in maintenance.
        :param time_type: elapsed or used
        :param resource: optional value to filter only one resource
        :return:
        """
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
        res_in_maint = set([res for res, period
                            in self.get_fixed_maintenances(resource=resource)])
        rt_fixed = {a: rt_max for a in param_resources if a in res_in_maint}

        rt_init = {a: rt_max for a in param_resources}
        rt_init.update(rt_read)
        rt_init.update(rt_fixed)

        rt_init = {k: min(rt_max, v) for k, v in rt_init.items()}

        return rt_init

    def get_min_assign(self):
        min_assign = dict(self.get_tasks('min_assign'))
        min_assign['M'] = self.get_param('maint_duration')
        return min_assign

    def get_fixed_states(self, resource=None):
        """
        This function returns the fixed states in the beginning of the planning period
        They can be maintenances or mission assignments
        :param resource: if given filters only for that resource
        :return:
        """
        previous_states = sd.SuperDict.from_dict(self.get_resources("states"))
        if resource is not None:
            previous_states = previous_states.filter(resource)
        first_period = self.get_param('start')
        period_0 = aux.get_prev_month(first_period)
        min_assign = self.get_min_assign()
        # we get the states into a tuple list,
        # we turn them into a start-finish tuple
        # we filter it so we only take the start-finish periods that end before the horizon
        assignments = \
            sd.SuperDict.from_dict(previous_states). \
                to_dictup().to_tuplist().tup_to_start_finish().\
                filter_list_f(lambda x: x[3] == period_0)

        fixed_assignments_q = \
            [(a[0], a[2], min_assign.get(a[2], 0) - len(aux.get_months(a[1], a[3])))
             for a in assignments if len(aux.get_months(a[1], a[3])) < min_assign.get(a[2], 0)]

        return tl.TupList(
            [(f_assign[0], f_assign[1], aux.shift_month(first_period, t))
             for f_assign in fixed_assignments_q for t in range(f_assign[2])]
            )

    def get_fixed_maintenances(self, dict_key=None, resource=None):
        fixed_states = self.get_fixed_states(resource)
        fixed_maints = tl.TupList([(a, t) for (a, s, t) in fixed_states if s == 'M'])
        if dict_key is None:
            return fixed_maints
        if dict_key == 'resource':
            return aux.tup_to_dict(fixed_maints, result_col=1)
        if dict_key == 'period':
            return aux.tup_to_dict(fixed_maints, result_col=0)

    def get_fixed_tasks(self):
        return tl.TupList([(a, s, t) for (a, s, t) in self.get_fixed_states() if s in self.get_tasks()])

    def get_task_period_list(self, in_dict=False):

        task_periods = {task:
            np.intersect1d(
                aux.get_months(self.get_tasks('start')[task], self.get_tasks('end')[task]),
                self.get_periods()
            ) for task in self.get_tasks()
        }
        if in_dict:
            return task_periods
        return [(task, period) for task in self.get_tasks() for period in task_periods[task]]

    def get_periods(self):
        return aux.get_months(self.get_param("start"), self.get_param("end"))

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
        in_maint_dict = aux.tup_to_dict(self.get_fixed_maintenances(), 0, is_list=True)
        return {k: len(v) for k, v in in_maint_dict.items()}

    def check_enough_candidates(self):
        task_num_resources = self.get_tasks('num_resource')
        task_num_candidates = {task: len(candidates) for task, candidates in self.get_task_candidates().items()}
        task_slack = {task: task_num_candidates[task] - task_num_resources[task] for task in task_num_resources}

        return {k: (v, v / task_num_resources[k]) for k, v in task_slack.items()}

    def get_info(self):
        assign = \
            sum(v * self.data['tasks'][k]['num_resource'] for k, v in
                aux.dict_to_lendict(self.get_task_period_list(True)).items())

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
            if task1 not in group:
                group[task1] = present_group
                present_group += 1
            for task2, cap2 in capacities.items():
                if task2 not in group:
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

    def get_task_candidates(self, task=None):

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

    # def cluster_candidates(instance, options=None):
    #     l = instance.get_domains_sets()
    #     av = list(set(aux.tup_filter(l['avt'], [0, 1])))
    #     a_v = aux.tup_to_dict(av, result_col=0, is_list=True)
    #     candidate = pl.LpVariable.dicts("cand", av, 0, 1, pl.LpInteger)
    #
    #     model = pl.LpProblem("Candidates", pl.LpMinimize)
    #     for v, num in instance.get_tasks('num_resource').items():
    #         model += pl.lpSum(candidate[(a, v)] for a in a_v[v]) >= max(num + 4, num * 1.1)
    #
    #     # # objective function:
    #     # max_unavail = pl.LpVariable("max_unavail")
    #     model += pl.lpSum(candidate[tup] for tup in av)
    #
    #     # MODEL
    #
    #     # # OBJECTIVE:
    #     # model += max_unavail + max_maint * maint_weight
    #
    #     config = conf.Config(options)
    #     result = config.solve_model(model)
    #
    #     return {}

if __name__ == "__main__":
    # path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712191655/"
    # model_data = di.load_data(path + "data_in.json")
    model_data = di.get_model_data()
    instance = Instance(model_data)
    instance.get_categories()
    result = instance.get_total_fixed_maintenances()

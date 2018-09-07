# /usr/bin/python3

import numpy as np
import package.auxiliar as aux
import package.data_input as di
import pandas as pd
import math


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
        default_params = {'maint_weight': 1, 'unavail_weight': 1,
                          'min_elapsed_time': 0, 'min_usage_period': 0}
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
        return self.get_category('resources', param)

    def get_bounds(self):
        param_data = self.get_param()
        task_data = self.get_tasks()

        # maximal bounds on continuous variables:
        max_elapsed_time = param_data['max_elapsed_time']  # me. in periods
        max_used_time = param_data['max_used_time']  # mu. in hours of usage
        consumption = aux.get_property_from_dic(task_data, 'consumption')  # rh. hours per period.
        num_resources = len(self.get_resources())
        num_periods = len(self.get_periods())
        max_num_maint = num_resources*num_periods/param_data['maint_duration']
        ret_total = max_elapsed_time*num_resources
        rut_total = max_used_time*num_resources

        return {
            'ret': math.floor(max_elapsed_time),
            'rut': math.floor(max_used_time),
            'used': math.ceil(max(consumption.values())),
            'num_maint': math.floor(max_num_maint),
            'ret_end': math.ceil(ret_total),
            'rut_end': math.ceil(rut_total)
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
        # previous_states = aux.get_property_from_dic(resources_data, 'states')

        """
        Indentation means "includes the following:".
        The elements represent a given combination resource-period.
        at0: all, including the previous period.
            at: all.                                                    => 'used'
                at_mission: a mission is assigned (fixed)               => 'assign'
                at_free: nothing is fixed                               => 'assign' and 'state'
                    at_free_start: can start a maintenance              => 'start_M'
                at_maint: maintenance is assigned (fixed)               => 'state' 
                    at_start: start of maintenance is assigned (fixed). => 'start_M'
        """

        at = [(a, t) for a in resources for t in periods]
        at0 = [(a, period_0) for a in resources] + at
        at_mission = []  # to be implemented. Fixed mission assignments.
        at_start = []  # to be implemented. Fixed maintenances starts
        at_maint = self.get_fixed_maintenances()
        at_free = [(a, t) for (a, t) in at if (a, t) not in list(at_maint + at_mission)]
        at_free_start = [(a, t) for (a, t) in at_free]
        # at_free_start = [(a, t) for (a, t) in at_free if periods_pos[t] % 3 == 0]

        vt = [(v, t) for v in tasks for t in periods if start_time[v] <= t <= end_time[v]]
        avt = [(a, v, t) for a in resources for (v, t) in vt
               if a in candidates[v]
               if (a, t) in list(at_free + at_mission)]
        ast = [(a, s, t) for (a, t) in list(at_free + at_maint) for s in states]
        att = [(a, t1, t2) for (a, t1) in list(at_start + at_free_start) for t2 in periods if
               periods_pos[t1] <= periods_pos[t2] < periods_pos[t1] + duration]
        avtt = [(a, v, t1, t2) for (a, v, t1) in avt for t2 in periods if
                periods_pos[t1] <= periods_pos[t2] < periods_pos[t1] + min_assign[v]
                ]

        a_t = aux.tup_to_dict(at, result_col=0, is_list=True)
        a_vt = aux.tup_to_dict(avt, result_col=0, is_list=True)
        v_at = aux.fill_dict_with_default(aux.tup_to_dict(avt, result_col=1, is_list=True), at, [])
        at1_t2 = aux.tup_to_dict(att, result_col=[0,1], is_list=True)
        t1_at2 = aux.fill_dict_with_default(aux.tup_to_dict(att, result_col=1, is_list=True), at, [])
        t2_at1 = aux.tup_to_dict(att, result_col=2, is_list=True)
        t2_avt1 = aux.tup_to_dict(avtt, result_col=3, is_list=True)
        t1_avt2 = aux.tup_to_dict(avtt, result_col=2, is_list=True)

        return {
         'periods'          :  periods
        ,'period_0'         :  period_0
        ,'periods_0'        :  periods_0
        ,'periods_pos'      :  periods_pos
        ,'previous'         :  previous
        ,'tasks'            :  tasks
        ,'candidates'       :  candidates
        ,'resources'        :  resources
        ,'planned_maint'    :  at_maint
        ,'states'           :  states
        ,'vt'               :  vt
        ,'avt'              :  avt
        ,'at'               :  at
        ,'at_maint'         :  at_maint
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
        # TODO: fill
        , 'att_m': 1
        , 't_at_M': 1
        , 'at_m_ini': 1
        , 't_a_M_ini': 1
        }

    def get_initial_state(self, time_type):
        if time_type not in ["elapsed", "used"]:
            raise KeyError("Wrong type in time_type parameter: elapsed or used only")

        key_initial = "initial_" + time_type
        key_max = "max_" + time_type + "_time"
        param_resources = self.get_resources()
        rt_max = self.get_param(key_max)

        rt_read = aux.get_property_from_dic(param_resources, key_initial)

        # we also check if the resources is currently in maintenance.
        # If it is: we assign the rt_max (according to convention).
        res_in_maint = set([res for res, period in self.get_fixed_maintenances()])
        rt_fixed = {a: rt_max for a in param_resources if a in res_in_maint}

        rt_init = {a: rt_max for a in param_resources}
        rt_init.update(rt_read)
        rt_init.update(rt_fixed)

        rt_init = {k: min(rt_max, v) for k, v in rt_init.items()}

        return rt_init

    def get_fixed_maintenances(self, dict_key=None):
        previous_states = self.get_resources("states")
        first_period = self.get_param('start')
        duration = self.get_param('maint_duration')

        last_maint = {}
        planned_maint = []
        previous_states_n = {key: [key2 for key2 in value if value[key2] == 'M']
                             for key, value in previous_states.items()}

        # after initialization, we search for the scheduled maintenances that:
        # 1. do not continue the maintenance of the previous month
        # 2. happen in the last X months before the start of the planning period.
        for res in previous_states_n:
            _list = list(previous_states_n[res])
            _list_n = [period for period in _list if aux.get_prev_month(period) not in _list
                       if aux.shift_month(first_period, -duration) < period < first_period]
            if not len(_list_n):
                continue
            last_maint[res] = max(_list_n)
            finish_maint = aux.shift_month(last_maint[res], duration - 1)
            for period in aux.get_months(first_period, finish_maint):
                planned_maint.append((res, period))
        if dict_key is None:
            return planned_maint
        if dict_key == 'resource':
            return aux.tup_to_dict(planned_maint, result_col=1)
        if dict_key == 'period':
            return aux.tup_to_dict(planned_maint, result_col=0)

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
        for (v, t), req in self.get_task_period_needs().items():
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
        capacities = self.get_tasks('capacities')
        group = {}
        present_group = 1
        for task1, cap1 in capacities.items():
            if task1 not in group:
                group[task1] = present_group
                present_group += 1
            for task2, cap2 in capacities.items():
                if task2 not in group:
                    int_caps = np.intersect1d(cap1, cap2)
                    if len(int_caps) == len(task1):
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

    def get_cluster_minimums(self, min_percent=0.1, min_value=1):
        num_resource_maint_cluster = aux.dict_to_lendict(self.get_fixed_maintenances_cluster())
        c_needs = self.get_cluster_needs()
        c_candidates = self.get_cluster_candidates()
        c_num_candidates = aux.dict_to_lendict(c_candidates)
        c_slack = {tup: c_num_candidates[tup[0]] - c_needs[tup] - num_resource_maint_cluster.get(tup, 0)
                   for tup in c_needs}
        c_min = {(k, t): min(
            c_slack[(k, t)],
            max(
                int(c_num_candidates[k] * min_percent),
                min_value)
            ) for (k, t) in c_slack
        }

        c_needs_num = {(k, t): c_num_candidates[k] - c_needs[k, t] - c_min[k, t] - num_resource_maint_cluster[k, t]
                       for k, t in c_needs
                       }
        # TODO; fill
        c_needs_hours = {(k, t): 0 for (k, t) in range(0)}

        return {'num': c_needs_num, 'hours': c_needs_hours}

    def get_task_candidates(self, task=None):
        r_cap = self.get_resources('capacities')
        t_cap = self.get_tasks('capacities')
        if task is not None:
            t_cap = t_cap[task]
        t_cap_df = pd.DataFrame([(t, c) for t in t_cap for c in t_cap[t]], columns=['IdTask', 'CAP'])
        r_cap_df = pd.DataFrame([(t, c) for t in r_cap for c in r_cap[t]], columns=['IdResource', 'CAP'])

        # task_df = .from_dict(, orient='index')
        num_capacites = t_cap_df.groupby("IdTask"). \
            agg(len).reset_index()
        capacites_join = t_cap_df.merge(r_cap_df, on='CAP').\
            groupby(['IdTask', 'IdResource']).agg(len).reset_index()
        # capacites_join = capacites_join.reset_index(). \
        #     groupby(['IdMission', 'IdAvion']).agg(len).reset_index()
        mission_aircraft = \
            pd.merge(capacites_join, num_capacites, on=["IdTask", "CAP"]) \
                [["IdTask", "IdResource"]]

        t_candidates =\
            aux.tup_to_dict(
                mission_aircraft.to_records(index=False).tolist(),
                result_col=1
            )
        if task is not None:
            return t_candidates[task]
        return t_candidates

    def get_cluster_candidates(self):
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

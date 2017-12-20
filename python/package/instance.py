# /usr/bin/python3

import numpy as np
import package.aux as aux
import package.data_input as di


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
        default_params = {'maint_weight': 1}
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

    def get_category(self, category, param):
        if param is None:
            return self.data[category]
        if param in list(self.data[category].values())[0]:
            return aux.get_property_from_dic(self.data[category], param)
        raise IndexError("param {} is not present in the category {}".format(param, category))

    def get_tasks(self, param=None):
        return self.get_category('tasks', param)

    def get_resources(self, param=None):
        return self.get_category('resources', param)

    def get_bounds(self):
        param_data = self.get_param()
        task_data = self.get_tasks()

        # maximal bounds on continuous variables:
        max_elapsed_time = param_data['max_elapsed_time']  # me. in periods
        max_used_time = param_data['max_used_time']  # mu. in hours of usage
        consumption = aux.get_property_from_dic(task_data, 'consumption')  # rh. hours per period.

        return {
            'ret': max_elapsed_time,
            'rut': max_used_time,
            'used': max(consumption.values())
        }

    def get_domains_sets(self):
        states = ['M']
        # dtype_at = [('V', '<U6'), ('D', 'U7')]

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
        start_time = aux.get_property_from_dic(task_data, 'start')
        end_time = aux.get_property_from_dic(task_data, 'end')
        candidates = aux.get_property_from_dic(task_data, 'candidates')

        # resources
        resources_data = self.get_resources()
        resources = list(resources_data.keys())
        duration = param_data['maint_duration']
        previous_states = aux.get_property_from_dic(resources_data, 'states')

        """
        Indentation means "includes the following:".
        The elements represent a given combination resource-period.
        at0: all, including the previous period.
            at: all.                                                    => 'used'
                at_mission: a mission is assigned (fixed)               => 'assign'
                at_free: nothing is fixed                               => 'assign' and 'state'
                    at_free_start: can start a maintenance              => 'start'
                at_maint: maintenance is assigned (fixed)               => 'state' 
                    at_start: start of maintenance is assigned (fixed). => 'start'
        """

        at = [(a, t) for a in resources for t in periods]
        at0 = [(a, period_0) for a in resources] + at
        at_mission = []  # to be implemented
        at_start = []  # to be implemented
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
               periods_pos[t1] <= periods_pos[t2] <= periods_pos[t1] + duration - 1]

        a_t = aux.tup_to_dict(at, result_col=0, is_list=True)
        a_vt = aux.tup_to_dict(avt, result_col=0, is_list=True)
        v_at = aux.fill_dict_with_default(aux.tup_to_dict(avt, result_col=1, is_list=True), at, [])
        at1_t2 = aux.tup_to_dict(att, result_col=[0,1], is_list=True)
        t1_at2 = aux.fill_dict_with_default(aux.tup_to_dict(att, result_col=1, is_list=True), at, [])
        t2_at1 = aux.fill_dict_with_default(aux.tup_to_dict(att, result_col=2, is_list=True), at, [])

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
        }

    def get_initial_state(self, time_type):
        if time_type not in ["elapsed", "used"]:
            raise KeyError("Wrong type in time_type parameter: elapsed or used only")

        key_initial = "initial_" + time_type
        key_max = "max_" + time_type + "_time"
        param_resources = self.get_resources()
        rt_max = self.get_param()[key_max]

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

    def get_fixed_maintenances(self):
        previous_states = aux.get_property_from_dic(self.get_resources(), "states")
        first_period = self.get_param()['start']
        duration = self.get_param()['maint_duration']

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
        return planned_maint

    def get_task_period_list(self):
        parameters_data = self.get_param()
        task_data = self.get_tasks()
        periods = aux.get_months(parameters_data['start'], parameters_data['end'])

        task_start = aux.get_property_from_dic(task_data, 'start')
        task_end = aux.get_property_from_dic(task_data, 'end')

        task_periods = {task:
            np.intersect1d(
                aux.get_months(task_start[task], task_end[task]),
                periods
            ) for task in task_data
        }
        return [(task, period) for task in task_data for period in task_periods[task]]

    def get_periods(self):
        return aux.get_months(self.get_param("start"), self.get_param("end"))

    def get_total_period_needs(self):
        requirement = self.get_tasks('num_resource')
        num_resource_working = {t: 0 for t in self.get_periods()}
        for (v, t) in self.get_task_period_list():
            num_resource_working[t] += requirement[v]
        return num_resource_working

    def get_total_fixed_maintenances(self):
        in_maint_dict = aux.tup_to_dict(self.get_fixed_maintenances(), 0, is_list=True)
        return {k: len(v) for k, v in in_maint_dict.items()}

    def check_enough_candidates(self):
        task_num_resources = self.get_tasks('num_resource')
        task_num_candidates = {task: len(candidates) for task, candidates in self.get_tasks('candidates').items()}
        task_slack = {task: task_num_candidates[task] - task_num_resources[task] for task in task_num_resources}

        return {k: (v, v / task_num_resources[k]) for k, v in task_slack.items()}


if __name__ == "__main__":
    path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712191655/"
    model_data = di.load_data(path + "data_in.json")
    # model_data = di.get_model_data()
    instance = Instance(model_data)
    instance.get_categories()
    result = instance.get_total_fixed_maintenances()

    candidates = instance.get_tasks('candidates')
    {k: len(v) for k, v in candidates.items()}
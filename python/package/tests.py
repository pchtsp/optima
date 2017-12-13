# /usr/bin/python3
import package.aux as aux
import numpy as np
import pandas as pd
import package.data_input as di
import package.solution as sol
import package.instance as inst


"""

## tasks

[*] are covered: number of resources
[*] covered by correct resources and no other

## resources

[ ] balance is well done
[*] consumption
[*] at most one task in each period => 
    this I cannot check because of the structure of the data.
    But it could be detected if tasks' needs are not taken into account.
[*] maintenance scheduled correctly: duration.
[*] no task+maintenance at the same time.
"""


class CheckModel(object):

    def __init__(self, instance, solution):
        self.instance = instance
        self.solution = solution

    def check_solution(self):
        func_list = {
            'duration':     self.check_maintenance_duration
            ,'candidates':  self.check_resource_in_candidates
            ,'state':       self.check_resource_state
            ,'resources':   self.check_task_num_resources
            ,'usage':       self.check_resource_consumption
        }
        return {k: v() for k, v in func_list.items()}

    def check_task_num_resources(self):

        task_periods_list = self.instance.get_task_period_list()
        task_reqs = aux.get_property_from_dic(self.instance.get_tasks(), 'num_resource')

        task_assigned = {key: 0 for key in task_periods_list}
        task_assigned.update(self.solution.get_task_num_resources())
        task_under_assigned = {
            (task, period): task_reqs[task] - task_assigned[(task, period)]
            for (task, period) in task_periods_list
            if task_reqs[task] - task_assigned[(task, period)] > 0
        }

        return task_under_assigned

    def check_resource_in_candidates(self):
        task_data = self.instance.get_tasks()
        task_solution = self.solution.get_tasks()

        task_candidates = aux.get_property_from_dic(task_data, 'candidates')

        bad_assignment = {
            (resource, period): task
            for (resource, period), task in task_solution.items()
            if resource not in task_candidates[task]
        }
        return bad_assignment

    def get_consumption(self):
        maint_hours = self.instance.get_param('max_used_time')
        hours = self.instance.get_tasks("consumption")
        demand = {k: hours[v] for k, v in self.solution.get_tasks().items()}
        supply = {k: maint_hours for k in self.solution.get_maintenance_starts()}
        netgain = {(resource, period): 0
                   for resource in self.instance.get_resources()
                   for period in self.instance.get_periods()}
        for k, v in demand.items():
            netgain[k] -= v
        for k, v in supply.items():
            netgain[k] += v
        return netgain

    def get_remaining_usage_time(self):
        prev_month = aux.get_prev_month(self.instance.get_param('start'))
        table_initial = \
            pd.DataFrame(
                aux.dict_to_tup(
                    self.instance.get_resources('initial_used')
                ),
                columns=['resource', 'netgain']).assign(period=prev_month)
        netgain = self.get_consumption()
        table = pd.DataFrame(aux.dict_to_tup(netgain), columns=['resource', 'period', 'netgain'])
        table = pd.concat([table_initial, table])
        table.sort_values(['resource', 'period'], inplace=True)
        table['netgain_c'] = \
            table.groupby('resource').netgain.cumsum()
        return table.set_index(['resource', 'period'])['netgain_c'].to_dict()

    def check_resource_consumption(self):
        rut = self.get_remaining_usage_time()
        return {k: v for k, v in rut.items() if v < 0}

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


if __name__ == "__main__":
    path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712121208/"
    model_data = di.load_data(path + "data_in.json")
    solution = di.load_data(path + "data_out.json")
    # result = aux.dicttup_to_dictdict(solution['task'])

    # aux.get_months('2017-08', '2018-03')

    check = CheckModel(inst.Instance(model_data), sol.Solution(solution))
    # sol.Solution(solution).get_schedule()
    # check.check_resource_consumption()
    results = check.check_solution()
    # check.check_resource_state()
    # pp.pprint(results)
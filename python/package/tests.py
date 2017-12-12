# /usr/bin/python3
import package.aux as aux
import numpy as np
import package.data_input as di
import package.solution as sol
import package.instance as inst
# import pandas as pd

"""

## tasks

[*] are covered: number of resources
[*] covered by correct resources and no other

## resources

[ ] balance is well done
[ ] consumption
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
            'duration': self.check_maintenance_duration
            ,'consumption': self.check_resource_consumption
            ,'candidates': self.check_resource_in_candidates
            ,'state': self.check_resource_state
            ,'resources': self.check_task_num_resources
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

    def get_conumption(self):
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

    def check_resource_consumption(self):
        # TODO: make balance correct. Still incomplete.
        netgain = self.get_conumption()



        return {}

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
        # TODO: improve this using methods.
        parameters = self.instance.get_param()
        state_solution = self.solution.get_state()
        in_maintenance = [key for key, value in state_solution.items() if value == 'M']
        first_period = parameters['start']
        last_period = parameters['end']
        start_finish = aux.tup_to_start_finish(in_maintenance)

        maintenance_duration_incorrect = {}
        for (resource, start, finish) in start_finish:
            if start == first_period or finish == last_period:
                continue
            size_period = len(aux.get_months(start, finish))
            if size_period < parameters['maint_duration']:
                maintenance_duration_incorrect[(resource, start)] = size_period

        return maintenance_duration_incorrect


if __name__ == "__main__":
    path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712121208/"
    model_data = di.load_data(path + "data_in.json")
    solution = di.load_data(path + "data_out.json")
    # result = aux.dicttup_to_dictdict(solution['task'])

    # aux.get_months('2017-08', '2018-03')

    check = CheckModel(inst.Instance(model_data), sol.Solution(solution))
    check.get_conumption()
    # results = check.check_solution()
    # check.check_resource_state()
    # pp.pprint(results)
# /usr/bin/python3
import package.aux as aux
import numpy as np
# import pandas as pd

"""

## tasks

[*] are covered: number of resources
[*] covered by correct resources and no other

## resources

* balance is well done
[*] at most one task in each period => 
    this I cannot check because of the structure of the data.
    But it could be detected if tasks' needs are not taken into account.
[*] maintenance scheduled correctly: duration.
[*] no task+maintenance at the same time.

"""


class CheckModel(object):

    def __init__(self, model_data, solution):
        self.model_data = model_data
        self.solution = solution

    def check_all(self):
        return

    def check_task_num_resources(self):
        parameters_data = self.model_data['parameters']
        task_data = self.model_data['tasks']
        task_solution = self.solution['task']
        periods = aux.get_months(parameters_data['start'], parameters_data['end'])

        task_reqs = aux.get_property_from_dic(task_data, 'num_resource')
        task_start = aux.get_property_from_dic(task_data, 'start')
        task_end = aux.get_property_from_dic(task_data, 'end')

        # task_start = {task: max(task_start[task], initial_period) for task in task_start}
        # task_end = {task: min(task_end[task], last_period) for task in task_end}

        task_periods = {task:
            np.intersect1d(
                aux.get_months(task_start[task], task_end[task]),
                periods
            ) for task in task_reqs
        }
        task_periods_list = [(task, period) for task in task_reqs for period in task_periods[task]]
        task_resources = aux.tup_to_dict(aux.dict_to_tup(task_solution), 0, is_list=True)

        # TODO: here the domain of task_assigned may be smaller. This wouls return an error.

        task_assigned = {key: len(value) for key, value in task_resources.items()}
        task_over_assigned = {
            (task, period): task_assigned[(task, period)] - task_reqs[task]
            for (task, period) in task_periods_list
            if task_assigned[(task, period)] - task_reqs[task] < 0
        }

        return task_over_assigned

    def check_resource_in_candidates(self):
        task_data = self.model_data['tasks']
        task_solution = self.solution['task']

        task_candidates = aux.get_property_from_dic(task_data, 'candidates')

        bad_assignment = {
            (resource, period): task
            for (resource, period), task in task_solution.items()
            if resource not in task_candidates[task]
        }
        return bad_assignment

    def check_resource_consumption(self):
        task_data = self.model_data['tasks']
        task_solution = self.solution['task']

        return

    def check_resource_state(self):
        task_solution = self.solution['task']
        state_solution = self.solution['state']

        duplicated_states = \
            np.intersect1d(task_solution.keys(), state_solution.keys())

        return duplicated_states

    def check_maintenance_duration(self):
        parameters = self.model_data['parameters']
        state_solution = self.solution['state']
        in_maintenance = [key for key, value in state_solution.items() if value == 'M']
        first_period = parameters['start']
        last_period = parameters['end']
        start_finish = aux.tup_tp_start_finish(in_maintenance)

        maintenance_duration_incorrect = {}
        for (resource, start, finish) in start_finish:
            if start == first_period or finish == last_period:
                continue
            size_period = len(aux.get_months(start, finish))
            if size_period < parameters['maint_duration']:
                maintenance_duration_incorrect[(resource, start)] = size_period

        return maintenance_duration_incorrect




if __name__ == "__main__":
    path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712051356/"
    model_data = aux.load_data(path + "data_in.pickle")
    solution = aux.load_data(path + "data_out.pickle")
    aux.get_months('2017-08', '2018-03')
    check = CheckModel(model_data, solution)
    check.check_maintenance_duration()
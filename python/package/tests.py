# /usr/bin/python3
import package.aux as aux
import numpy as np
import package.data_input as di
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
        func_list = {
            'duration': self.check_maintenance_duration
            ,'consumption': self.check_resource_consumption
            ,'candidates': self.check_resource_in_candidates
            ,'state': self.check_resource_state
            ,'resources': self.check_task_num_resources
        }
        return {k: v() for k, v in func_list.items()}

    def check_task_num_resources(self):
        parameters_data = self.model_data['parameters']
        task_data = self.model_data['tasks']
        task_solution = self.solution['task']
        periods = aux.get_months(parameters_data['start'], parameters_data['end'])

        task_reqs = aux.get_property_from_dic(task_data, 'num_resource')
        task_start = aux.get_property_from_dic(task_data, 'start')
        task_end = aux.get_property_from_dic(task_data, 'end')

        task_periods = {task:
            np.intersect1d(
                aux.get_months(task_start[task], task_end[task]),
                periods
            ) for task in task_reqs
        }
        task_periods_list = [(task, period) for task in task_reqs for period in task_periods[task]]
        task_resources = aux.tup_to_dict(aux.dict_to_tup(task_solution), 0, is_list=True)
        task_resources = {(a, t): v for (t, a), v in task_resources.items()}

        task_assigned = {key: 0 for key in task_periods_list}
        task_assigned.update({key: len(value) for key, value in task_resources.items()})
        task_under_assigned = {
            (task, period): task_reqs[task] - task_assigned[(task, period)]
            for (task, period) in task_periods_list
            if task_reqs[task] - task_assigned[(task, period)] > 0
        }

        return task_under_assigned

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

        return {}

    def check_resource_state(self):
        task_solution = self.solution['task']
        state_solution = self.solution['state']

        task_solution_k = np.fromiter(task_solution.keys(),
                                      dtype=[('A', '<U6'), ('T', 'U7')])
        state_solution_k = np.fromiter(state_solution.keys(),
                                      dtype=[('A', '<U6'), ('T', 'U7')])
        duplicated_states = \
            np.intersect1d(task_solution_k, state_solution_k)

        return [tuple(item) for item in duplicated_states]

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
    path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712111530/"
    model_data = di.load_data(path + "data_in.pickle", "pickle")
    solution = di.load_data(path + "data_out.pickle", "pickle")
    # aux.get_months('2017-08', '2018-03')
    check = CheckModel(model_data, solution)
    results = check.check_all()
    # check.check_resource_state()
    # pp.pprint(results)
# /usr/bin/python3

import package.aux as aux


class Solution(object):

    def __init__(self, solution):
        self.data = solution

    def get_tasks(self):
        return aux.dictdict_to_dictup(self.data['task'])

    def get_state(self):
        return aux.dictdict_to_dictup(self.data['state'])

    def get_maintenance_periods(self):
        in_maintenance = [k for k, v in self.get_state().items() if v == 'M']
        return aux.tup_to_start_finish(in_maintenance)

    def get_maintenance_starts(self):
        maintenances = self.get_maintenance_periods()
        return [(r, s) for (r, s, e) in maintenances]

    def get_task_resources(self):
        task_solution = self.get_tasks()
        task_resources = aux.tup_to_dict(aux.dict_to_tup(task_solution), 0, is_list=True)
        return {(a, t): v for (t, a), v in task_resources.items()}

    def get_task_num_resources(self):
        return {key: len(value) for key, value in self.get_task_resources().items()}
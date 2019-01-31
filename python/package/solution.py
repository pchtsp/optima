# /usr/bin/python3

import package.auxiliar as aux
import pandas as pd
import numpy as np
import package.superdict as sd
# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
# from rpy2.robjects import pandas2ri
# import plotly.figure_factory as ff
# import plotly


class Solution(object):
    """
    These objects represent the solution to the assignment problem
    It does not include the initial data.
    The methods are made to make it easy to get information.
    They also draw graphs.
    data format:
    {
    'state': {(r, p): state}, state = 'M',
    'task': {(r, p): task}, task = 'O10', 'O5'
    }

    """

    def __init__(self, solution):
        self.data = solution

    def get_category(self, category, param):
        if param is None:
            return self.data[category]
        if param in list(self.data[category].values())[0]:
            return aux.get_property_from_dic(self.data[category], param)
        raise IndexError("param {} is not present in the category {}".format(param, category))

    def get_periods(self):
        resource_period = list(self.get_tasks().keys())
        return sorted(aux.tup_to_dict(resource_period, result_col=0).keys())

    def get_tasks(self):
        return aux.dictdict_to_dictup(self.data['task'])

    def get_state(self, resource=None):
        data = sd.SuperDict.from_dict(self.data['state'])
        if resource is not None:
            data = data.filter(resource, check=False)
        return data.to_dictup()

    def get_maintenance_periods(self, resource=None):
        result = aux.tup_to_start_finish(
            aux.dict_to_tup(self.get_state(resource))
        )
        return [(k[0], k[1], k[3]) for k in result if k[2] == 'M']

    def get_task_periods(self):
        result = aux.tup_to_start_finish(
            aux.dict_to_tup(self.get_tasks())
        )
        return result

    def get_maintenance_starts(self):
        maintenances = self.get_maintenance_periods()
        return [(r, s) for (r, s, e) in maintenances]

    def get_task_resources(self):
        task_solution = self.get_tasks()
        task_resources = aux.tup_to_dict(aux.dict_to_tup(task_solution), 0, is_list=True)
        return {(a, t): v for (t, a), v in task_resources.items()}

    def get_task_num_resources(self):
        task_resources = self.get_task_resources()
        return {key: len(value) for key, value in task_resources.items()}

    def get_schedule(self):
        """
        returns periods of time where a resources has a specific state
        In a way, collapses single period assignments into a start-finish
        :return: a (resource, start, finish, task) tuple
        """
        statesMissions = aux.dict_to_tup(self.get_state()) + aux.dict_to_tup(self.get_tasks())
        result = aux.tup_to_start_finish(statesMissions)
        return result

    def get_unavailable(self):
        num_tasks = self.get_in_task()
        in_maint = self.get_in_maintenance()
        return {k: in_maint[k] + num_tasks[k] for k in in_maint}  # in_maint has all periods already

    def get_in_task(self):
        tasks = [(t, r) for (r, t) in self.get_tasks()]
        in_mission = {k: len(v) for k, v in aux.tup_to_dict(tasks, 1, is_list=True).items()}
        return aux.fill_dict_with_default(in_mission, self.get_periods())

    def get_in_maintenance(self):
        states = [(t, r) for (r, t), v in self.get_state().items() if v == 'M']
        in_maint = {k: len(v) for k, v in aux.tup_to_dict(states, 1, is_list=True).items()}
        # fixed maintenances should be included in the states already
        return aux.fill_dict_with_default(in_maint, self.get_periods())

    def get_number_maintenances(self, resource):
        return sum(v == 'M' for v in self.data['state'].get(resource, {}).values())

    # def graph_maintenances(self, path, **kwags):
    #     """
    #     uses bokeh
    #     :param path: path to generate image
    #     :return: graph object
    #     """
    #     return aux.graph_dict_time(self.get_in_maintenance(), path, **kwags)
    #
    # def graph_unavailable(self, path, **kwags):
    #     return aux.graph_dict_time(self.get_unavailable(), path, **kwags)


if __name__ == "__main__":
    path_states = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712190002/"
    path_nostates = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712181704/"
    import package.data_input as di
    sol_states = Solution(di.load_data(path_states + "data_out.json"))
    sol_nostates = Solution(di.load_data(path_nostates + "data_out.json"))
    # sol_nostates.graph_maintenances(path="/home/pchtsp/Documents/projects/OPTIMA/img/maintenances.html",
    #                                 title="Maintenances")
    # sol_nostates.graph_unavailable(path="/home/pchtsp/Documents/projects/OPTIMA/img/unavailable.html",
    #                                title="Affectations")
    # sol_nostates.print_solution("/home/pchtsp/Documents/projects/OPTIMA/img/calendar.html")

    # sol.print_solution("/home/pchtsp/Downloads/calendar_temp3.html")
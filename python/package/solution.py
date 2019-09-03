# /usr/bin/python3

import package.auxiliar as aux
import pytups.superdict as sd
import pytups.tuplist as tl

class Solution(object):
    """
    These objects represent the solution to the assignment problem
    It does not include the initial data.
    The methods are made to make it easy to get information.
    They also draw graphs.
    data format:
    {
    'state': {r: {p: s}}, s = 'M', 'VI'
    'task': {r: {p: task}}, task = 'O10', 'O5'
    'state_m': {r: {p: {s: 1}}}, s = 'M', 'VI'
    }

    """

    def __init__(self, solution):
        data_default = {'state_m': {}, 'task': {}, 'aux': {'ret': {}, 'rut': {}, 'start': {}}}
        self.data = sd.SuperDict.from_dict(data_default)
        self.data.update(solution)
        # self.data['aux'] = sd.SuperDict()

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
        return sd.SuperDict.from_dict(self.data['task']).to_dictup()

    def get_state_tuplist(self, resource=None):
        states = self.get_state(resource)
        return tl.TupList(states.keys())

    def get_state(self, resource=None):
        data = self.data['state_m']
        if resource is not None:
            data = data.filter(resource, check=False)
        return data.to_dictup()

    def get_task_resources(self):
        task_solution = self.get_tasks()
        task_resources = aux.tup_to_dict(aux.dict_to_tup(task_solution), 0, is_list=True)
        return {(a, t): v for (t, a), v in task_resources.items()}

    def get_task_num_resources(self):
        task_resources = self.get_task_resources()
        return {key: len(value) for key, value in task_resources.items()}

    def get_state_tasks(self):
        statesMissions = self.get_state_tuplist() + self.get_tasks().to_tuplist()
        return tl.TupList(statesMissions)

    def get_schedule(self, compare_tups):
        """
        returns periods of time where a resources has a specific state
        In a way, collapses single period assignments into a start-finish
        :return: a (resource, start, finish, task) tuple
        """
        statesMissions = self.get_state_tasks()
        result = statesMissions.tup_to_start_finish(ct=compare_tups)
        return result

    def get_unavailable(self):
        num_tasks = self.get_in_task()
        in_maint = self.get_in_some_maintenance()
        return {k: in_maint[k] + num_tasks[k] for k in in_maint}  # in_maint has all periods already

    def get_in_task(self):
        tasks = [(t, r) for (r, t) in self.get_tasks()]
        in_mission = {k: len(v) for k, v in aux.tup_to_dict(tasks, 1, is_list=True).items()}
        return aux.fill_dict_with_default(in_mission, self.get_periods())

    def get_in_some_maintenance(self):
        raise ValueError("This is no longer supported")
        # _states = self.get_state_tuplist()
        # states = [(t, r) for r, t, v in _states if maint in v]
        # in_maint = {k: len(v) for k, v in aux.tup_to_dict(states, 1, is_list=True).items()}
        # # fixed maintenances should be included in the states already
        # return aux.fill_dict_with_default(in_maint, self.get_periods())

    def get_period_state(self, resource, period, cat):
        try:
            return self.data[cat][resource][period]
        except KeyError:
            return None

    def is_resource_free(self, resource, period):
        if self.get_period_state(resource, period, 'task') is not None:
            return False
        states = self.get_period_state(resource, period, 'state_m')
        if states is not None and 'M' in states:
            return False
        return True

if __name__ == "__main__":
    path_states = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712190002/"
    path_nostates = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712181704/"
    import data.data_input as di
    sol_states = Solution(di.load_data(path_states + "data_out.json"))
    sol_nostates = Solution(di.load_data(path_nostates + "data_out.json"))
    # sol_nostates.graph_maintenances(path="/home/pchtsp/Documents/projects/OPTIMA/img/maintenances.html",
    #                                 title="Maintenances")
    # sol_nostates.graph_unavailable(path="/home/pchtsp/Documents/projects/OPTIMA/img/unavailable.html",
    #                                title="Affectations")
    # sol_nostates.print_solution("/home/pchtsp/Documents/projects/OPTIMA/img/calendar.html")

    # sol.print_solution("/home/pchtsp/Downloads/calendar_temp3.html")
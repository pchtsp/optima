# /usr/bin/python3

import pytups.superdict as sd
import pytups.tuplist as tl
import numpy as np

class Solution(object):
    """
    These objects represent the solution to the assignment problem
    It does not include the initial data.
    The methods are made to make it easy to get information.
    They also draw graphs.
    data format:

    >>> {'task': {'RESOURCE_1': {'PERIOD_1': 'TASK_1'}}, 'state_m': {'RESOURCE_1': {'PERIOD_2': {'MAINT_1': 1}}}}

    """

    def __init__(self, solution_data):
        data_default = {'state_m': {}, 'task': {}, 'aux': {'ret': {}, 'rut': {}, 'start': {}}}
        self.data = sd.SuperDict.from_dict(data_default)
        self.data.update(sd.SuperDict.from_dict(solution_data))
        self.migrate_to_multimaint()

    def migrate_to_multimaint(self):
        states = self.data.get('state')
        if not states:
            return
        # here, we have an old solution format
        # we just convert the maint into a dict of maints
        self.data['state_m'] = \
            states.to_dictup().\
            vapply(lambda v: sd.SuperDict({v: 1})).\
            to_dictdict()
        self.data.pop('state')
        return

    def get_category(self, category, param):
        if param is None:
            return self.data[category]
        if param in list(self.data[category].values())[0]:
            return sd.SuperDict.from_dict(self.data[category]).get_property(param)
        raise IndexError("param {} is not present in the category {}".format(param, category))

    def get_periods(self):
        resource_period = self.get_tasks().keys_tl().to_dict(result_col=0).keys_l()
        return sorted(resource_period)

    def get_tasks(self):
        return self.data['task'].to_dictup()

    def get_state_tuplist(self, resource=None):
        states = self.get_state(resource)
        return tl.TupList(states.keys())

    def get_state(self, resource=None):
        data = self.data['state_m']
        if resource is not None:
            data = data.filter(resource, check=False)
        return data.to_dictup()

    def get_task_resources(self, periods=None):
        tasks = self.get_tasks()
        if periods:
            periods = set(periods)
            tasks = tasks.kfilter(lambda k: k[1] in periods)
        return tasks.to_tuplist().to_dict(result_col=0, indices=[2, 1], is_list=True)

    def get_task_num_resources(self, periods=None):
        tasks = self.get_tasks()
        if periods:
            periods = set(periods)
            tasks = tasks.kfilter(lambda k: k[1] in periods)
        if not len(tasks):
            return sd.SuperDict()
        resource, period = zip(*tasks.keys())
        task = tasks.values_l()
        keys, values = np.unique(np.array(list(zip(task, period))), axis=0, return_counts=True)
        result = sd.SuperDict({tuple(k): v for k, v in zip(keys, values)})
        return result
        # result.to_tuplist().to_set() ^ result2.to_tuplist().to_set()
        # import unittest
        # unittest.TestCase().assertDictEqual(result, result2)
        # result2 = self.get_task_resources().vapply(len)
        # assert result==result2

    def get_task_num_resources_old(self, periods=None):
        result2 = self.get_task_resources(periods).vapply(len)
        return result2

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
        result = statesMissions.to_start_finish(compare_tups=compare_tups)
        return result

    def get_unavailable(self):
        num_tasks = self.get_in_task()
        in_maint = self.get_in_some_maintenance()
        return {k: in_maint[k] + num_tasks[k] for k in in_maint}  # in_maint has all periods already

    def get_in_task(self):
        tasks = [(t, r) for (r, t) in self.get_tasks()]
        return tl.TupList(tasks).\
            to_dict(1, is_list=True).\
            to_lendict().\
            fill_with_default(self.get_periods())

    def get_in_some_maintenance(self):
        raise ValueError("This is no longer supported")

    def get_period_state(self, resource, period, cat):
        try:
            return self.data[cat][resource][period]
        except KeyError:
            return None

    def get_period_state_category(self, resource, period):
        task = self.get_period_state(resource, period, 'task')
        if task is not None:
            return task, 'task'
        states = self.get_period_state(resource, period, 'state_m')
        if states is not None:
            return states, 'state_m'
        return None, None

    def is_resource_free(self, resource, period):
        if self.get_period_state(resource, period, 'task') is not None:
            return False
        states = self.get_period_state(resource, period, 'state_m')
        if states is not None and 'M' in states:
            return False
        return True

if __name__ == "__main__":
    import package.params as params

    path_states = params.PATHS['experiments'] + "experiments/201712190002/"
    path_nostates = params.PATHS['experiments'] + "experiments/201712181704/"
    import data.data_input as di
    sol_states = Solution(di.load_data(path_states + "data_out.json"))
    sol_nostates = Solution(di.load_data(path_nostates + "data_out.json"))

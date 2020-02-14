import pytups.superdict as sd
import pytups.tuplist as tl
import os
import package.solution as sol
import data.data_input as di
import numpy as np
import solvers.heuristics as heur
import patterns.create_patterns as cp
import random as rn
import math
import time
import logging as log


class GraphOriented(heur.GreedyByMission):

    # statuses for detecting a new worse solution
    status_worse = {2, 3}
    # statuses for detecting if we accept a solution
    status_accept = {0, 1, 2}

    def __init__(self, instance, solution=None):

        super().__init__(instance, solution)
        resources = self.instance.get_resources()

        self.instance.data['aux']['graphs'] = sd.SuperDict()
        for r in resources:
            graph_data = cp.get_graph_of_resource(instance=instance, resource=r)
            self.instance.data['aux']['graphs'][r] = graph_data


    def solve(self, options):
        """
         Solves an instance using the metaheuristic.

         :param options: dictionary with options
         :return: solution to the planning problem
         :rtype: :py:class:`package.solution.Solution`
         """
        pass

    def date_to_node(self, resource, date, previous=True):
        # previous implies to look for the latest previous assignment.
        # if false, we look for the soonest next assignment
        period, category = self.get_next_assignment(resource, date, search_future=not previous)
        _rts = sd.SuperDict(rut=None, ret=None)
        if period is None:
            # there are no other assignments until reaching the end of the horizon
            if previous:
                return self.get_source_node(resource)
            return self.get_sink_node(resource)

        # Now we know there is something in this period and what type.
        # But we do not know if it started in this period
        period_start = self.get_start_of_assignment(resource, period, category)
        period_end = self.get_start_of_assignment(resource, period, category, get_last=True)
        assignment = self.solution.get_period_state(resource, period, cat=category)
        _type = 0
        if category == 'task':
            _type = 1

        # we return the state of type: period, period_end, assignment, type, rut, ret
        state = sd.SuperDict(period=period_start, period_end=period_end,
                    assignment=assignment, type=_type, **_rts)
        return cp.state_to_node(self.instance, resource=resource, state=state)

    def get_graph_data(self, resource):
        return self.instance.data['aux']['graphs'][resource]

    def get_source_node(self, resource):
        return self.instance.data['aux']['graphs'][resource]['source']

    def get_sink_node(self, resource):
        return self.instance.data['aux']['graphs'][resource]['sink']

    def window_to_patterns(self, resource, date1, date2):
        """
        This method is accessed by the repairing method.
        To know the options of assignments

        :param resource:
        :param date1:
        :param date2:
        :return:
        """
        node1 = self.date_to_node(resource, date1, previous=True)
        node2 = self.date_to_node(resource, date2, previous=False)

        return cp.nodes_to_patterns(node1=node1, node2=node2, **self.get_graph_data(resource))

    def select_pattern(self, patterns):
        # TODO: here, we need to filter feasible patterns with the next maintenance cycle
        # TODO: also, maybe calculate the impact on coupling constraints, OF, etc.
        return patterns[0]

    def apply_pattern(self, pattern):
        # TODO: warning, here I'm ignoring the two extremes
        # TODO: there is a problem with filtering the tuplist
        for node in pattern[1:-1]:
            if node.type == 1:
                self.apply_task(node)
            elif node.type == 0:
                self.apply_maint(node)


    def apply_maint(self, node):
        for period in self.instance.get_periods_range(node.period, node.period_end):
            self.solution.data.set_m('state_m', node.resource, period, node.assignment, value=1)

    def apply_task(self, node):
        for period in self.instance.get_periods_range(node.period, node.period_end):
            self.solution.data.set_m('task', node.resource, period, value=node.assignment)

    def erase_window(self, resource, start, end, key='task'):
        # here, we will be take the risk of just deleting everything.
        data = self.solution.data
        fixed_periods = self.instance.get_fixed_periods().to_set()

        delete_tasks = 0
        periods = self.instance.get_periods_range(start, end)
        if resource not in data[key]:
            return delete_tasks
        for period in periods:
            if (resource, period) in fixed_periods:
                continue
            data[key][resource].pop(period, None)
            delete_tasks = 1
        return delete_tasks


if __name__ == '__main__':
    import package.params as pm
    import package.instance as inst
    import data.test_data as test_d

    data_in = test_d.dataset3()
    instance = inst.Instance(data_in)
    self = GraphOriented(instance)
    resource = '1'
    self.date_to_node(resource, '2018-12')
    patterns = self.window_to_patterns('1', '2018-01', '2018-12')
    pattern = self.select_pattern(patterns)
    _next = self.instance.get_next_period
    _prev = self.instance.get_prev_period
    start = _next(pattern[0].period_end)
    end = _prev(pattern[-1].period)
    update_tasks = self.erase_window(resource, start, end, 'task')
    update_maints = self.erase_window(resource, start, end, 'state_m')
    self.apply_pattern(pattern)
    # Update!
    times = ['rut']
    if True:
        # TODO: I need to check if I add / delete a maintenance
        times.append('ret')
    for t in times:
        for m in self.instance.get_rt_maints(t):
            self.update_rt_until_next_maint(resource, start, m, t)
    self.solution.data

    # solution = self.solve(pm.OPTIONS)

    pass
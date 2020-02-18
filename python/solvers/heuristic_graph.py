import pytups.superdict as sd
import pytups.tuplist as tl
import os
import package.solution as sol
import data.data_input as di
import numpy as np
import solvers.heuristics as heur
import patterns.create_patterns as cp
import random as rn
import pulp as pl
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

        if previous:
            # TODO: no, we want to use ret and rut for the source.
            _rts = sd.SuperDict(rut=None, ret=None)
        else:
            _rts = sd.SuperDict(rut=None, ret=None)
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

    def filter_patterns(self, resource, date1, date2, patterns):
        # TODO: here, we need to filter feasible patterns with the next maintenance cycle
        # TODO: also, maybe calculate the impact on coupling constraints, OF, etc.
        # we are using 1 and -2 because of the tails we are not using.
        # also, here. Filter assignments previous to dates, potentially
        return \
            patterns.\
                vapply(lambda v: v).\
                vfilter(lambda v: v and
                                  v[1].period >= date1 and
                                  v[-2].period_end <= date2)

    def apply_pattern(self, pattern):
        resource = pattern[0].resource
        start = _next(pattern[0].period_end)
        end = _prev(pattern[-1].period)
        _shift = self.instance.shift_period
        deleted_tasks = self.erase_window(resource, start, end, 'task')
        deleted_maints = self.erase_window(resource, start, end, 'state_m')

        # We apply all the assignments
        # Warning, here I'm ignoring the two extremes[1:-1]
        added_maints = 0
        added_tasks = 0
        for node in pattern[1:-1]:
            if node.type == 1:
                added_tasks += self.apply_task(node)
            elif node.type == 0:
                added_maints += self.apply_maint(node)

        # Update!
        times = ['rut']
        if added_maints or deleted_maints:
            times.append('ret')
        maints = self.instance.get_maintenances()
        for t in times:
            for m in self.instance.get_rt_maints(t):
                # Instead of "start" we can choose the first thing that passes
                _start = start
                while _start <= end:
                    # iterate over maintenance periods
                    updated_periods = self.update_rt_until_next_maint(resource, _start, m, t)
                    # TODO: if the maintenance falls inside the period: update the maintenance values.
                    _start = _shift(updated_periods[-1], maints[m]['duration_periods'])
        return True


    def apply_maint(self, node):
        for period in self.instance.get_periods_range(node.period, node.period_end):
            self.solution.data.set_m('state_m', node.resource, period, node.assignment, value=1)
        return 1

    def apply_task(self, node):
        for period in self.instance.get_periods_range(node.period, node.period_end):
            self.solution.data.set_m('task', node.resource, period, value=node.assignment)
        return 1

    def erase_window(self, resource, start, end, key='task'):
        # here, we will be take the risk of just deleting everything.
        data = self.solution.data
        fixed_periods = self.instance.get_fixed_periods().to_set()

        delete_assigned = []
        periods = self.instance.get_periods_range(start, end)
        if resource not in data[key]:
            return delete_assigned
        for period in periods:
            if (resource, period) in fixed_periods:
                continue
            info = data[key][resource].pop(period, None)
            if info is not None:
                delete_assigned.append((period, info))
        return delete_assigned

    def solve_repair(self, patterns):

        # patterns is a sd.SuperDict of resource: [list of possible patterns]
        _range = self.instance.get_periods_range
        combos = tl.TupList()
        info = tl.TupList()
        for res, _pat_list in patterns.items():
            for p, pattern in enumerate(_pat_list[1:-1]):
                combos.add(res, p)
                for elem in pattern:
                    for period in _range(elem.period, elem.period_end):
                        info.add(res, p, period, elem.assignment, elem.type)

        self.assign = pl.LpVariable.dicts(name="assign", indexs=combos, cat=pl.LpBinary)



if __name__ == '__main__':
    import package.params as pm
    import package.instance as inst
    import data.test_data as test_d

    data_in = test_d.dataset3()
    instance = inst.Instance(data_in)
    self = GraphOriented(instance)
    resources = self.instance.get_resources()
    # self.date_to_node(resource, '2018-12')
    date1 = '2018-03'
    date2 = '2018-08'

    # resource = '1'
    res_patterns = sd.SuperDict()
    for resource in resources:
        patterns = self.window_to_patterns(resource, date1, date2)
        res_patterns[resource] = self.filter_patterns(resource, date1, date2, patterns)

    _next = self.instance.get_next_period
    _prev = self.instance.get_prev_period

    # TODO: here we add the logic to add a model that decides patterns
    # for each resource.
    # dummy solution:
    solution_model = res_patterns.vapply(rn.choice)

    # Once we have the solution to the assignment problem:
    # we just apply each pattern to each resource
    for resource, pattern in solution_model.items():
        self.apply_pattern(pattern)



    self.solution.data

    # solution = self.solve(pm.OPTIONS)

    pass
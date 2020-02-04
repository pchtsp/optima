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
            graph, refs = cp.get_graph_of_resource(instance=instance, resource=r)
            self.instance.data['aux']['graphs'][r] = sd.SuperDict(graph = graph, refs=refs, refs_inv=refs.reverse())
        pass

    def solve(self, options):
        """
         Solves an instance using the metaheuristic.

         :param options: dictionary with options
         :return: solution to the planning problem
         :rtype: :py:class:`package.solution.Solution`
         """
        pass

    def date_to_state(self, resource, date, previous=True):
        # previous implies to look for the latest previous assignment.
        # if false, we look for the soonest next assignment
        period, category = self.get_next_assignment(resource, date, search_future=not previous)
        if period is None:
            # there are no other assignments until reaching the end of the horizon
            # TODO: Here, I just choose the source of the sink for the resource
            # depending on search_future value
            return None
        # Now we know there is something in this period and what type.
        # But we do not know if it started in this period
        period_start = self.get_start_of_assignment(resource, period, category)
        period_end = self.get_start_of_assignment(resource, period, category, get_last=True)
        assignment = self.solution.get_period_state(resource, period, cat=category)
        _type = 0
        if category == 'task':
            _type = 1

        # assumes set_remaining_usage_time_all
        maints = self.instance.get_maintenances()
        _rts = {rt:
            {m: self.get_remainingtime(resource, period_start, rt, m) for m in maints}
                for rt in ('ret', 'rut')}

        # we return the state of type: period, period_end, assignment, type, rut, ret
        return dict(period=period_start, period_end=period_end,
                    assignment=assignment, type=_type, **_rts)

    def get_graph_data(self, resource):
        return self.instance.data['aux']['graphs'][resource]

    def window_to_patterns(self, resource, date1, date2):
        state1 = self.date_to_state(resource, date1)
        state2 = self.date_to_state(resource, date2)
        node1 = cp.state_to_node(self.instance, resource=resource, state=state1)
        node2 = cp.state_to_node(self.instance, resource=resource, state=state2)
        return cp.nodes_to_patterns(node1=node1, node2=node2, **self.get_graph_data(resource))

if __name__ == '__main__':
    import package.params as pm
    import package.instance as inst
    import data.test_data as test_d

    data_in = test_d.dataset3()
    instance = inst.Instance(data_in)
    self = GraphOriented(instance)
    resource = '1'
    self.date_to_state(resource, '2018-12')
    self.window_to_patterns('1', '2018-01', '2018-12')

    solution = self.solve(pm.OPTIONS)

    pass
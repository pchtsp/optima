import pytups.superdict as sd
import pytups.tuplist as tl
import package.solution as sol
import numpy as np
import solvers.heuristics as heur
import solvers.model as mdl
import patterns.create_patterns as cp
import patterns.node as nd
import solvers.config as conf
import random as rn
import pulp as pl
import math
import time
import logging as log


class GraphOriented(heur.GreedyByMission,mdl.Model):

    # statuses for detecting a new worse solution
    status_worse = {2, 3}
    # statuses for detecting if we accept a solution
    status_accept = {0, 1, 2}
    {'aux': {'graphs': {'RESOURCE': {'graph': {}, 'refs': {}, 'refs_inv': {}, 'source': {}, 'sink': {}}}}}
    # graph=g, refs=refs, refs_inv=refs.reverse(), source=source, sink=sink

    def __init__(self, instance, solution=None):

        heur.GreedyByMission.__init__(self, instance, solution)
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

        # We check what the resource has in this period:
        assignment, category = self.solution.get_period_state_category(resource, date)
        if category is None:
            # if nothing, we format as an EMPTY node
            assignment = ''
            period_start = date
            period_end = date
        else:
            # if there is something, we look for the limits of the assignment
            # to create the correct node.
            period_start = self.get_limit_of_assignment(resource, date, category)
            period_end = self.get_limit_of_assignment(resource, date, category, get_last=True)

        if category is None:
            _type = nd.EMPTY_TYPE
        elif category == 'task':
            _type = nd.TASK_TYPE
        elif category == 'state_m':
            _type = nd.MAINT_TYPE
        else:
            _type = nd.DUMMY_TYPE

        if previous:
            maints = self.instance.get_maintenances().vapply(lambda v: 1)
            # we want to have the status of the resource in the moment the assignment ends
            # because that will be the status on the node
            _defaults = dict(resource=resource, period=period_end)
            _rts = \
                sd.SuperDict(ret=maints, rut=maints).\
                to_dictup().\
                kapply(lambda k: dict(**_defaults, time=k[0], maint=k[1])).\
                vapply(lambda v: self.get_remainingtime(**v)).\
                to_dictdict()
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
        _next = self.instance.get_next_period
        _prev = self.instance.get_prev_period
        start = _next(pattern[0].period_end)
        end = _prev(pattern[-1].period)

        deleted_tasks = self.erase_between_dates(resource, start, end, 'task')
        deleted_maints = self.erase_between_dates(resource, start, end, 'state_m')

        # We apply all the assignments
        # Warning, here I'm ignoring the two extremes[1:-1]
        added_maints = tl.TupList()
        added_tasks = tl.TupList()
        for node in pattern[1:-1]:
            if node.type == nd.TASK_TYPE:
                added_tasks += self.apply_task(node)
            elif node.type == nd.MAINT_TYPE:
                added_maints += self.apply_maint(node)

        # Update rut and ret.
        # for this we need to join all added and deleted things:
        all_del = (deleted_tasks + deleted_maints + added_maints + added_tasks).unique2().sorted()
        if all_del:
            first_date_to_update = all_del[0][0]
        else:
            first_date_to_update = start
        times = ['rut']
        if added_maints or deleted_maints:
            times.append('ret')
        for t in times:
            for m in self.instance.get_rt_maints(t):
                self.update_rt_between_dates(resource, m, first_date_to_update, end, t)
        return True

    def update_rt_between_dates(self, resource, maint, start, end, rt):
        _shift = self.instance.shift_period
        _start = start
        while _start <= end:
            # iterate over maintenance periods
            updated_periods = self.update_rt_until_next_maint(resource, _start, maint, rt)
            # TODO: what if not updated_periods
            maint_start = _shift(updated_periods[-1], 1)
            if maint_start > end:
                # we've reached the end, no more updating needed
                break
            # if the maintenance falls inside the period: update the maintenance values
            # and continue the while
            horizon_end = self.instance.get_param('end')
            maint_data = self.instance.data['maintenances'][maint]
            affected_maints = maint_data['affects']
            duration = maint_data['duration_periods']
            maint_end = min(self.instance.shift_period(maint_start, duration - 1), horizon_end)
            periods_maint = self.instance.get_periods_range(maint_start, maint_end)
            self.set_state_and_clean(resource, maint, periods_maint)
            for m in affected_maints:
                self.update_time_maint(resource, periods_maint, time='ret', maint=m)
                self.update_time_maint(resource, periods_maint, time='rut', maint=m)
            _start = _shift(maint_end, 1)
        return True

    def apply_maint(self, node):
        result = tl.TupList()
        for period in self.instance.get_periods_range(node.period, node.period_end):
            self.solution.data.set_m('state_m', node.resource, period, node.assignment, value=1)
            result.add(period, node.assignment)
        return result

    def apply_task(self, node):
        result = tl.TupList()
        for period in self.instance.get_periods_range(node.period, node.period_end):
            self.solution.data.set_m('task', node.resource, period, value=node.assignment)
            result.add(period, node.assignment)
        return result

    def erase_between_dates(self, resource, start, end, key='task'):
        """
        cleans all assignments and returns them.
        :param resource:
        :param start:
        :param end:
        :param key:
        :return: tl.TupList
        """
        # here, we will be take the risk of just deleting everything.
        data = self.solution.data
        fixed_periods = self.instance.get_fixed_periods().to_set()

        delete_assigned = tl.TupList()
        periods = self.instance.get_periods_range(start, end)
        if resource not in data[key]:
            return delete_assigned
        for period in periods:
            if (resource, period) in fixed_periods:
                continue
            info = data[key][resource].pop(period, None)
            if info is not None:
                delete_assigned.add(period, info)
        return delete_assigned

    def solve_repair(self, res_patterns, options):

        # res_patterns is a sd.SuperDict of resource: [list of possible patterns]
        _range = self.instance.get_periods_range
        _dist = self.instance.get_dist_periods
        first, last = self.instance.get_first_last_period()
        combos = sd.SuperDict()
        info = tl.TupList()
        for res, _pat_list in res_patterns.items():
            for p, pattern in enumerate(_pat_list):
                pattern_clean = pattern
                combos[res, p] = pattern_clean
                info += tl.TupList(
                    (res, p, period, elem.assignment, elem.type)
                    for elem in pattern_clean for period in _range(elem.period, elem.period_end)
                )
        weight_pattern = lambda pattern: sum(_dist(elem.period, last) for elem in pattern
                                              if elem.type == nd.MAINT_TYPE)
        assign_p = combos.vapply(weight_pattern)

        # Variables:
        assign = pl.LpVariable.dicts(name="assign", indexs=combos, cat=pl.LpBinary)
        assign = sd.SuperDict.from_dict(assign)

        l = self.domains
        if l is not None:
            l = self.domains = self.get_domains_sets(options)
        ub = self.get_variable_bounds()
        p_s = {s: p for p, s in enumerate(l['slots'])}
        p_t = self.instance.data['aux']['period_i']

        def sum_of_slots(variable_slots):
            result_col = len(variable_slots.keys_l()[0])
            indices = range(result_col - 1)
            return \
                variable_slots.\
                to_tuplist().\
                to_dict(result_col=result_col, indices=indices).\
                vapply(pl.lpSum)

        def make_slack_var(name, domain, ub_func):
            _var = sd.SuperDict()
            for tup in domain:
                _var[tup] = pl.LpVariable(name="{}_{}".format(name, tup), lowBound=0,
                                               upBound=ub_func(tup), cat=pl.LpContinuous)
            return _var

        slack_kts = make_slack_var('slack_kts', l['kts'], lambda kts: ub['slack_kts'][kts[2]])
        _slack_s_kt = sum_of_slots(slack_kts)
        slack_kts_p = {(k, t, s): (p_s[s] + 1 - 0.001 * p_t[t]) * 50 for k, t, s in l['kts']}

        slack_kts_h = make_slack_var('slack_kts_h', l['kts'],
                                     lambda kts: ub['slack_kts_h'][(kts[0], kts[2])])
        _slack_s_kt_h = sum_of_slots(slack_kts_h)
        slack_kts_h_p = {(k, t, s): (p_s[s] + 2 - 0.001 * p_t[t]) ** 2 for k, t, s in l['kts']}

        slack_ts = make_slack_var('slack_ts', l['ts'],
                                     lambda ts: ub['slack_ts'][ts[1]])
        _slack_s_t = sum_of_slots(slack_ts)
        slack_ts_p = {(t, s): (p_s[s] + 1 - 0.001 * p_t[t]) * 1000 for t, s in l['ts']}

        slack_vts = make_slack_var('slack_vts', [(*vt, s) for vt in l['vt'] for s in l['slots']],
                                  lambda vts: p_s[vts[2]]*2+1)
        _slack_s_vt = sum_of_slots(slack_vts)
        slack_vts_p = slack_vts.kapply(lambda vts: (p_s[vts[2]] + 1) * 10000)

        # Constraints
        constraints = tl.TupList()

        # objective:
        objective_function = \
            pl.lpSum((assign * assign_p).values_tl()) + \
            pl.lpSum((slack_vts * slack_vts_p).values_tl()) + \
            pl.lpSum((slack_ts * slack_ts_p).values_tl()) + \
            pl.lpSum((slack_kts * slack_kts_p).values_tl()) + \
            pl.lpSum((slack_kts_h * slack_kts_h_p).values_tl())

        # one pattern per resource:
        constraints += \
            combos.keys_tl().\
            vapply(lambda v: (v[0], *v)).to_dict(result_col=[1, 2]).\
            vapply(lambda v: pl.lpSum(assign[vv] for vv in v) == 1).\
            values_tl()

        # ##################################
        # Tasks and tasks starts
        # ##################################

        # num resources:
        mission_needs = self.check_task_num_resources()
        p_mission_needs = \
            info.\
            vfilter(lambda v: v[4]==nd.TASK_TYPE).\
            to_dict(indices=[3, 2], result_col=[0, 1])
        _constraints += \
            p_mission_needs.\
            vapply(lambda v:  pl.lpSum(assign[vv] for vv in v)).\
            kvapply(lambda k, v: v + _slack_s_vt[k] >= mission_needs[k]).\
            values_tl()

        # # ##################################
        # Clusters
        # ##################################
        c_cand = self.instance.get_cluster_candidates().list_reverse().vapply(lambda v: v[0])
        # minimum availability per cluster and period
        min_aircraft_slack = self.check_min_available(deficit_only=False)
        p_clustdate = \
            info.\
            vfilter(lambda v: v[4] == nd.MAINT_TYPE). \
            vapply(lambda v: (*v, c_cand[v[0]])). \
            to_dict(indices=[5, 2], result_col=[0, 1])

        constraints += \
            p_clustdate. \
            vapply(lambda v: pl.lpSum(assign[vv] for vv in v)). \
            kvapply(lambda k, v: v - _slack_s_kt[k] <= min_aircraft_slack[k]). \
            values_tl()

        # for (k, t), num in cluster_data['num'].items():
        #     model += \
        #         pl.lpSum(start_M[a, t1, t2] for a in c_cand[k]
        #                  for (t1, t2) in l['tt_maints_at'].get((a, t), [])
        #                  ) <= num + pl.lpSum(slack_kts[k, t, s] for s in l['slots'])

        # Each cluster has a minimum number of usage hours to have
        # at each period.
        # min_hours_slack = self.check_min_flight_hours(recalculate=False, deficit_only=False)
        # combos.\
        #     vapply(lambda v: sd.SuperDict({vv.period: vv.ret for vv in v})).\
        #     to_dictdict().\
        #     to_dictup().\
        #     to_tuplist()
        #
        # p_clustdate.vapply()
        # # TODO: flight hour constraints. This will be somewhat hard
        # for (k, t), hours in cluster_data['hours'].items():
        #     model += pl.lpSum(rut[a, t] for a in c_cand[k] if (a, t) in l['at']) >= hours - \
        #              pl.lpSum(slack_kts_h[k, t, s] for s in l['slots'])


        # maintenance capacity
        # TODO: fill periods_to_check with max window to check.
        rem_maint_capacity = self.check_sub_maintenance_capacity(ref_compare=None, periods_to_check=None)
        type_m = self.instance.get_maintenances('type')
        p_maint_used = \
            info.\
            vfilter(lambda v: v[4] == nd.MAINT_TYPE).\
            vapply(lambda v: (*v, type_m[v[3]])).\
            to_dict(indices=[5, 2], result_col=[0, 1])
        constraints += \
            p_maint_used.\
            vapply(lambda v:  pl.lpSum(assign[vv] for vv in v)).\
            kvapply(lambda k, v: v - _slack_s_t[k[1]] <= rem_maint_capacity[k]).\
            values_tl()

        # We all constraints at the same time:

        model = pl.LpProblem('MFMP_repair', sense=pl.LpMinimize)
        model += objective_function
        for c in constraints:
            model += c

        config = conf.Config(options)
        if options.get('writeMPS', False):
            model.writeMPS(filename=options['path'] + 'formulation.mps')
        if options.get('writeLP', False):
            model.writeLP(filename=options['path'] + 'formulation.lp')

        if options.get('do_not_solve', False):
            print('Not solved because of option "do_not_solve".')
            return self.solution

        result = config.solve_model(model)

        if result != 1:
            print("Model resulted in non-feasible status: {}".format(result))
            return None
        print('model solved correctly')
        # tl.TupList(model.variables()).to_dict(None).vapply(pl.value).vfilter(lambda v: v)
        self.solution = self.get_repaired_solution(combos, assign)

        return self.solution

    def get_repaired_solution(self, combos, assign):
        patterns = assign.vapply(pl.value).vfilter(lambda v: v).kapply(lambda k: combos[k]).values_tl()
        for p in patterns:
            self.apply_pattern(p)
        return self.solution

    def draw_graph(self, resource):
        graph_data = self.get_graph_data(resource)
        cp.draw_graph(self.instance, graph_data['graph'], graph_data['refs_inv'])
        return True

if __name__ == '__main__':
    import package.params as pm
    import package.instance as inst
    import data.test_data as test_d

    # data_in = test_d.dataset3_no_default()
    data_in = test_d.dataset4()
    instance = inst.Instance(data_in)
    self = GraphOriented(instance)
    resources = self.instance.get_resources()
    # self.draw_graph("1")
    # self.draw_graph("2")
    # data_graph = self.get_graph_data('1')
    # vertices = tl.TupList(data_graph['graph'].vertices()).\
    #     vapply(lambda v: data_graph['refs_inv'][v]).\
    #     vfilter(lambda v: v.rut is None and v.period=='2018-02')
    # vertices[1]
    # vertices[0]
    # vertices[1]

    # self.date_to_node(resource, '2018-12')
    date1 = '2018-02'
    # date2 = '2018-09'
    date2 = '2019-10'

    # resource = '1'
    res_patterns = sd.SuperDict()
    for resource in resources:
        patterns = self.window_to_patterns(resource, date1, date2)
        # res_patterns[resource] = self.filter_patterns(resource, date1, date2, patterns)
        res_patterns[resource] = patterns

    data = self.solve_repair(res_patterns, options=pm.OPTIONS)
    # data.data

    # solution = self.solve(pm.OPTIONS)

    pass

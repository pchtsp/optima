import package.instance as inst

import pytups.superdict as sd
import pytups.tuplist as tl
import ujson as json
import math
import numpy as np


MAINT_TYPE = 0
TASK_TYPE = 1
DUMMY_TYPE = -1
EMPTY_TYPE = 2

class Node(object):
    """
    corresponds to a combination of: a period t, a maintenance m, and the remaining rets and ruts for all maintenances
    """

    def __init__(self, instance, resource, period, ret, rut, assignment, period_end, type=MAINT_TYPE):
        """
        :param inst.Instance instance: input data.
        :param str period: period where assignment takes place
        :param str period_end: period where unavailability ends
        :param sd.SuperDict ret: remaining elapsed time at the end of the period
        :param sd.SuperDict rut: remaining usage time at the end of the period
        :param str or None assignment: a maintenance or a task.
        :param int type: 0 => maintenance, 1 => task, -1 => dummy, 2 => empty

        """
        self.instance = instance
        self.ret = ret
        self.rut = rut
        self.assignment = assignment
        self.type = type
        self.period = period
        self.period_end = period_end
        self.resource = resource
        data = self.get_data()
        self.jsondump = json.dumps(data, sort_keys=True)
        self.hash = hash(self.jsondump)
        self._backup_tasks = None
        self._backup_maints = None
        self._backup_vtt2_between_tt = None
        self._backup_vtt2_that_start_t = None
        self._backup_vtt2_that_start_t_and_end_before_t2 = None
        return

    @classmethod
    def from_node(cls, node, **kwargs):
        """
        :param Node node: another node to copy from
        :param kwargs: replacement properties for new node
        :return:
        """
        chars = ['instance', 'resource', 'period', 'period_end', 'ret', 'rut', 'assignment', 'type']
        data = {v: getattr(node, v) for v in chars}
        data = sd.SuperDict(data)
        for k, v in kwargs.items():
            data[k] = v
        new_node = cls(**data)

        # we keep the cache from the previous node:
        new_node._backup_tasks = node._backup_tasks
        new_node._backup_maints = node._backup_maints
        new_node._backup_vtt2_between_tt = node._backup_vtt2_between_tt
        new_node._backup_vtt2_that_start_t = node._backup_vtt2_that_start_t
        new_node._backup_vtt2_that_start_t_and_end_before_t2 = node._backup_vtt2_that_start_t_and_end_before_t2
        return new_node

    @classmethod
    def from_state(cls, instance, resource, state):
        return Node(instance=instance, resource=resource, **state)

    def __repr__(self):
        return repr('({}<>{}) => {}'.format(self.period, self.period_end, self.assignment))

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.jsondump == other.jsondump

    def get_data(self):
        return sd.SuperDict({'ret': self.ret, 'rut': self.rut,
                                       'assignment': self.assignment,
                                       'period': self.period,
                                       'period_end': self.period_end,
                                       'type': self.type})

    def get_maints_rets(self):
        return self.ret.keys()

    def get_maints_ruts(self):
        return self.rut.keys()

    def is_maint(self):
        return self.type == MAINT_TYPE

    def is_task(self):
        return self.type == TASK_TYPE

    def get_tasks_data(self, param):
        if not self._backup_tasks:
            self._backup_tasks = sd.SuperDict()
        if param not in self._backup_tasks:
            self._backup_tasks[param] = self.instance.get_tasks(param)
        return self._backup_tasks[param]

    def get_maints_data(self, param):
        if not self._backup_maints:
            self._backup_maints = sd.SuperDict()
        if param not in self._backup_maints:
            self._backup_maints[param] = self.instance.get_maintenances(param)
        return self._backup_maints[param]

    def get_maint_options_ret(self):
        elapsed_time_sizes = self.get_maints_data('elapsed_time_size')
        last = self.instance.get_param('end')
        maints_ret = self.get_maints_rets()
        opts_ret = sd.SuperDict()
        for m in maints_ret:
            start = max(self.ret[m] - elapsed_time_sizes[m] + 1, 1)
            end = min(self.ret[m], self.dif_period_end(last))
            _range = range(int(start), int(end)+1)
            if _range:
                opts_ret[m] = set(_range)
        return opts_ret

    def get_vtt2(self):
        tasks = self.instance.get_task_candidates(resource=self.resource).vfilter(lambda v: v)
        periods = self.instance.get_periods()
        start_time = self.get_tasks_data('start')
        end_time = self.get_tasks_data('end')
        _range = self.instance.get_periods_range
        vt = tl.TupList((v, t) for v in tasks for t in _range(start_time[v], end_time[v]))
        t_v = vt.to_dict(result_col=1)
        p_pos = {periods[pos]: pos for pos in range(len(periods))}
        _prev = self.instance.get_prev_period
        min_assign = self.instance.get_tasks('min_assign')
        first_period, last_period = self.instance.get_start_end()
        at_mission_m = self.instance.get_fixed_tasks(resource=self.resource).take([1, 2])
        at_mission_m_horizon = at_mission_m.vfilter(lambda x: first_period <= x[1] <= last_period).to_set()

        vtt2 = tl.TupList([(v, t1, t2) for (v, t1) in vt for t2 in t_v[v] if
                            (p_pos[t2] >= p_pos[t1] + min_assign[v] - 1) or
                            (p_pos[t2] >= p_pos[t1] and t2 == last_period)
                            ])

        # For Start-stop options, during fixed periods, we do not care of the minimum time assignment.
        vtt2_fixed = tl.TupList([(v, t1, t2) for (v, t1) in vt for t2 in t_v[v] if
                                  (p_pos[t2] >= p_pos[t1]) and
                                  ((v, t1) in at_mission_m_horizon or
                                   (v, t2) in at_mission_m_horizon or
                                   (v, _prev(t1)) in at_mission_m)
                                  ])
        vtt2.extend(vtt2_fixed)
        # we had a repetition problem:
        vtt2 = vtt2.unique2()
        return vtt2

    def get_assignments_start_date(self, period):
        if self._backup_vtt2_that_start_t is not None:
            return self._backup_vtt2_that_start_t.get(period, tl.TupList())
        vtt2 = self.get_vtt2()
        self._backup_vtt2_that_start_t = \
            vtt2.\
                to_dict(result_col=[0, 1, 2], indices=[1]).\
                vapply(lambda v: v.sorted())
        return self._backup_vtt2_that_start_t.get(period, tl.TupList())

    def get_assignments_start_date_end_before_date(self, period1, period2):
        if self._backup_vtt2_that_start_t_and_end_before_t2 is not None:
            return self._backup_vtt2_that_start_t_and_end_before_t2.get((period1, period2), tl.TupList())
        vtt2 = self.get_vtt2()
        periods = self.instance.get_periods()
        vtt2_before_t = {t: vtt2.vfilter(lambda x: x[2] <= t) for t in periods}
        vtt2_before_t = sd.SuperDict(vtt2_before_t).vapply(set)
        vtt2_start_t = vtt2.to_dict(result_col=[0, 1, 2], indices=[1]).fill_with_default(periods, []).vapply(set)
        result = \
            {(t1, t2): vtt2_start_t[t1] & vtt2_before_t[t2]
             for pos1, t1 in enumerate(periods) for t2 in periods[pos1:]}
        self._backup_vtt2_that_start_t_and_end_before_t2 = \
            sd.SuperDict(result).vapply(lambda v: tl.TupList(v).sorted())
        return self._backup_vtt2_that_start_t_and_end_before_t2.get((period1, period2), tl.TupList())

    def get_assignments_between_dates(self, period1, period2):
        if self._backup_vtt2_between_tt is not None:
            return self._backup_vtt2_between_tt.get((period1, period2), tl.TupList())
        vtt2 = self.get_vtt2()
        periods = self.instance.get_periods()
        vtt2_after_t = {t: vtt2.vfilter(lambda x: x[1] >= t) for t in periods}
        vtt2_after_t = sd.SuperDict(vtt2_after_t).vapply(set)
        vtt2_before_t = {t: vtt2.vfilter(lambda x: x[2] <= t) for t in periods}
        vtt2_before_t = sd.SuperDict(vtt2_before_t).vapply(set)
        self._backup_vtt2_between_tt = {(t1, t2): vtt2_after_t[t1] & vtt2_before_t[t2]
                            for pos1, t1 in enumerate(periods) for t2 in periods[pos1:]}
        self._backup_vtt2_between_tt = \
            sd.SuperDict(self._backup_vtt2_between_tt).\
            vapply(lambda v: tl.TupList(v).sorted())
        return self._backup_vtt2_between_tt.get((period1, period2), tl.TupList())

    def get_maint_options_rut(self):
        return sd.SuperDict()
        # TODO: maybe reformulate this so it's not run for missions.
        #   I need to redo this whole function to work with missions and multiple maintenances
        last = self.instance.get_param('end')
        acc_cons = 0
        maints_rut = self.get_maints_ruts()
        opts_rut = sd.SuperDict({m: set() for m in maints_rut})
        maints_rut = set(maints_rut)
        dif_m = self.get_maints_data('used_time_size')
        max_period = None
        for num, period in enumerate(self.iter_until_period(last)):
            acc_cons += self.get_consume(period)
            for m in maints_rut:
                rut_future = self.rut[m] - acc_cons
                if rut_future < 0:
                    # we have reached the first limit on maintenance, we should stop.
                    max_period = num
                    if num == 0:
                        # if its the first period and we're already too late,
                        # we still try.
                        opts_rut[m].add(num)
                    break
                elif rut_future < dif_m[m]:
                    # we still can do the maintenance in this period
                    # but we don't want the same one at 0; duh.
                    if (self.type != MAINT_TYPE) or \
                        (m != self.assignment and num != 0):
                        opts_rut[m].add(num)
            # it means we reached the upper limit of at least
            # one of the maintenances
            if max_period is not None:
                break
        return opts_rut

    def get_adjacency_list(self, only_next_period=False):

        hard_last = self.instance.get_param('end')
        if self.period_end >= hard_last:
            return [get_sink_node(self.instance, self.resource)]
        opts_tot = self.get_maint_options()
        adj_per_maints = self.get_adjacency_list_maints(opts_tot=opts_tot, only_next_period=only_next_period)
        mandatory_maint_period = self.get_last_possible_nonmaint_period(opts_tot=opts_tot)
        extra_nodes = []
        if mandatory_maint_period == hard_last or self.dif_period_end(mandatory_maint_period) > 0:
            # if there is no mandatory maintenance: we can do nothing.
            extra_nodes.extend(self.get_adjancency_list_nothing())
        return adj_per_maints + extra_nodes + \
               self.get_adjacency_list_tasks(mandatory_maint_period, only_next_period=only_next_period)

    def get_adjancency_list_nothing(self):
        # distance = self.dif_period_1(self.period_end)
        node = self.create_adjacent(assignment="", num_periods=1, duration=1, type=EMPTY_TYPE)
        if node is None:
            return []
        return [node]

    def get_maint_options(self):
        opts_ret = self.get_maint_options_ret()
        opts_rut = self.get_maint_options_rut()

        intersect = opts_rut.keys() & opts_ret.keys()
        int_dict = \
            opts_ret. \
            filter(intersect). \
            kvapply(lambda k, v: opts_rut[k] & v). \
            kvapply(lambda k, v: v if len(v) else opts_ret[k])

        opts_tot = opts_rut.clean(func=lambda v: v)
        opts_tot.update(opts_ret)
        opts_tot.update(int_dict)
        return opts_tot

    def get_last_possible_nonmaint_period(self, opts_tot):
        # returns the moment where maintenance will be mandatory

        hard_last = self.instance.get_param('end')
        if not opts_tot:
            return hard_last
        max_opts = min(opts_tot.vapply(lambda l: max(l)).values())
        next_to_last = self.shift_period_end(max_opts)
        # next_to_last = max(n.period for n in adj_per_maints)
        # only update if it really is before the last period.
        if next_to_last < hard_last:
            return self.instance.get_prev_period(next_to_last)
        else:
            return hard_last

    # @profile
    def get_adjacency_list_maints(self, opts_tot, only_next_period=False):
        """
        gets all nodes that are reachable from this node.
        :return: a list of nodes.
        """
        if not opts_tot:
            return []
        if only_next_period:
            # soonest possible maintenance
            min_opts = min(opts_tot.vapply(lambda l: min(l)).values())
            # if the soonest maintenance is more than a period away: do not bother
            if min_opts > 1:
                return []
            opts_tot = opts_tot.vfilter(lambda v: min_opts in v).vapply(lambda v: {min_opts})
        max_opts = min(opts_tot.vapply(lambda l: max(l)).values())
        opts_tot = opts_tot.vapply(lambda v: [vv for vv in v if vv <= max_opts]).vapply(set)
        durations = self.get_maints_data('duration_periods')

        candidates = [self.create_adjacent(assignment=maint, num_periods=opt, duration=durations[maint], type=MAINT_TYPE)
                      for maint, opts in opts_tot.items() for opt in opts]
        return [c for c in candidates if c is not None and c.assignment is not None]

    def get_adjacency_list_tasks(self, max_period_to_check, only_next_period=False):

        min_period_to_check = self.shift_period_end()
        if only_next_period:
            possible_assignments_task = \
                self.get_assignments_start_date_end_before_date(min_period_to_check, max_period_to_check)
        else:
            possible_assignments_task = \
                self.get_assignments_between_dates(min_period_to_check, max_period_to_check)

        diff = self.dif_period_end
        dist = self.instance.get_dist_periods

        def prepare_tuple(tuple):
            task, period1, period2 = tuple
            return task, diff(period1), dist(period1, period2)+1

        # 2. budget
        consumption = self.get_tasks_data('consumption')
        min_rut = min(self.rut.values())
        max_duration = consumption.vapply(lambda v: math.floor(min_rut / v))

        # potentially controversial hypothesis implemented.
            # not two consecutive assignments of the same task
        # TODO: use numpy
        if self.type == TASK_TYPE:
            possible_assignments_task = \
                possible_assignments_task. \
                vfilter(lambda v: v[0] != self.assignment)
        return possible_assignments_task.\
            vapply(prepare_tuple).\
            vfilter(lambda v: v[2] <= max_duration[v[0]]).\
            vapply(lambda v: self.create_adjacent(*v, type=TASK_TYPE)).\
            vfilter(lambda v: v is not None)

    def calculate_rut(self, assignment, period, num_periods):
        """
        If next m=assignment and is done in t=period, how does rut[m][t] change?
        :param str assignment:
        :param str period:
        :param num_periods: number of periods for assignment to mission
        :return:
        """
        maints_ruts = self.get_maints_ruts()
        m_affects = set()
        affects = self.get_maints_data('affects')
        task_consum = 0
        default_period_end = period
        if assignment:
            # there is a maintenance or mission
            if assignment in affects:
                # if it's a maintenance, we check which maintenances to restart
                m_affects = affects[assignment] & maints_ruts
            else:
                # if it's a task, we calculate the amount of flight hours
                tasks_consumption = self.get_tasks_data('consumption')
                task_consum = tasks_consumption.get(assignment, 0) * num_periods
            # in order to count default hours, we do not count "period"
            prev = self.instance.get_prev_period
            default_period_end = prev(period)

        m_not_affect = maints_ruts - m_affects

        # we want consumption starting after the end of the present node (period_end).
        # until the period we are arriving to (period).
        # we only count "period" if it's a blank period.
        total_consumption = sum(self.get_consume(p) for p in
                                self.iter_until_period(default_period_end))
        # if it's a task, we need to add the consumption of the task.
        total_consumption += task_consum

        # for not affected maintenances, we reduce rut:
        rut = sd.SuperDict({m: self.rut[m] - total_consumption for m in m_not_affect})

        # if no maintenance is affected, we leave early:
        if not len(m_affects):
            return rut
        # for affected maintenances, we set at max:
        max_used_times = self.get_maints_data('max_used_time')
        for m in m_affects:
            rut[m] = max_used_times[m]
            # We do not add a fake consumption because we now only count consumption
            # from the end of the maintenance
        return rut

    def calculate_ret(self, assignment, period, num_periods):
        """
        If next m=assignment and is done in t=period for duration=num_periods, how does ret change?
        :param str assignment: a maintenance or a task
        :param str period:
        :return:
        """
        # for each maint, except the affected ones, I reduce the ret
        # for maint, I put it at max
        # in case maint is a mission, there are no affected ones.
        maints_rets = self.get_maints_rets()
        m_affects = set()
        affects = self.get_maints_data('affects')

        if assignment is not None and assignment in affects:
            m_affects = set(affects[assignment]) & maints_rets

        m_not_affect = maints_rets - m_affects
        time = self.dif_period_end(period) + num_periods - 1
        ret = sd.SuperDict({m: self.ret[m] - time for m in m_not_affect})

        # in case of no affects, we leave early:
        if not len(m_affects):
            return ret
        # for affected maintenances, we set at max:
        max_elapsed_times = self.get_maints_data('max_elapsed_time')
        for m in m_affects:
            ret[m] = max_elapsed_times[m]
        return ret

    def shift_period_end(self, num=1):
        return self.instance.shift_period(self.period_end, num)

    def dif_period_end(self, period):
        return self.instance.get_dist_periods(self.period_end, period)

    def get_consume(self, period):
        return self.instance.get_default_consumption(self.resource, period)

    def iter_until_period(self, period):
        return self.iter_between_periods(self.period_end, period)

    def iter_between_periods(self, start, end):
        current = start
        while current < end:
            current = self.instance.get_next_period(current)
            yield current

    def create_adjacent(self, assignment, num_periods, duration, type=0):
        """

        :param str assignment: actual thing to do
        :param int num_periods: periods since the end of the present node (self.period_end)
            and the start of the new_node (self.period)
        :param int duration: periods the new_node will take (new_node.period => new_node.period_end)
        :param int type: type of assignment (maintenance, mission, dummy, etc.)
        :return: Node
        """
        period = self.shift_period_end(num_periods)
        period_end = self.shift_period_end(num_periods + duration - 1)
        last = self.instance.get_param('end')
        if period_end > last:
            # this should only happen with maintenances and blank nodes.
            period_end = last
        ret = self.calculate_ret(assignment, period, duration)
        rut = self.calculate_rut(assignment, period, duration)
        if type != MAINT_TYPE:
            ret_min = min(ret.values())
            rut_min = min(rut.values())
            if rut_min < 0 or ret_min <= 0:
                return None
        return Node.from_node(self, period=period, assignment=assignment, rut=rut, ret=ret, period_end=period_end, type=type)

    def walk_over_nodes(self):
        """

        :param node: node from where we start the DFS
        :return: all arcs
        """

        remaining_nodes = [self]
        # we store the neighbors of visited nodes, not to recalculate them
        cache_neighbors = sd.SuperDict()
        i = 0
        last_node = get_sink_node(self.instance, self.resource)
        ct = self.instance.compare_tups
        fixed = \
            self.instance.\
            get_fixed_states(resource=self.resource, filter_horizon=True).\
            to_start_finish(ct, pp=2)
        if len(fixed):
            # we start from the initial fixed nodes
            # instead of starting from the initial node
            neighbors = self.get_fixed_initial_nodes(fixed)
            cache_neighbors[self] = neighbors
            # we replace remaining_nodes, this losing the origin.
            remaining_nodes = list(neighbors)

        while len(remaining_nodes) and i < 10000000:
            i += 1
            node = remaining_nodes.pop()
            # we need to make a copy of the path
            if node == last_node:
                # if last_node reached, go back
                continue
            # we're not in the last_node.
            neighbors = cache_neighbors.get(node)
            if neighbors is None:
                # I don't have any cache of the node.
                # I'll get neighbors and do cache
                cache_neighbors[node] = neighbors = node.get_adjacency_list(only_next_period=True)
                # since the node is new, we want to visit it's neighbors
                remaining_nodes += neighbors
            # log.debug("iteration: {}, remaining: {}, stored: {}".format(i, len(remaining_nodes), len(cache_neighbors)))
        return cache_neighbors

    def get_fixed_initial_nodes(self, fixed):
        _, assignment, start, end = fixed[0]
        # we take out the source from the list
        if assignment == 'M':
            # if the fixed assignment is a maintenance
            # we just need to create a small(er) maintenance node
            return [self.create_adjacent(assignment, 1, self.dif_period_end(end), 0)]
        # if the fixed assignment is a task
        # we have to look for nodes that comply with the fixed period
        neighbors = self.get_adjacency_list(only_next_period=True)
        return \
            tl.TupList(neighbors). \
                vfilter(lambda v: v.period <= start and
                                  v.period_end >= end and
                                  v.assignment == assignment and
                                  v.type == TASK_TYPE)


def get_source_node(instance, resource):
    start = instance.get_param('start')
    period = instance.get_prev_period(start)
    resources = instance.get_resources()

    maints = instance.get_maintenances()
    rut = \
        maints.\
        kapply(lambda m: resources[resource]['initial'][m]['used']).\
        clean(func=lambda v: v is not None)
    ret = \
        maints.\
        kapply(lambda m: resources[resource]['initial'][m]['elapsed']).\
        clean(func=lambda v: v is not None)
    return Node(instance=instance, resource=resource, period=period, ret=ret, rut=rut, assignment='',
                period_end=period, type=EMPTY_TYPE)


def get_sink_node(instance, resource):
    last = instance.get_param('end')
    last_next = instance.get_next_period(last)
    defaults = dict(instance=instance, resource=resource)
    return Node(period=last_next, assignment='', rut=None,
                ret=None, period_end = last_next, type=EMPTY_TYPE, **defaults)



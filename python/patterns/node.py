import ujson as json
import pytups.superdict as sd
import pytups.tuplist as tl
import math


class Node(object):
    """
    corresponds to a combination of: a period t, a maintenance m, and the remaining rets and ruts for all maintenances
    """

    def __init__(self, instance, resource, period, ret, rut, assignment, period_end, type=0, node=None):
        """
        :param inst.Instance instance: input data.
        :param period: period where assignment takes place
        :param period_end: period where unavailability ends
        :param sd.SuperDict ret: remaining elapsed time at the end of the period
        :param sd.SuperDict rut: remaining usage time at the end of the period
        :param assignment: a maintenance or a task.
        :param type: 0 means it's assignment is a maintenance, 1 means it's a task, -1 means dummy
        :param node: another node to copy from

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
        self.hash = hash(json.dumps(data, sort_keys=True))
        self.data_set = data.to_dictup().to_tuplist().to_set()
        self._backup_tasks = None
        self._backup_maints = None
        self._backup_vtt2_between_tt = None
        if node:
            self._backup_tasks = node._backup_tasks
            self._backup_maints = node._backup_maints
            self._backup_vtt2_between_tt = node._backup_vtt2_between_tt
        return

    def __repr__(self):
        return repr('({}<>{}) => {}'.format(self.period, self.period_end, self.assignment))

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return not (self.data_set ^ other.data_set)

    def get_data(self):
        return sd.SuperDict.from_dict({'ret': self.ret, 'rut': self.rut,
                                       'assignment': self.assignment,
                                       'period': self.period,
                                       'period_end': self.period_end,
                                       'type': self.type})

    def get_maints_rets(self):
        return self.ret.keys()

    def get_maints_ruts(self):
        return self.rut.keys()

    def is_maint(self):
        return self.type == 0

    def is_task(self):
        return self.type == 1

    def get_tasks_data(self, param=None):
        if not self._backup_tasks:
            self._backup_tasks = self.instance.get_tasks()
        if not param:
            return self._backup_tasks
        return self._backup_tasks.get_property(param)

    def get_maints_data(self, param):
        if not self._backup_maints:
            self._backup_maints = self.instance.get_maintenances()
        if not param:
            return self._backup_maints
        return self._backup_maints.get_property(param)

    def get_maint_options_ret(self):
        elapsed_time_sizes = self.get_maints_data('elapsed_time_size')
        last = self.instance.get_param('end')
        maints_ret = self.get_maints_rets()
        opts_ret = sd.SuperDict()
        for m in maints_ret:
            start = max(self.ret[m] - elapsed_time_sizes[m], self.dif_period_1(self.period_end))
            end = min(self.ret[m], self.dif_period(last))
            _range = range(int(start), int(end)+1)
            if _range:
                opts_ret[m] = _range
                opts_ret[m] = set(opts_ret[m])
        return opts_ret

    def get_assignment_between_dates(self, period1, period2):
        if self._backup_vtt2_between_tt is not None:
            return self._backup_vtt2_between_tt.get((period1, period2), tl.TupList())
        tasks = self.instance.get_task_candidates(resource=self.resource).vfilter(lambda v: v)
        periods = self.instance.get_periods()
        task_data = self.get_tasks_data()
        start_time = task_data.get_property('start')
        end_time = task_data.get_property('end')
        _range = self.instance.get_periods_range
        vt = tl.TupList((v, t) for v in tasks for t in _range(start_time[v], end_time[v]))
        t_v = vt.to_dict(result_col=1)
        p_pos = {periods[pos]: pos for pos in range(len(periods))}
        min_assign = self.instance.get_tasks('min_assign')
        last_period = self.instance.get_param('end')

        vtt2 = tl.TupList([(v, t1, t2) for (v, t1) in vt for t2 in t_v[v] if
                            (p_pos[t2] >= p_pos[t1] + min_assign[v] - 1) or
                            (p_pos[t2] >= p_pos[t1] and t2 == last_period)
                            ])
        # vtt2_a = vtt2.to_dict(result_col=[1, 2, 3]).vapply(tl.TupList)
        vtt2_after_t = {t: vtt2.vfilter(lambda x: x[1] >= t) for t in periods}
        vtt2_after_t = sd.SuperDict.from_dict(vtt2_after_t).vapply(set)
        vtt2_before_t = {t: vtt2.vfilter(lambda x: x[2] <= t) for t in periods}
        vtt2_before_t = sd.SuperDict.from_dict(vtt2_before_t).vapply(set)
        self._backup_vtt2_between_tt = {(t1, t2): vtt2_after_t[t1] & vtt2_before_t[t2]
                            for pos1, t1 in enumerate(periods) for t2 in periods[pos1:]}
        self._backup_vtt2_between_tt = \
            sd.SuperDict.\
            from_dict(self._backup_vtt2_between_tt).\
            vapply(lambda v: tl.TupList(v).sorted())
        return self._backup_vtt2_between_tt.get((period1, period2), tl.TupList())


    def get_maint_options_rut(self):
        return sd.SuperDict()
        # TODO: maybe reformulate this so it's not run for missions.
        last = self.instance.get_param('end')
        acc_cons = 0
        maints_rut = self.get_maints_ruts()
        opts_rut = sd.SuperDict.from_dict({m: set() for m in maints_rut})
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
                    # TODO: maybe check this now that it's assignment instead of maint
                    if m != self.assignment and num != 0:
                        opts_rut[m].add(num)
            # it means we reached the upper limit of at least
            # one of the maintenances
            if max_period is not None:
                break
        return opts_rut

    def get_adjacency_list(self):
        adj_per_maints = self.get_adjacency_list_maints()
        hard_last = self.instance.get_param('end')
        extra_nodes = []
        last = hard_last
        if adj_per_maints:
            next_to_last = max(n.period for n in adj_per_maints)
            # only update if it really is before the last period.
            if next_to_last < hard_last:
                last = self.instance.get_prev_period(next_to_last)
        if last == hard_last:
            # this means that we are not obliged to any node, we can go to the end.
            extra_nodes = [get_sink_node(self.instance, self.resource)]
        # real_last = self.instance.get_param('end')
        return adj_per_maints + self.get_adjacency_list_tasks(last) + extra_nodes

    # @profile
    def get_adjacency_list_maints(self):
        """
        gets all nodes that are reachable from this node.
        :return: a list of nodes.
        """
        last = self.instance.get_param('end')
        if self.period >= last:
            return []
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
        # print(opts_tot)
        # opts_tot = opts_rut._update(opts_ret)._update(int_dict).clean(func=lambda v: v)
        if not opts_tot:
            return []
        max_opts = min(opts_tot.vapply(lambda l: max(l)).values())
        opts_tot = opts_tot.vapply(lambda v: [vv for vv in v if vv <= max_opts]).vapply(set)
        durations = self.get_maints_data('duration_periods')
        candidates = [self.create_adjacent(assignment=maint, num_periods=opt, duration=durations[maint], type=0)
                      for maint, opts in opts_tot.items() for opt in opts]
        return [c for c in candidates if c.assignment is not None]

    def get_adjacency_list_tasks(self, max_period_to_check):

        shift = self.instance.shift_period
        diff = self.dif_period
        dist = self.instance.get_dist_periods
        min_period_to_check = shift(self.period_end, 1)
        possible_assignments_task = \
            self.get_assignment_between_dates(min_period_to_check, max_period_to_check)
        def prepare_tuple(tuple):
            task, period1, period2 = tuple
            return task, diff(period1), dist(period1, period2)+1

        # 2. budget
        task_data = self.get_tasks_data()
        min_rut = min(self.rut.values())
        max_duration = task_data.\
            get_property('consumption').\
            vapply(lambda v: math.floor(min_rut / v))

        return possible_assignments_task.vapply(prepare_tuple).\
            vfilter(lambda v: v[2] <= max_duration[v[0]]).\
            vapply(lambda v: self.create_adjacent(*v, type=1))

    def calculate_rut(self, assignment, period, num_periods):
        """
        If next m=assignment and is done in t=period, how does rut[m][t] change?
        :param str assignment:
        :param str period:
        :param num_periods: number of periods for assignment to mission
        :return:
        """
        # TODO: rounding down to
        maints_ruts = self.get_maints_ruts()
        m_affects = set()
        affects = self.get_maints_data('affects')
        task_consum = 0
        if assignment is not None:
            if assignment in affects:
                m_affects = affects[assignment] & maints_ruts
            else:
                tasks_consumption = self.get_tasks_data('consumption')
                task_consum = tasks_consumption.get(assignment, 0) * num_periods

        m_not_affect = maints_ruts - m_affects

        # we want consumption starting after the end of the present period.
        # until the period we are arriving to.
        total_consumption = sum(self.get_consume(p) for p in self.iter_until_period(period))
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

    def calculate_ret(self, assignment, period):
        """
        If next m=assignment and is done in t=period, how does ret change?
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
        time = self.dif_period(period)
        ret = sd.SuperDict({m: self.ret[m] - time for m in m_not_affect})

        # in case of no affects, we leave early:
        if not len(m_affects):
            return ret
        # for affected maintenances, we set at max:
        max_elapsed_times = self.get_maints_data('max_elapsed_time')
        duration_periods = self.get_maints_data('duration_periods')
        for m in m_affects:
            ret[m] = max_elapsed_times[m] + \
                     duration_periods[assignment]
        return ret

    def next_period(self, num=1):
        return self.instance.shift_period(self.period, num)

    def dif_period(self, period):
        return self.instance.get_dist_periods(self.period, period)

    def dif_period_1(self, period):
        return self.instance.get_dist_periods(self.period, period) + 1

    def get_consume(self, period):
        return self.instance.get_default_consumption(self.resource, period)

    def iter_until_period(self, period):
        return self.iter_between_periods(self.period_end, period)

    def iter_between_periods(self, start, end):
        # TODO: I need to be sure this logic has no off-by-one errors.
        current = start
        while current < end:
            current = self.instance.get_next_period(current)
            yield current

    def create_adjacent(self, assignment, num_periods, duration, type=0):
        period = self.next_period(num_periods)
        period_end = self.next_period(num_periods + duration - 1)
        last = self.instance.get_param('end')
        if period > last:
            # we link with the last node and we finish
            period = last
            assignment = None
            type = -1
        ret = self.calculate_ret(assignment, period)
        rut = self.calculate_rut(assignment, period, duration)
        defaults = dict(instance=self.instance, resource=self.resource, node=self)
        return Node(period=period, assignment=assignment, rut=rut, ret=ret, period_end=period_end, type=type, **defaults)


def get_source_node(instance, resource):
    instance.data = sd.SuperDict.from_dict(instance.data)
    start = instance.get_param('start')
    period = instance.get_prev_period(start)
    resources = instance.get_resources()

    maints = instance.get_maintenances()
    rut = maints.kapply(lambda m: resources[resource]['initial'][m]['used']).clean(func=lambda v: v)
    ret = maints.kapply(lambda m: resources[resource]['initial'][m]['elapsed']).clean(func=lambda v: v)
    return Node(instance=instance, resource=resource, period=period, ret=ret, rut=rut, assignment=None,
                period_end=period, type=-1)


def get_sink_node(instance, resource):
    last = instance.get_param('end')
    last_next = instance.get_next_period(last)
    defaults = dict(instance=instance, resource=resource)
    return Node(period=last_next, assignment=None, rut=sd.SuperDict(),
                ret=sd.SuperDict(), period_end = last_next, type=-1, **defaults)

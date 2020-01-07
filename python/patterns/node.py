import ujson as json
import pytups.superdict as sd


class Node(object):
    """
    corresponds to a combination of: a period t, a maintenance m, and the remaining rets and ruts for all maintenances
    """

    def __init__(self, instance, resource, period, ret, rut, assignment, period_end, type=0):
        """
        :param inst.Instance instance: input data.
        :param period: period where assignment takes place
        :param period_end: period where unavailability ends
        :param sd.SuperDict ret: remaining elapsed time at the end of the period
        :param sd.SuperDict rut: remaining usage time at the end of the period
        :param assignment: a maintenance or a task.
        :param type: 0 means it's assignment is a maintenance, 1 means it's a task, -1 means dummy

        """
        self.instance = instance
        self.ret = ret
        self.rut = rut
        self.assignment = assignment
        self.type = type
        self.period = period
        self.period_end = period_end
        self.resource = resource
        self.hash = hash(json.dumps(self.get_data(), sort_keys=True))
        return

    def __repr__(self):
        return repr('({}-{}) => {}'.format(self.period, self.period_end, self.assignment))

    def get_maints_rets(self):
        return self.ret.keys()

    def get_maints_ruts(self):
        return self.rut.keys()

    def is_maint(self):
        return self.type == 0

    def is_task(self):
        return self.type == 1

    def get_tasks_data(self, *args):
        return self.instance.get_tasks(*args)

    def get_maints_data(self, *args):
        return self.instance.get_maintenances(*args)

    def get_maint_options_ret(self):
        elapsed_time_sizes = self.get_maints_data('elapsed_time_size')
        maints_ret = self.get_maints_rets()
        opts_ret = sd.SuperDict()
        for m in maints_ret:
            start = max(self.ret[m] - elapsed_time_sizes[m], 0)
            end = self.ret[m]
            _range = range(int(start), int(end))
            if _range:
                opts_ret[m] = _range
                opts_ret[m] = set(opts_ret[m])
        return opts_ret

    def get_maint_options_rut(self):
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
        max_opts = min(opts_tot.vapply(lambda l: max(l)).values())
        opts_tot = opts_tot.vapply(lambda v: [vv for vv in v if vv <= max_opts]).vapply(set)
        durations = self.get_maints_data('duration_periods')
        candidates = [self.create_adjacent(maint, opt, duration=durations[maint], type=0)
                      for maint, opts in opts_tot.items() for opt in opts]
        return [c for c in candidates if c.assignment is not None]

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
        is_task = False
        # is_maint = True
        tasks_consumption = {}
        if assignment is not None:
            if assignment in affects:
                m_affects = affects[assignment] & maints_ruts
            else:
                tasks_consumption = self.get_tasks_data('consumption')
                is_task = assignment in tasks_consumption
                # is_maint = False
        m_not_affect = maints_ruts - m_affects

        # we want consumption starting after the end of the present period.
        # until the period we are arriving to.
        total_consumption = sum(self.get_consume(p)
                                for p in self.iter_until_period(period))
        # if it's a task, we need to add the consumption of the task.
        if is_task:
            total_consumption += tasks_consumption[assignment] * num_periods

        # for not affected maintenances, we reduce rut:
        rut = sd.SuperDict()
        for m in m_not_affect:
            rut[m] = self.rut[m] - total_consumption

        # if no maintenances is affected, we leave early:
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
        ret = sd.SuperDict()
        for m in m_not_affect:
            ret[m] = self.ret[m] - time

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

    def get_consume(self, period):
        return self.instance.get_default_consumption(self.resource, period)

    def iter_until_period(self, period):
        return self.iter_between_periods(self.period_end, period)

    def iter_between_periods(self, start, end):
        # TODO: I need to be sure this logic has no off-by-one errors.
        current = start
        while current < end:
            current = self.next_period(current)
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
        defaults = dict(instance=self.instance, resource=self.resource)
        return Node(period=period, assignment=assignment, rut=rut, ret=ret, period_end=period_end, type=type, **defaults)

    def get_data(self):
        return sd.SuperDict.from_dict({'ret': self.ret, 'rut': self.rut,
                                       'assignment': self.assignment,
                                       'period': self.period,
                                       'period_end': self.period_end,
                                       'type': self.type})

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        data1 = self.get_data().to_dictup().to_tuplist().to_set()
        data2 = other.get_data().to_dictup().to_tuplist().to_set()
        dif = data1 ^ data2
        return len(dif) == 0


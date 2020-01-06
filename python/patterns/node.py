import ujson as json
import pytups.superdict as sd


class Node(object):
    """
    corresponds to a combination of: a period t, a maintenance m, and the remaining rets and ruts for all maintenances
    """

    def __init__(self, instance, resource, period, ret, rut, maint):
        """
        :param inst.Instance instance:
        :param period:
        :param sd.SuperDict ret:
        :param sd.SuperDict rut:
        :param maint:
        """
        self.instance = instance
        self.ret = ret
        self.rut = rut
        self.maint = maint
        self.period = period
        self.resource = resource
        self.maint_data = instance.data['maintenances']
        self.hash = hash(json.dumps(self.get_data(), sort_keys=True))
        return

    def __repr__(self):
        return repr('{} => {}'.format(self.period, self.maint))

    def get_maints_rets(self):
        return self.ret.keys()

    def get_maints_ruts(self):
        return self.rut.keys()

    # @profile
    def get_adjacency_list(self):
        """
        gets all nodes that are reachable from this node.
        :return: a list of nodes.
        """
        last = self.instance.get_param('end')
        if self.period >= last:
            return []

        data_m = self.maint_data
        maints_ret = self.get_maints_rets()
        opts_ret = sd.SuperDict()
        for m in maints_ret:
            start = max(self.ret[m] - data_m[m]['elapsed_time_size'], 0)
            end = self.ret[m]
            _range = range(int(start), int(end))
            if _range:
                opts_ret[m] = _range
                opts_ret[m] = set(opts_ret[m])

        acc_cons = 0
        maints_rut = self.get_maints_ruts()
        opts_rut = sd.SuperDict.from_dict({m: set() for m in maints_rut})
        maints_rut = set(maints_rut)
        dif_m = {m: info['used_time_size'] for m, info in data_m.items()}
        max_period = None
        # duration_past_maint = data_m.get_m(self.maint, 'duration_periods')
        # if duration_past_maint is None:
        # duration_past_maint = 0
        for num, period in enumerate(self.iter_until_period(last)):
            acc_cons += self.get_consume(period)
            for m in maints_rut:
                rut_future = self.rut[m] - acc_cons
                if rut_future < 0:
                    # remove_maint.append(m)
                    # we have reached the first limit on maintenances, we should stop.
                    max_period = num
                    if num == 0:
                        # if its the first period and we're already too late,
                        # we still try.
                        opts_rut[m].add(num)
                    break
                elif rut_future < dif_m[m]:
                    # we still can do the maintenance in this period
                    # but we don't want the same one at 0; duh.
                    if m != self.maint and num != 0:
                        opts_rut[m].add(num)
            # it means we reached the upper limit of at least
            # one of the maintenances
            if max_period is not None:
                break

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
        candidates = [self.create_adjacent(maint, opt) for maint, opts in opts_tot.items() for opt in opts]
        return [c for c in candidates if c.maint is not None]

    def calculate_rut(self, maint, period):
        """
        If next m=maint and is done in t=period, how does rut[m][t] change?
        :param str maint:
        :param str period:
        :return:
        """
        maints_ruts = self.get_maints_ruts()
        m_affects = set()
        if maint is not None:
            m_affects = self.maint_data[maint]['affects'] & maints_ruts
        m_not_affect = maints_ruts - m_affects
        total_consumption = sum(self.get_consume(p) for p in self.iter_until_period(period))
        # there is an assumption that the next maintenance
        # will always be duration_past_maint periods away.

        # for not affected maintenances, we reduce:
        # check if past maintenance had a duration.
        # and discount the consumption for the first months
        rut = sd.SuperDict()
        for m in m_not_affect:
            rut[m] = self.rut[m] - total_consumption

        # for affected maintenances, we set at max:
        for m in m_affects:
            rut[m] = self.maint_data[m]['max_used_time']
            duration = self.maint_data.get_m(maint, 'duration_periods')
            fake_consumption = sum(self.get_consume(p) for p in range(duration))
            rut[m] += fake_consumption

        return rut

    def calculate_ret(self, maint, period):
        """
        If next m=maint and is done in t=period, how does ret change?
        :param str maint:
        :param str period:
        :return:
        """
        # for each maint, except maint, I reduce the ret
        # for maint, I put at max
        maints_rets = self.get_maints_rets()
        m_affects = set()
        maint_data = self.maint_data
        if maint is not None:
            m_affects = set(maint_data[maint]['affects']) & maints_rets
        m_not_affect = maints_rets - m_affects
        time = self.dif_period(period)
        ret = sd.SuperDict()
        for m in m_not_affect:
            ret[m] = self.ret[m] - time
        # for affected maintenances, we set at max:
        for m in m_affects:
            ret[m] = maint_data[m]['max_elapsed_time'] + \
                     maint_data[maint]['duration_periods']
        return ret

    def next_period(self, num=1):
        return self.instance.shift_period(self.period, num)

    def dif_period(self, period):
        return self.instance.get_dist_periods(self.period, period)

    def get_consume(self, period):
        return self.instance.get_default_consumption(self.resource, period)

    def iter_until_period(self, period):
        current = self.period
        while current < period:
            yield current
            current = self.instance.get_next_period(current)

    def create_adjacent(self, maint, num_periods):
        period = self.next_period(num_periods)
        last = self.instance.get_param('end')
        if period > last:
            # we link with the last node and we finish
            period = last
            maint = None
        ret = self.calculate_ret(maint, period)
        rut = self.calculate_rut(maint, period)
        defaults = dict(instance=self.instance, resource=self.resource)
        return Node(period=period, maint=maint, rut=rut, ret=ret, **defaults)

    def get_data(self):
        return sd.SuperDict.from_dict({'ret': self.ret, 'rut': self.rut,
                                       'maint': self.maint, 'period': self.period})

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        data1 = self.get_data().to_dictup().to_tuplist().to_set()
        data2 = other.get_data().to_dictup().to_tuplist().to_set()
        dif = data1 ^ data2
        return len(dif) == 0

import solvers.model as model
import copy
import pytups.tuplist as tl
import pytups.superdict as sd


class ModelFixLP(model.Model):
    """
    1. Solve LP
    2. Get patterns where the m_itt' > 0
    3. Only use those for MILP
    """

    def solve(self, options=None):
        my_options = copy.deepcopy(options)
        my_options['relax'] = True
        relaxed_solution = model.Model.solve(self, my_options)
        l = self.domains
        if relaxed_solution is None:
            return None
        # we get non-zero maintenance variables
        l['att_maints'] = att_maints = \
            sd.SuperDict.from_dict(self.start_M).\
            vapply(lambda v: v.value()).\
            vfilter(lambda v: v).\
            keys_tl()
        # we filter the domains corresponding to the maintenance variables values.
        # this part we have copied it from the get_domains_sets method from the model.Model object
        last_period = self.instance.get_param('end')
        # this is the TTT_t set.
        # periods that are maintenance periods because of having assign a maintenance
        attt_maints = tl.TupList((a, t1, t2, t) for a, t1, t2 in att_maints for t in l['t2_at1'].get((a, t1), []))
        attt_maints += tl.TupList((a, t1, t2, t) for a, t1, t2 in att_maints for t in l['t2_at1'].get((a, t2), [])
                                  if t2 < last_period)
        attt_maints = attt_maints.unique2()
        l['tt_maints_at'] = attt_maints.to_dict(result_col=[1, 2], is_list=True)
        l['att_maints_t'] = attt_maints.to_dict(result_col=[0, 1, 2], is_list=True)
        l['tt_maints_a'] = att_maints.to_dict(result_col=[1, 2], is_list=True)

        # now, we want to avoid recalculating domains...
        my_options = copy.deepcopy(options)
        my_options['calculate_domains'] = False
        solution = model.Model.solve(self, my_options)
        return solution


class ModelFixFlexLP(model.Model):
    """
    0. A pattern (p \in P also called m_{itt'}) has a first date of maintenance t1_p = t
     and a second date of maintenance t2_p=t'
    1. Solve LP
    2. Get patterns p where there is at least one {t' | m_i(p_t)t' > 0} or {t | m_it(p_t) > 0}
    3. Only use those patterns for MILP
    """
    def solve(self, options=None):
        my_options = copy.deepcopy(options)
        my_options['relax'] = True
        relaxed_solution = model.Model.solve(self, my_options)
        l = self.domains
        if relaxed_solution is None:
            return None

        start_M_values =\
            sd.SuperDict.from_dict(self.start_M).\
            vapply(lambda v: v.value())

        # we get all t1 and t2 that have at least a maintenance.
        # for each aircraft
        all_ts_a = {a: set() for a in l['resources']}
        for (a, t1, t2), v in start_M_values.items():
            if v:
                all_ts_a[a].add(t1)
                all_ts_a[a].add(t2)

        _filter = lambda v: v[1] in all_ts_a[v[0]] or v[2] in all_ts_a[v[0]]
        l['att_maints'] = att_maints = start_M_values.keys_tl().vfilter(_filter)

        # we filter the domains corresponding to the maintenance variables values.
        # this part we have copied it from the get_domains_sets method from the model.Model object
        last_period = self.instance.get_param('end')
        # this is the TTT_t set.
        # periods that are maintenance periods because of having assign a maintenance
        attt_maints = tl.TupList((a, t1, t2, t) for a, t1, t2 in att_maints for t in l['t2_at1'].get((a, t1), []))
        attt_maints += tl.TupList((a, t1, t2, t) for a, t1, t2 in att_maints for t in l['t2_at1'].get((a, t2), [])
                                  if t2 < last_period)
        attt_maints = attt_maints.unique2()
        l['tt_maints_at'] = attt_maints.to_dict(result_col=[1, 2], is_list=True)
        l['att_maints_t'] = attt_maints.to_dict(result_col=[0, 1, 2], is_list=True)
        l['tt_maints_a'] = att_maints.to_dict(result_col=[1, 2], is_list=True)

        # now, we want to avoid recalculating domains...
        my_options = copy.deepcopy(options)
        my_options['calculate_domains'] = False
        solution = model.Model.solve(self, my_options)
        return solution


class ModelFixFlexLP_3(model.Model):
    """
    0. A pattern (p \in P also called m_{itt'}) has a first date of maintenance t1_p = t
     and a second date of maintenance t2_p=t'
    1. Solve LP, get non-zero patterns for each aircraft: SPL^*_i
    2. For each aircraft,
        first possible maintenance: t1^{min}_i = \min_{p \in SPL^*_i} \{t1_p\}
        last possible maintenance: t2^{max}_i = \max_{p \in SPL^*_i} \{t2_p\}
    3. We calculate all patterns for each aircraft: SPLNE_i
    4. We only use patterns where:
        SPLNE^{mod}_i= \{p \in SPLNE_i |t1^{min}_i <= t1_p and t2_p <= t2^{max}_i\}
    5. Only use those patterns for MILP
    """

    def solve(self, options=None):
        my_options = copy.deepcopy(options)
        my_options['relax'] = True
        last_period = self.instance.get_param('end')
        # 1.
        relaxed_solution = model.Model.solve(self, my_options)
        l = self.domains
        if relaxed_solution is None:
            return None

        start_M_values =\
            sd.SuperDict.from_dict(self.start_M).\
            vapply(lambda v: v.value())

        # 2.
        patterns_aircraft = \
            start_M_values.\
            vfilter(lambda v: v).\
            keys_tl().\
            to_dict([1, 2], is_list=True).\
            vapply(tl.TupList)
        first_maint = patterns_aircraft.vapply(lambda v: min(v.take(0)))
        last_maint = patterns_aircraft.vapply(lambda v: max(v.take(1)))

        # 3 and 4
        _filter = lambda v: v[1] >= first_maint[v[0]] and v[2] <= last_maint[v[0]]
        # l['att_maints'] = att_maints = start_M_values.keys_tl().vfilter(_filter)
        l['att_maints'] = att_maints = l['att_maints'].vfilter(_filter)

        # we filter the domains corresponding to the maintenance variables values.
        # this part we have copied it from the get_domains_sets method from the model.Model object
        # this is the TTT_t set.
        # periods that are maintenance periods because of having assign a maintenance
        attt_maints = tl.TupList((a, t1, t2, t) for a, t1, t2 in att_maints for t in l['t2_at1'].get((a, t1), []))
        attt_maints += tl.TupList((a, t1, t2, t) for a, t1, t2 in att_maints for t in l['t2_at1'].get((a, t2), [])
                                  if t2 < last_period)
        attt_maints = attt_maints.unique2()
        l['tt_maints_at'] = attt_maints.to_dict(result_col=[1, 2], is_list=True)
        l['att_maints_t'] = attt_maints.to_dict(result_col=[0, 1, 2], is_list=True)
        l['tt_maints_a'] = att_maints.to_dict(result_col=[1, 2], is_list=True)

        # now, we want to avoid recalculating domains...
        my_options = copy.deepcopy(options)
        my_options['calculate_domains'] = False
        # 5.
        solution = model.Model.solve(self, my_options)
        return solution

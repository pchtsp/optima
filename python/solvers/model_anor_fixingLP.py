import solvers.model_anor as model
import copy
import pytups.tuplist as tl
import pytups.superdict as sd


class ModelANORFixLP(model.ModelANOR):
    """
    1. Solve LP
    2. Get periods where the m_it > 0
    3. Only use those for MILP
    """

    def solve(self, options=None):
        my_options = copy.deepcopy(options)
        my_options['relax'] = True
        relaxed_solution = model.ModelANOR.solve(self, my_options)
        l = self.domains
        if relaxed_solution is None:
            return None

        # we get non-zero maintenance variables
        l['at_start_maint'] = at_free_start = \
            sd.SuperDict.from_dict(self.start_M).\
            vfilter(lambda v: v.value()).\
            keys_tl()

        # we filter the domains corresponding to the maintenance variables values.
        # this part we have copied it from the get_domains_sets method from the model.Model object
        # this is the TTT_t set.
        # periods that are maintenance periods because of having assign a maintenance
        ret_init = self.instance.get_initial_state("elapsed")
        first_period, last_period = self.instance.get_start_end()
        param_data = self.instance.get_param()
        duration = param_data['maint_duration']
        max_elapsed = param_data['max_elapsed_time'] + duration
        min_elapsed = max_elapsed - param_data['elapsed_time_size']
        ret_init_adjusted = {k: v - max_elapsed + min_elapsed for k, v in ret_init.items()}
        periods = self.instance.get_periods_range(first_period, last_period)
        p_pos = {periods[pos]: pos for pos in range(len(periods))}
        resources = self.instance.get_resources()
        min_elapsed_2M = {r: min_elapsed for r in resources}
        max_elapsed_2M = {r: max_elapsed for r in resources}

        l['at_M_ini'] = at_M_ini = tl.TupList([(a, t) for (a, t) in at_free_start
                               if ret_init_adjusted[a] <= p_pos[t] <= ret_init[a]
                               ])
        l['t_a_M_ini'] = at_M_ini.to_dict(result_col=1, is_list=True)
        l['att'] = att = tl.TupList([(a, t1, t2) for (a, t1) in at_free_start for t2 in periods
                                     if (p_pos[t1] <= p_pos[t2] < p_pos[t1] + duration)
                                     and (a, t2) in at_free_start])
        l['at1_t2'] = at1_t2 = att.to_dict(result_col=[0, 1], is_list=True)
        l['t2_at1'] = t2_at1 = att.to_dict(result_col=2, is_list=True)
        l['t_a_M_ini'] = t_a_M_ini = at_M_ini.to_dict(result_col=1, is_list=True)
        l['att_m'] = att_m = tl.TupList([(a, t1, t2) for (a, t1) in at_free_start for t2 in periods
                 if (p_pos[t1] < p_pos[t2] < p_pos[t1] + min_elapsed) and (a, t2) in at_free_start
                 ])
        l['att_maints_no_last'] = att_maints_no_last = \
            tl.TupList((a, t1, t2) for (a, t1) in at_M_ini for t2 in periods
                       if (p_pos[t1] + min_elapsed_2M[a] <= p_pos[t2] < p_pos[t1] + max_elapsed_2M[a])
                       and len(periods) <= p_pos[t2] + max_elapsed
                       and t2 < last_period
                       and (a, t2) in at_free_start
                       )
        l['att_M'] = att_M = \
            att_maints_no_last.vfilter(lambda x: p_pos[x[1]] + max_elapsed < len(periods))

        l['t_at_M'] = t_at_M = att_M.to_dict(result_col=2, is_list=True)
        l['t1_at2'] = t1_at2 = att.to_dict(result_col=1, is_list=True).fill_with_default(l['at'], [])

        # now, we want to avoid recalculating domains...
        my_options = copy.deepcopy(options)
        my_options['calculate_domains'] = False
        solution = model.ModelANOR.solve(self, my_options)
        return solution

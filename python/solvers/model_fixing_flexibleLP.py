import solvers.model as model
import copy
import pytups.tuplist as tl
import pytups.superdict as sd


class ModelFixFlexLP(model.Model):

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

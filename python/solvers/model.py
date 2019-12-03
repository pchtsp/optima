import math
import pulp as pl
import solvers.config as conf
import package.solution as sol
import package.experiment as exp
import pytups.tuplist as tl
import pytups.superdict as sd
import random as rn

import stochastic.instance_stats as istats


######################################################
# @profile

class Model(exp.Experiment):

    M = 'M'

    def __init__(self, instance, solution=None):

        if solution is None:
            solution = sol.Solution({'state': {}, 'task': {}})
        super().__init__(instance, solution)

        self.domains = {}
        self.start_M = {}
        self.start_T = {}
        self.rut = {}
        self.slack_kts_h = {}
        self.slack_kts = {}
        self.slack_ts = {}
        self.model = None

    def solve(self, options=None):
        l = self.domains
        if not self.domains or options.get('calculate_domains', True):
            l = self.domains = self.get_domains_sets(options)
        ub = self.get_variable_bounds()
        first_period, last_period = self.instance.get_start_end()
        consumption = sd.SuperDict.from_dict(self.instance.get_tasks('consumption'))
        requirement = self.instance.get_tasks('num_resource')
        rut_init = self.instance.get_initial_state("used")
        num_resource_maint = \
            sd.SuperDict.from_dict(self.instance.get_total_fixed_maintenances()). \
                fill_with_default(keys=l['periods'])
        maint_info = self.instance.get_maintenances()[self.M]
        maint_capacity = maint_info['capacity']
        min_usage = self.instance.get_param('min_usage_period')
        duration = maint_info['duration_periods']
        cluster_data = self.instance.get_cluster_constraints()
        c_cand = self.instance.get_cluster_candidates()

        # shortcut functions
        def dist(t1, t2):
            # t_2 - t_1 + 1
            return self.instance.get_dist_periods(t1, t2) + 1

        shift = self.instance.shift_period
        prev = self.instance.get_prev_period

        # In order to break some symmetries, we're gonna give a
        # (different) price for each assignment:
        price_assign = {(a, v): 0 for v in l['tasks'] for a in l['candidates'][v]}
        if options.get('noise_assignment', True):
            price_assign = {(a, v): rn.random() for v in l['tasks'] for a in l['candidates'][v]}

        # Sometimes we want to force variables to be integer.
        normally_continuous = pl.LpContinuous
        normally_integer = pl.LpInteger
        integer_problem = options.get('integer', False)
        relaxed_problem = options.get('relax', False)
        if integer_problem:
            normally_continuous = pl.LpInteger
        elif relaxed_problem:
            normally_integer = pl.LpContinuous

        # VARIABLES:
        # we save all variables as part in the object.
        # binary:
        # mission assignment
        self.start_T = start_T = pl.LpVariable.dicts(name="start_T", indexs=l['avtt2'], lowBound=0, upBound=1,
                                                     cat=normally_integer)
        # maintenance cycle
        self.start_M = start_M = pl.LpVariable.dicts(name="start_M", indexs=l['att_maints'], lowBound=0, upBound=1,
                                                     cat=normally_integer)

        # numeric:
        # remaining flight hours per period
        self.rut = rut = pl.LpVariable.dicts(name="rut", indexs=l['at0'], lowBound=0, upBound=ub['rut'],
                                             cat=normally_continuous)

        # slack variables:
        price_slack_kts = {s: (p + 1) * 50 for p, s in enumerate(l['slots'])}
        price_slack_ts = {s: (p + 1) * 1000 for p, s in enumerate(l['slots'])}
        price_slack_kts_h = {s: (p + 2) ** 2 for p, s in enumerate(l['slots'])}

        slack_kts_h, slack_ts, slack_kts = {}, {}, {}
        for tup in l['kts']:
            k, t, s = tup
            slack_kts[tup] = pl.LpVariable(name="slack_kts_{}".format(tup), lowBound=0,
                                           upBound=ub['slack_kts'][s], cat=normally_continuous)
        for tup in l['kts']:
            k, t, s = tup
            slack_kts_h[tup] = pl.LpVariable(name="slack_kts_h_{}".format(tup), lowBound=0,
                                             upBound=ub['slack_kts_h'][k, s], cat=normally_continuous)
        for tup in l['ts']:
            t, s = tup
            slack_ts[tup] = pl.LpVariable(name="slack_ts_{}".format(tup), lowBound=0,
                                          upBound=ub['slack_ts'][s], cat=normally_continuous)

        self.slack_ts = slack_ts
        self.slack_kts_h = slack_kts_h
        self.slack_kts = slack_kts

        slack_vt = {tup: 0 for tup in l['vt']}
        slack_at = {tup: 0 for tup in l['at']}

        slack_p = options.get('slack_vars')
        if slack_p == 'Yes':
            slack_vt = pl.LpVariable.dicts(name="slack_vt", lowBound=0, indexs=l['vt'], cat=normally_continuous)
            slack_at = pl.LpVariable.dicts(name="slack_at", lowBound=0, indexs=l['at'], cat=normally_continuous)
        elif slack_p is int:
            # first X months only
            first_months = self.instance.get_next_periods(first_period, slack_p)
            _vt = [(v, t) for v, t in l['vt'] if t in first_months]
            _kt = [(k, t) for k, t in l['kt'] if t in first_months]
            slack_vt = pl.LpVariable.dicts(name="slack_vt", lowBound=0, indexs=_vt, cat=normally_continuous)

        if options.get('mip_start') and self.solution is not None:
            self.fill_initial_solution()
            self.fix_variables(options.get('fix_vars', []))

        # MODEL
        self.model = model = pl.LpProblem("MFMP_v0003", pl.LpMinimize)

        # try to make the second maintenance the most late possible
        period_pos = self.instance.data['aux']['period_i']

        # OBJECTIVE:
        # maints
        objective = \
            pl.lpSum(price_assign[a, v] * task for (a, v, t, t2), task in start_T.items()) + \
            + 10 * pl.lpSum(price_slack_kts[s] * slack for (k, t, s), slack in slack_kts.items()) \
            + 10 * pl.lpSum(price_slack_kts_h[s] * slack for (k, t, s), slack in slack_kts_h.items()) \
            + 10 * pl.lpSum(price_slack_ts[s] * slack for (t, s), slack in slack_ts.items()) \
            - pl.lpSum(start_M[a, t1, t2] * period_pos[t2] for a, t1, t2 in l['att_maints']) \
            + 1000000 * pl.lpSum(slack_vt.values()) \
            + 1000 * pl.lpSum(slack_at.values())

        if not integer_problem:
            model += objective
        else:
            objective *= 100
            objective_r = objective.roundCoefs()
            model += objective_r

        # CONSTRAINTS:

        # max one task per period or unavailable state:
        for at in l['at']:
            a, t = at
            v_at = l['v_at'].get(at, [])  # possible missions for that "at"
            t1t2_list = l['tt_maints_at'].get(at, [])
            if len(v_at) + len(t1t2_list) == 0:
                continue
            model += pl.lpSum(start_T[a, v, t1, t2] for v in v_at for (t1, t2) in l['tt2_avt'][a, v, t]) + \
                     pl.lpSum(start_M[a, t1, t2] for (t1, t2) in t1t2_list) + \
                     (at in l['at_maint']) <= 1

        # ##################################
        # Tasks and tasks starts
        # ##################################

        # num resources:
        for (v, t), a_list in l['a_vt'].items():
            model += pl.lpSum(start_T[a, v, t1, t2] for a in a_list for (t1, t2) in l['tt2_avt'][a, v, t]) \
                     >= requirement[v] - slack_vt.get((v, t), 0)

        # at the beginning of the planning horizon, we may have fixed assignments of tasks.
        # we need to fix the corresponding variable.
        for avt in l['at_mission_m']:
            a, v, t = avt
            if t < first_period:
                continue
            t1t2_list = l['tt2_avt'][avt]
            model += pl.lpSum(start_T[a, v, t1, t2] for t1, t2 in t1t2_list) >= 1

        # # ##################################
        # Clusters
        # ##################################

        # minimum availability per cluster and period
        for (k, t), num in cluster_data['num'].items():
            model += \
                pl.lpSum(start_M[a, t1, t2] for a in c_cand[k]
                         for (t1, t2) in l['tt_maints_at'].get((a, t), [])
                         ) <= num + pl.lpSum(slack_kts[k, t, s] for s in l['slots'])

        # Each cluster has a minimum number of usage hours to have
        # at each period.
        for (k, t), hours in cluster_data['hours'].items():
            model += pl.lpSum(rut[a, t] for a in c_cand[k] if (a, t) in l['at']) >= hours - \
                     pl.lpSum(slack_kts_h[k, t, s] for s in l['slots'])

        # # ##################################
        # # Usage time
        # # ##################################

        # remaining used time calculations:
        for at in l['at']:
            a, t = at
            v_at = l['v_at'].get(at, [])  # possible missions for that "at"
            t1t2_list = l['tt_maints_at'].get(at, [])
            at_prev = a, l['previous'][t]
            model += rut[at] <= rut[at_prev] + ub['rut'] * pl.lpSum(start_M[a, t1, t2] for (t1, t2) in t1t2_list) \
                     - pl.lpSum(start_T[a, v, t1, t2] * (consumption[v] - min_usage)
                                for v in v_at for (t1, t2) in l['tt2_avt'][a, v, t]) \
                     - min_usage
            model += rut[at] >= ub['rut'] * pl.lpSum(start_M[a, t1, t2] for (t1, t2) in t1t2_list)

        for a in l['resources']:
            model += rut[a, l['period_0']] == rut_init[a]
        #
        # if maintenance cycle is between t1 and t2, we must have at most:
        # 1. rut_init[a] flight hours between the origin and the beginning of maint in t1
        # 2. H flight hours between maint in t1 and maint in t2.
        # 3. H flight hours between the end of the second maint and the end

        for a, t1, t2 in l['att_maints']:
            # t1 and t2 cut the horizon in three parts
            # we're going to calculate the three ranges
            part1 = first_period, prev(t1), rut_init[a]  # before the first maintenance
            part2 = shift(t1, duration), prev(t2), ub['rut']  # in between maintenances
            if t2 == last_period:
                part2 = shift(t1, duration), last_period, ub['rut']
            part3 = shift(t2, duration), last_period, ub['rut']  # after the second maintenance

            # Each part of the horizon needs to satisfy max hour consumption
            for pos, (p1, p2, limit) in enumerate([part1, part2, part3]):
                if p1 > p2:
                    continue
                _vtt2 = l['vtt2_between_att'][a, p1, p2]

                # shorter version of the number of periods between p1 and p2, inclusive
                # this applies for both constraints.

                d_p1_p2 = dist(p1, p2)
                # Max hours in between maintenances:
                _vars_tup = [
                    (start_T[a, v, t11, t22],
                     dist(t11, t22) * (consumption[v] - min_usage)
                     )
                    for v, t11, t22 in _vtt2]
                _vars_tup.append((start_M[a, t1, t2], ub['rut']))
                _constant = min_usage * d_p1_p2 - limit - ub['rut']
                model += pl.LpAffineExpression(_vars_tup, constant=_constant) <= 0

        # ##################################
        # Maintenances
        # ##################################

        # NOTE: we are now assuming one maintenance assignment (be it simple or double)

        # For small horizons, we want to add a an upper limit on maintenances
        # in case the previous constraint is not active for that resource
        for a, tt_list in l['tt_maints_a'].items():
            model += pl.lpSum(start_M[a, t1, t2] for t1, t2 in tt_list) == 1

        # max number of maintenances:
        for t in l['periods']:
            at1t1_list = l['att_maints_t'].get(t, [])
            if not len(at1t1_list):
                continue
            model += pl.lpSum(start_M[a, t1, t2] for (a, t1, t2) in at1t1_list) + \
                     num_resource_maint[t] <= \
                     maint_capacity + pl.lpSum(slack_ts.get((t, s), 0) for s in l['slots'])

        # Adding trained cuts.
        StochCuts = options.get('StochCuts', {})
        if StochCuts.get('active', False):
            types = self.instance.get_types()
            for t in types:
                self.add_stochastic_cuts(model, start_M, _type=t, options=options)

        if options.get('DetermCuts', False):
            self.get_valid_cuts(model=model, start_M=start_M, start_T=start_T)

        # ##################################
        # SOLVING
        # ##################################

        # SOLVING
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

        self.solution = self.get_solution()

        return self.solution

    def fill_initial_solution(self):
        """
        :param vars_to_fix: possible list of variables to fix. List of dicts assumed)
        :return:
        """
        l = self.domains
        if not len(l):
            raise ValueError('Model has not been solved yet. No domains are generated')

        first = self.instance.get_param('start')
        last = self.instance.get_param('end')

        start_M = self.start_M
        start_T = self.start_T
        rut = self.rut

        _next = self.instance.get_next_period
        _prev = self.instance.get_prev_period

        # we need to get the starts of the potentially two maintenances.
        # first we calculate the maintenances that start at the first period:
        fixed_maints = \
            self.instance.get_fixed_maintenances(dict_key='resource'). \
                clean(func=lambda x: first in x).keys_l()

        # by filtering the maintenances that start in the first period:
        # we are not including the fixed maintenances (that start in a previous period)
        maint_starts = \
            tl.TupList(self.get_maintenance_starts()). \
                to_dict(result_col=1). \
                clean(func=lambda x: first in x). \
                clean(func=lambda x: x not in fixed_maints). \
                vapply(lambda x: [first])

        # then we get the two starts of cycles
        # we add the maintenance at the beginning if exists
        maint_cycles = \
            self.get_all_maintenance_cycles(). \
                vapply(lambda x: [_next(ii) for i, ii in x if _next(ii) <= last][:2]). \
                apply(lambda k, v: maint_starts.get(k, []) + v)
        # if the resource has only one maintenance: we add one at the end (by convention)
        only_one = maint_cycles.to_lendict().clean(default_value=2).vapply(lambda x: [last])
        maints = maint_cycles.apply(lambda k, v: v + only_one.get(k, []))

        # task assignment only take into account assignments during the planning period:
        # start_periods = self.get_task_periods()
        ct = self.instance.compare_tups
        start_periods = self.solution.get_tasks().to_tuplist().to_start_finish(compare_tups=ct)

        # Initialize values:
        for tup in start_M:
            start_M[tup].setInitialValue(0)

        for tup in start_T:
            start_T[tup].setInitialValue(0)

        for a, (t, t2) in maints.items():
            if (a, t, t2) in start_M:
                # we check this because of fixed maints
                start_M[a, t, t2].setInitialValue(1)
            else:
                print('fail fixing maintenance in {}'.format((a, t, t2)))

        for (a, t, v, t2) in start_periods:
            if (a, v, t, t2) in start_T:
                start_T[a, v, t, t2].setInitialValue(1)
            else:
                print('fail fixing task assginment in {}'.format((a, v, t, t2)))

        rut_data = self.set_remaining_usage_time('rut')
        for a, date_info in rut_data.items():
            for t, v in date_info.items():
                if (a, t) in rut:
                    rut[a, t].setInitialValue(v, check=False)

        return True

    def fix_variables(self, vars_names):
        if not vars_names:
            return
        vars_to_fix = [getattr(self, el) for el in vars_names]
        for _vars in vars_to_fix:
            for var in _vars.values():
                var.fixValue()

    def get_solution(self):

        l = self.domains
        if not len(l):
            raise ValueError('Model has not been solved yet. No domains are generated')

        start_M = self.start_M
        start_T = self.start_T
        rut = self.rut
        slack_kts_h = self.slack_kts_h
        slack_kts = self.slack_kts
        slack_ts = self.slack_ts

        first_period = self.instance.get_param('start')
        last_period = self.instance.get_param('end')

        _task = {}
        task_periods = tl.TupList(self.vars_to_tups(start_T))
        for a, v, t1, t2 in task_periods:
            for t in self.instance.get_periods_range(t1, t2):
                _task[a, t] = v

        # we store the start of maintenances and tasks in the same place
        _start = self.vars_to_tups(start_T).\
            take([0, 1, 2]).\
            to_dict(result_col=1, is_list=False)
        _start_M_aux = self.vars_to_tups(start_M)
        starts1_M = _start_M_aux.take([0, 1]).unique2()
        starts2_M = _start_M_aux.vfilter(lambda x: x[2] != last_period).take([0, 2]).unique2()
        _start_M = {k: self.M for k in starts1_M + starts2_M}
        _start.update(_start_M)

        fixed_maints_horizon = l['at_maint'].vfilter(lambda x: first_period <= x[1] <= last_period)
        _state = {tup: {self.M: 1} for tup in fixed_maints_horizon}
        _state.update({(a, t2): {self.M: 1} for (a, t) in _start_M for t2 in l['t2_at1'][(a, t)]})

        solution_data_pre = {
            'state_m': _state,
            'task': _task,
            'aux': {
                'start': _start,
                'rut': self.vars_to_dicts(rut),
                'rem': {},
                'slack_kts_h': self.vars_to_dicts(slack_kts_h),
                'slack_kts': self.vars_to_dicts(slack_kts),
                'slack_ts': self.vars_to_dicts(slack_ts)
            }
        }
        solution_data_pre = sd.SuperDict.from_dict(solution_data_pre)

        solution_data = {k: v.to_dictdict()
                         for k, v in solution_data_pre.items() if k != "aux"}
        solution_data['aux'] = {k: v.to_dictdict()
                                for k, v in solution_data_pre['aux'].items()}
        solution = sol.Solution(solution_data)
        return solution

    def get_variable_bounds(self):
        l = self.domains
        if not len(l):
            raise ValueError('No domains are generated: aborting.')

        param_data = self.instance.get_param()
        task_data = self.instance.get_tasks()

        # maximal bounds on continuous variables:
        max_used_time = param_data['max_used_time']  # mu. in hours of usage
        maint_duration = param_data['maint_duration']
        max_elapsed_time = param_data['max_elapsed_time'] + maint_duration  # me. in periods
        consumption = sd.SuperDict(task_data).get_property('consumption')  # rh. hours per period.
        num_resources = len(self.instance.get_resources())
        num_periods = len(self.instance.get_periods())
        max_num_maint = num_resources * num_periods / maint_duration
        ret_total = max_elapsed_time * num_resources
        rut_total = max_used_time * num_resources
        c_cand = self.instance.get_cluster_candidates()

        ub = {
            'ret': math.floor(max_elapsed_time),
            'rut': math.floor(max_used_time),
            'used_max': math.ceil(max(consumption.values())),
            'num_maint': math.floor(max_num_maint),
            'ret_end': math.ceil(ret_total),
            'rut_end': math.ceil(rut_total)
        }

        ub['slack_kts'] = {s: (p + 1) ** 2 for p, s in enumerate(l['slots'])}
        ub['slack_kts_h'] = {(k, s): (p + 1) * len(cands) * 100 if s != 2 else ub['rut'] * len(cands)
                             for p, s in enumerate(l['slots'])
                             for k, cands in c_cand.items()}
        ub['slack_ts'] = {s: (p + 1) * 2 for p, s in enumerate(l['slots'])}

        return ub

    @staticmethod
    def vars_to_tups(var):
        # because of rounding approximations; we need to check if its bigger than half:
        # we check if the var is None in case the solver doesn't return a value
        # (this does not happen very often)
        return tl.TupList(tup for tup in var if
                          var[tup].value() is not None and
                          var[tup].value() > 0.5)

    @staticmethod
    def vars_to_dicts(var):
        return sd.SuperDict.from_dict(var). \
            apply(lambda k, v: v.value()). \
            clean()

    def add_stochastic_cuts(self, model, start_M, _type, options):

        StochCuts = options.get('StochCuts', {})

        l = self.domains
        if not len(l):
            raise ValueError('No domains are generated: aborting.')

        # TODO: M is assumed
        duration = self.instance.get_param('maint_duration')
        _dist = self.instance.get_dist_periods
        resources = istats.get_resources_of_type(self.instance, _type=_type)
        size_res = len(resources)
        first_period, last_period = (self.instance.get_param(d) for d in ['start', 'end'])

        def get_min_max(variable, func=None):
            bounds = StochCuts.get('bounds', ['min', 'max'])
            _func = {'min': math.floor, 'max': math.ceil}
            contents = \
                sd.SuperDict({k: '{}_{}'.format(k, variable) for k in bounds}). \
                    vapply(lambda v: istats.get_bound_var(self.instance, v, _type))
            if func is not None:
                contents = contents.vapply(func)

            return contents.apply(lambda k, v: _func[k](v))

        filtered_att_no_last = \
            tl.TupList(l['att_maints_no_last']). \
                vfilter(function=lambda v: v[0] in resources)

        # we get for each combination: the distance between the second and last period
        dist_m2_end = \
            filtered_att_no_last. \
                to_dict(result_col=None). \
                apply(lambda k, v: _dist(k[2], last_period))
        # we get for each combination: the distance between the first and second manintenance
        # we need to add a distance when the second maintenance is at the end
        # because that is not really a maintenance.
        dist_m1_m2 = \
            tl.TupList(l['att_maints']). \
                vfilter(function=lambda v: v[0] in resources). \
                to_dict(result_col=None). \
                apply(lambda k, v: _dist(k[1], k[2]) - duration). \
                apply(lambda k, v: v + (k[2] == last_period))

        active_cuts = StochCuts.get('cuts', ['maints', 'mean_2maint', 'mean_dist'])
        str_cut = {}

        if 'maints' in active_cuts:
            str_cut['maints'] = get_min_max('maints', lambda v: v - size_res)
            str_cut['maints']['expression'] = pl.lpSum(start_M[tup] for tup in filtered_att_no_last)

        if 'mean_2maint' in active_cuts:
            str_cut['mean_2maint'] = get_min_max('mean_2maint', lambda v: size_res * v)
            str_cut['mean_2maint']['expression'] = pl.lpSum(start_M[tup] * dist for tup, dist in dist_m2_end.items())

        if 'mean_dist' in active_cuts:
            str_cut['mean_dist'] = get_min_max('mean_dist', lambda v: size_res * v)
            str_cut['mean_dist']['expression'] = pl.lpSum(start_M[tup] * dist for tup, dist in dist_m1_m2.items())

        for cut, info in str_cut.items():
            bound_min, bound_max = info.get('min'), info.get('max')
            if bound_min is not None:
                model += info['expression'] >= bound_min
            if bound_max is not None:
                model += info['expression'] <= bound_max

        return True

    def get_min_max_2M(self, options):

        param_data = self.instance.get_param()
        reduce_2M_window = options['reduce_2M_window']
        # TODO: M is assumed
        duration = param_data['maint_duration']
        types = self.instance.get_types()
        tolerance = reduce_2M_window.get('tolerance_mean')

        min_max_dist_types = \
            types.to_dict(None). \
                kapply(lambda k: istats.get_range_dist_2M(self.instance, k, tolerance))
        min_dist_types = min_max_dist_types.get_property('min')
        max_dist_types = min_max_dist_types.get_property('max')

        # TODO: M is assumed
        max_et = self.instance.get_param('max_elapsed_time')
        max_dist_types = \
            max_dist_types. \
                vapply(min, max_et). \
                vapply(lambda v: v + duration)

        min_dist_types = \
            min_dist_types. \
                vapply(max, 0). \
                vapply(lambda v: v + duration)

        max_elapsed_2M = \
            sd.SuperDict.from_dict(self.instance.get_resources('type')). \
                vapply(lambda v: sd.SuperDict(min=min_dist_types[v], max=max_dist_types[v])). \
                to_dictup().to_tuplist().to_dict(result_col=2, indices=[1, 0], is_list=False).to_dictdict()

        return max_elapsed_2M['min'], max_elapsed_2M['max']

    def get_domains_sets(self, options):
        states = [self.M]

        param_data = self.instance.get_param()

        # periods
        first_period, last_period = param_data['start'], param_data['end']
        periods = self.instance.get_periods_range(first_period, last_period)
        period_0 = self.instance.get_prev_period(param_data['start'])
        periods_0 = [period_0] + periods
        p_pos = {periods[pos]: pos for pos in range(len(periods))}
        previous = {period: periods_0[p_pos[period]] for period in periods}

        # tasks
        task_data = self.instance.get_tasks()
        tasks = list(task_data.keys())
        start_time = self.instance.get_tasks('start')
        end_time = self.instance.get_tasks('end')
        min_assign = self.instance.get_tasks('min_assign')
        candidates = self.instance.get_task_candidates()

        # resources
        resources_data = self.instance.get_resources()
        resources = list(resources_data.keys())
        # TODO: M is assumed
        duration = param_data['maint_duration']
        max_elapsed = param_data['max_elapsed_time'] + duration
        min_elapsed = max_elapsed - param_data['elapsed_time_size']

        # second maintenance can have a more limited size of calendar.
        # and depends on the aircraft
        min_elapsed_2M = {r: min_elapsed for r in resources}
        max_elapsed_2M = {r: max_elapsed for r in resources}

        ret_init = self.instance.get_initial_state("elapsed")
        ret_init_adjusted = {k: v - max_elapsed + min_elapsed for k, v in ret_init.items()}
        kt = sd.SuperDict(self.instance.get_cluster_constraints()['num']).keys_l()

        """
        Indentation means "includes the following:".
        The elements represent a given combination resource-period.
        at0: all, including the previous period.
            at: all.                                                    => 'used'
                at_mission: a mission is assigned (fixed)               => 'assign'
                    at_mission_m: specific mission is assigned (fixed)  => 'assign'
                at_free: nothing is fixed                               => 'assign' and 'state'
                    at_free_start: can start a maintenance              => 'start_M'
                at_maint: maintenance is assigned (fixed)               => 'state' 
                    at_start: start of maintenance is assigned (fixed). => 'start_M'
        These values should not be assumed to all fall inside the planning horizon.
        There is also the initial values before it.
        """

        at = tl.TupList((a, t) for a in resources for t in periods)
        at0 = tl.TupList([(a, period_0) for a in resources] + at)
        at_mission_m = self.instance.get_fixed_tasks()
        at_mission_m_horizon = at_mission_m.vfilter(lambda x: first_period <= x[2] <= last_period)
        at_mission = tl.TupList((a, t) for (a, s, t) in at_mission_m_horizon)  # Fixed mission assignments.
        at_start = []  # Fixed maintenances starts
        at_maint = self.instance.get_fixed_maintenances()  # Fixed maintenance assignments.
        at_free = tl.TupList((a, t) for (a, t) in at if (a, t) not in list(at_maint + at_mission))

        # we update the possibilities of starting a maintenance
        # depending on the rule of minimal time between maintenance
        # an the initial "ret" state of the resource
        at_free_start = [(a, t) for (a, t) in at_free]
        # at_free_start = [(a, t) for (a, t) in at_free if periods_pos[t] % 3 == 0]
        at_m_ini = [(a, t) for (a, t) in at_free_start
                    if p_pos[t] < ret_init_adjusted[a]
                    ]
        at_m_ini_s = set(at_m_ini)

        at_free_start = tl.TupList(i for i in at_free_start if i not in at_m_ini_s)

        vt = tl.TupList((v, t) for v in tasks for t in periods if start_time[v] <= t <= end_time[v])
        t_v = vt.to_dict(result_col=1)
        # t_v = {k: sorted(v) for k, v in t_v.items()}
        # last_v = {k: v[-1] for k, v in t_v.items()}
        avt = tl.TupList([(a, v, t) for a in resources for (v, t) in vt
                          if a in candidates[v]
                          if (a, t) in at_free] + \
                         at_mission_m_horizon)
        ast = tl.TupList((a, s, t) for (a, t) in list(at_free + at_maint) for s in states)
        att = tl.TupList([(a, t1, t2) for (a, t1) in list(at_start + at_free_start) for t2 in periods if
                          p_pos[t1] <= p_pos[t2] < p_pos[t1] + duration])
        # when I could have started maintenance (t2s) to be still in maintenance in period t1
        t2_at1 = att.to_dict(result_col=2, is_list=True)
        # start-assignment options for task assignments.
        # We assume assignments can happen anywhere to really apply the constraint correctly.
        avtt = tl.TupList([(a, v, t1, t2) for (a, v, t1) in avt for t2 in periods if
                           p_pos[t1] <= p_pos[t2] < p_pos[t1] + min_assign[v]
                           ])
        # Start-stop options for task assignments.
        avtt2 = tl.TupList([(a, v, t1, t2) for (a, v, t1) in avt for t2 in t_v[v] if
                            (p_pos[t2] >= p_pos[t1] + min_assign[v] - 1) or
                            (p_pos[t2] >= p_pos[t1] and t2 == last_period)
                            ])
        # For Start-stop options, during fixed periods, we do not care of the minimum time assignment.
        avtt2_fixed = tl.TupList([(a, v, t1, t2) for (a, v, t1) in avt for t2 in t_v[v] if
                                  (p_pos[t2] >= p_pos[t1]) and
                                  ((a, v, t1) in at_mission_m_horizon or
                                   (a, v, t2) in at_mission_m_horizon or
                                   (a, v, previous[t1]) in at_mission_m)
                                  ])
        avtt2.extend(avtt2_fixed)
        # we had a repetition problem:
        avtt2 = avtt2.unique2()

        att_m = tl.TupList([(a, t1, t2) for (a, t1) in at_free_start for t2 in periods
                            if p_pos[t1] < p_pos[t2] < p_pos[t1] + min_elapsed
                            ])

        # first maintenance starts possibilities because of initial state of aircraft
        at_M_ini = tl.TupList([(a, t) for (a, t) in at_free_start
                               if ret_init_adjusted[a] <= p_pos[t] <= ret_init[a]
                               ])

        # att_maints is the domain for the maintenance m_itt variable
        # we want all t1, t2 combinations such as t1 and t2 make possible cycle combinations.
        # without using the last period as a start of a new cycle (as a convention)
        # since we are only assuming max 1 assignment, we need to take out the possibilities that leave
        # more than max_elapsed after it
        # only allow maintenance starts that follow the initial state (at_M_ini)
        att_maints_no_last = tl.TupList((a, t1, t2) for (a, t1) in at_M_ini for t2 in periods
                                        if (p_pos[t1] + min_elapsed_2M[a] <= p_pos[t2] < p_pos[t1] + max_elapsed_2M[a])
                                        and len(periods) <= p_pos[t2] + max_elapsed
                                        and t2 < last_period
                                        )

        # also, we want to permit incomplete cycles that finish in the last period.
        # the additional possibilities are very similar to the previous ones
        # but with the last_period instead of t2
        # and they do not constraint the min distance between maintenances
        _t2 = last_period
        att_maints_last = ((a, t1, _t2) for (a, t1) in at_M_ini if
                           p_pos[t1] + duration <= p_pos[_t2] < p_pos[t1] + max_elapsed_2M[a])
        att_maints_last = tl.TupList(att_maints_last)

        # as an auxiliary second step, we want to apply the filters based on the stochastic cuts.
        # we want to keep a certain number of non-conforming cycle combinations.
        # the number is a percentage of filtered value cuts.
        reduce_2M_window = options.get('reduce_2M_window', {})
        if reduce_2M_window.get('active', False):
            seed = options.get('seed')
            if not seed:
                seed = math.ceil(rn.random() * 100000)
                options['seed'] = seed
            rn.seed(seed)

            min_elapsed_2M_cut, max_elapsed_2M_cut = self.get_min_max_2M(options)
            percent_add = reduce_2M_window.get('percent_add', 0)

            def _filter_funct(args):
                a, t1, t2 = args
                return (p_pos[t1] + min_elapsed_2M_cut[a] <= p_pos[t2] < p_pos[t1] + max_elapsed_2M_cut[a])

            def limit_partially(tuplist):
                tuplist_filtered = tuplist.vfilter(_filter_funct)
                rejected_combos = set(tuplist) - set(tuplist_filtered)
                size_percent = min(round(len(tuplist_filtered) * percent_add), len(rejected_combos))
                rejected_combos_accepted = rn.sample(list(rejected_combos), k=size_percent)
                return tuplist_filtered + tl.TupList(rejected_combos_accepted)

            # we reduce the first list.
            att_maints_no_last = limit_partially(att_maints_no_last)
            # we reduce the second one.
            att_maints_last = limit_partially(att_maints_last)

        # Finally, cuts or not: we sum both sets into a single set of patterns.
        att_maints = att_maints_no_last + att_maints_last
        att_maints = tl.TupList(att_maints)

        # at_cycles are three times for each combination of maints possibility
        # to represent the "before, during and after" of the maintenance cycles
        cycles = [str(n) for n in range(3)]
        att_cycles = tl.TupList((a, n) for a in resources for n in cycles)

        # these are the periods where we know we have to do a second maintenance, given we did a maintenance in t.
        # (OBSOLETE)
        att_M = att_maints.vfilter(lambda x: p_pos[x[1]] + max_elapsed < len(periods))
        # this is the TTT_t set.
        # periods that are maintenance periods because of having assign a maintenance
        attt_maints = tl.TupList((a, t1, t2, t) for a, t1, t2 in att_maints for t in t2_at1.get((a, t1), []))
        attt_maints += tl.TupList((a, t1, t2, t) for a, t1, t2 in att_maints for t in t2_at1.get((a, t2), [])
                                  if t2 < last_period)
        attt_maints = attt_maints.unique2()

        # all possible mission assignments (a, v, t1, t2) with their respective specific affected periods t
        avtt2t = tl.TupList(
            [(a, v, t1, t2, t) for (a, v, t1, t2) in avtt2 for t in self.instance.get_periods_range(t1, t2)]
        )
        tt2_avt = avtt2t.to_dict(result_col=[2, 3]).fill_with_default(avt, [])

        a_t = at.to_dict(result_col=0, is_list=True)
        a_vt = avt.to_dict(result_col=0, is_list=True)
        v_at = avt.to_dict(result_col=1, is_list=True).fill_with_default(at, [])
        at1_t2 = att.to_dict(result_col=[0, 1], is_list=True)
        t1_at2 = att.to_dict(result_col=1, is_list=True).fill_with_default(at, [])
        t2_avt1 = avtt.to_dict(result_col=3, is_list=True)
        t2_avt1 = avtt.to_dict(result_col=3, is_list=True)
        t1_avt2 = avtt.to_dict(result_col=2, is_list=True)
        t_at_M = att_M.to_dict(result_col=2, is_list=True)
        t_a_M_ini = at_M_ini.to_dict(result_col=1, is_list=True)
        tt_maints_at = attt_maints.to_dict(result_col=[1, 2], is_list=True)
        att_maints_t = attt_maints.to_dict(result_col=[0, 1, 2], is_list=True)
        tt_maints_a = att_maints.to_dict(result_col=[1, 2], is_list=True)

        vtt2_a = avtt2.to_dict(result_col=[1, 2, 3]).apply(lambda _, v: tl.TupList(v))
        vtt2_a_after_t = {(a, t): vtt2_a[a].vfilter(lambda x: x[1] >= t) for a in resources for t in periods}
        vtt2_a_before_t = {(a, t): vtt2_a[a].vfilter(lambda x: x[2] <= t) for a in resources for t in periods}
        vtt2_between_att = {(a, t1, t2): vtt2_a_after_t[a, t1].intersect(vtt2_a_before_t[a, t2])
                            for a in resources for pos1, t1 in enumerate(periods) for t2 in periods[pos1:]}

        slots = [str(s) for s in range(3)]
        k = tl.TupList(kt).filter(0).unique2()
        kts = [(k, t, s) for k, t in kt for s in slots]
        ts = [(t, s) for t in periods for s in slots]

        return {
            'periods': periods
            , 'period_0': period_0
            , 'periods_0': periods_0
            , 'periods_pos': p_pos
            , 'previous': previous
            , 'tasks': tasks
            , 'candidates': candidates
            , 'resources': resources
            , 'states': states
            , 'vt': vt
            , 'avt': avt
            , 'at': at
            , 'at_mission_m': at_mission_m
            , 'ast': ast
            , 'at_start_maint': tl.TupList(at_start + at_free_start)
            , 'at0': at0
            , 'att': att
            , 'a_t': a_t
            , 'a_vt': a_vt
            , 'v_at': v_at
            , 't1_at2': t1_at2
            , 'at1_t2': at1_t2
            , 't2_at1': t2_at1
            , 'at_avail': tl.TupList(at_free + at_mission)
            , 't2_avt1': t2_avt1
            , 't1_avt2': t1_avt2
            , 'avtt': avtt
            , 'avtt2': avtt2
            , 'tt2_avt': tt2_avt
            , 'att_m': att_m
            , 't_at_M': t_at_M
            , 'att_M': att_M
            , 'at_m_ini': at_m_ini
            , 't_a_M_ini': t_a_M_ini
            , 'kt': kt
            , 'slots': slots
            , 'k': k
            , 'kts': kts
            , 'ts': ts
            , 'at_maint': at_maint
            , 'att_maints': att_maints
            , 'att_cycles': att_cycles
            , 'cycles': cycles
            , 'att_maints_no_last': att_maints_no_last
            , 'tt_maints_at': tt_maints_at
            , 'att_maints_t': att_maints_t
            , 'tt_maints_a': tt_maints_a
            , 'vtt2_between_att': vtt2_between_att
        }

    def get_valid_cuts(self, model, start_M, start_T):
        l = self.domains
        if not len(l):
            raise ValueError('No domains are generated: aborting.')
        resources = l['resources']
        periods = l['periods']
        att_maints_no_last = l['att_maints_no_last']
        first, last = self.instance.get_start_end()
        U_min = self.instance.get_param('min_usage_period')
        RftInit = self.instance.get_resources('initial_used')

        # TODO: M is assumed
        M = self.instance.get_param('maint_duration')
        maxH = self.instance.get_param('max_used_time')
        _shift = self.instance.shift_period
        _prev = self.instance.get_prev_period

        # 6.1 Accumulated checks per aircraft and period
        TM1_set = att_maints_no_last.to_dict(result_col=[1], indices=[0]).vapply(set)
        TM1 = sd.SuperDict(min=TM1_set.vapply(min),
                           max=TM1_set.vapply(max))
        if U_min:
            _TM1_max = RftInit.vapply(lambda v: _shift(first, math.floor(v / U_min)))
            TM1['max'] = TM1['max'].sapply(min, _TM1_max)
        TM2_set = att_maints_no_last.to_dict(result_col=[2], indices=[0]).vapply(set)
        TM2 = sd.SuperDict(min=TM2_set.vapply(min),
                           max=TM2_set.vapply(max))
        if U_min:
            _TM2_max = TM1['max'].vapply(_shift, M + math.floor(maxH / U_min))
            TM2['max'] = TM2['max'].sapply(min, _TM2_max)

        for r in resources:
            assert TM1['min'][r] <= TM1['max'][r]
            assert TM2['min'][r] <= TM2['max'][r]

        def get_M_Acc_S(r, t, sense='min'):
            if sense == 'min':
                ref1 = TM1['max']
                ref2 = TM2['max']
            else:  # sense=='max'
                ref1 = TM1['min'].vapply(_prev)
                ref2 = TM2['min'].vapply(_prev)
            if t <= ref1[r]:
                return 0
            elif ref1[r] < t <= ref2[r]:
                return 1
            elif last > t > ref2[r]:
                return 2
            else:
                return None

        M_Acc_S = {
            sense: {
                (r, t): get_M_Acc_S(r, t, sense=sense)
                for r in resources for t in periods[:-1]}
            for sense in ['min', 'max']
        }
        M_Acc_S = sd.SuperDict.from_dict(M_Acc_S)
        backM_periods = lambda k: (k[0], _shift(k[1], -M))
        M_Acc_F = M_Acc_S.vapply(lambda v: v.kapply(lambda k: v.get(backM_periods(k))))

        # 6.2 Accumulated checks at the end of the horizon per aircraft
        I_1M = M_Acc_S['max'].to_dictdict().clean(func=lambda v: v[_prev(last)] == 1).keys_l()
        I_2M = M_Acc_S['min'].to_dictdict().clean(func=lambda v: v[_prev(last)] == 2).keys_l()

        for r in I_1M:
            for (t1, t2) in l['tt_maints_a'][r]:
                _tup = r, t1, t2
                if t2 != last and _tup in start_M:
                    start_M[_tup].setInitialValue(0)
                    start_M[_tup].fixValue()
                    # model += start_M[_tup] == 0
                # model += pl.lpSum(start_M[r, t, last] for t in l['t_a_M_ini'][r]) == 1
        for r in I_2M:
            for t1 in l['t_a_M_ini'][r]:
                _tup = r, t1, last
                if _tup in start_M:
                    start_M[_tup].setInitialValue(0)
                    start_M[_tup].fixValue()
                    # model += start_M[_tup] == 0
                # start_M.pop((r, t, last))

        # 6.3 Mission assignments at the start of the horizon for each aircraft
        tup_r = {r: (r, first, _shift(TM1['min'][r], -1)) for r in resources}
        _range = self.instance.get_periods_range
        _dist = lambda *x: self.instance.get_dist_periods(*x) + 1
        init_assigns = {r: l['vtt2_between_att'].get(_tup) for r, _tup in tup_r.items()}
        init_assigns = sd.SuperDict(init_assigns).clean(func=lambda v: v)
        consum = self.instance.get_tasks('consumption')
        res_type = self.instance.get_resources('type')

        acc_consum = \
            init_assigns. \
                list_reverse(). \
                kapply(lambda k: (consum[k[0]] - U_min) * _dist(k[1], k[2]))
        # init_assigns.vapply()
        acc_min_usage = {r: U_min * (_dist(t1, t2)) for r, t1, t2 in tup_r.values()}

        for r, _tuplist in init_assigns.items():
            # we make a cut for all assignments in general.
            model += pl.lpSum(acc_consum[vtt] * start_T[(r, *vtt)] for vtt in _tuplist) \
                     + acc_min_usage[r] <= RftInit[r]

            for vtt in _tuplist:
                # we take out the worst assignments
                if RftInit[r] < acc_consum[vtt] + acc_min_usage[r]:
                    start_T[(r, *vtt)].setInitialValue(0)
                    start_T[(r, *vtt)].fixValue()

        # 6.4 Accumulated checks per aircraft type and period
        def agg_by_type(M_Acc):
            YM1_Acc = sd.SuperDict()
            for bound, info in M_Acc.items():
                for (res, period), value in info.items():
                    if value is None:
                        continue
                    new_key = res_type[res], period
                    prev_val = YM1_Acc.get_m(bound, new_key, default=0)
                    YM1_Acc.set_m(bound, new_key, value=prev_val + value)
            return YM1_Acc

        YM_Acc_S = agg_by_type(M_Acc_S)
        YM1_Acc_F = agg_by_type(M_Acc_F)

        task_units = self.instance.get_tasks('num_resource')
        v_first = l['vt'].to_dict(1).vapply(min)
        v_last = l['vt'].to_dict(1).vapply(max)
        JR_Acc = \
            l['vt'].to_dict(None). \
                vapply(lambda v: _dist(v_first[v[0]], v[1]) * task_units[v[0]])
        # I need to fill the following periods with the last one's value:
        # for it to be a try accumulate
        for t in periods:
            for v in v_last:
                if t > v_last[v]:
                    JR_Acc[v, t] = JR_Acc[v, v_last[v]]
                elif t < v_first[v]:
                    JR_Acc[v, t] = 0

        type_res = res_type.vapply(lambda v: [v]).list_reverse()
        type_num = type_res.vapply(len)
        type_tasks = \
            self.instance.get_tasks('type_resource'). \
                vapply(lambda v: [v]). \
                list_reverse()
        IR_Acc = \
            {(tt, t): num * _dist(first, t) -
                      sum(JR_Acc[v, t] for v in type_tasks[tt])
             for tt, num in type_num.items() for t in periods}
        IR_Acc = sd.SuperDict.from_dict(IR_Acc)
        YM2_Acc_F = sd.SuperDict()
        YM2_Acc_F['max'] = IR_Acc.vapply(lambda v: math.floor(v / M))

        YH_Acc = \
            IR_Acc.kapply(lambda tup:
                          sum(consum[v] * JR_Acc[v, tup[1]]
                              for v in type_tasks[tup[0]])
                          )
        YM2_Acc_F['min'] = YH_Acc.apply(
            lambda k, v: (v - sum(RftInit[a] for a in type_res[k[0]])) / maxH
        ).vapply(math.ceil).vapply(max, 0)

        YM_Acc_F = sd.SuperDict()
        YM_Acc_F['min'] = YM1_Acc_F['min'].sapply(max, YM2_Acc_F['min'])
        YM_Acc_F['max'] = YM1_Acc_F['max'].sapply(min, YM2_Acc_F['max'])
        t1_t1 = l['att_maints'].filter([1, 2]).unique2()

        def _QM(tup):
            t, t1, t2 = tup
            t1_M = _shift(t1, M)
            t2_M = _shift(t2, M)
            if t < t1_M:
                return 0
            elif t1_M <= t < t2_M:
                return 1
            elif t >= t2_M:
                return 2

        QM = {(t, t1, t2): 1 for t1, t2 in t1_t1 for t in periods}
        QM = sd.SuperDict.from_dict(QM).kapply(_QM)

        for y, t in YM_Acc_F['min']:
            _expresion = pl.lpSum(start_M[r, t1, t2] * QM[t, t1, t2]
                                  for r in type_res[y]
                                  for t1, t2 in l['tt_maints_a'][r]
                                  if QM[t, t1, t2] > 0)
            model += _expresion >= YM_Acc_F['min'][y, t]
            model += _expresion <= YM_Acc_F['max'][y, t]

        # 6.5 Accumulated checks per period
        _YM_Acc_F = YM_Acc_F.to_dictdict().to_dictup().to_tuplist().to_dict(3, False, [0, 2, 1]).to_dictdict()
        TM1_Acc_F = {(b, t): sum(info2.values())
                     for b, info in _YM_Acc_F.items()
                     for t, info2 in info.items()}
        TM1_Acc_F = sd.SuperDict.from_dict(TM1_Acc_F).to_dictdict()
        # TODO: M is assumed
        capacity = self.instance.get_param('maint_capacity')
        TM2_Acc_F_max = {t: math.floor(_dist(first, t) / M) * capacity for t in periods}
        TM_Acc_F = TM1_Acc_F
        TM_Acc_F['max'] = TM1_Acc_F['max'].sapply(min, TM2_Acc_F_max)

        for t in TM_Acc_F['min']:
            _expresion = pl.lpSum(start_M[r, t1, t2] * QM[t, t1, t2]
                                  for r in resources
                                  for t1, t2 in l['tt_maints_a'][r]
                                  if QM[t, t1, t2] > 0)
            model += _expresion >= TM_Acc_F['min'][t]
            model += _expresion <= TM_Acc_F['max'][t]
        return True


if __name__ == "__main__":
    import data.simulation as sim
    import package.params as params
    import package.instance as inst

    options = params.OPTIONS
    model_data = sim.create_dataset(options)
    instance = inst.Instance(model_data)
    self = Model(instance)
    l = self.domains = self.get_domains_sets(options)

    pass
    pass
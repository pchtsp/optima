import math
import pulp as pl
import package.auxiliar as aux
import package.config as conf
import package.solution as sol
import package.experiment as exp
import package.tuplist as tl
import package.superdict as sd
import random as rn
import numpy as np


######################################################
# @profile

class Model(exp.Experiment):

    def __init__(self, instance, solution=None):

        if solution is None:
            solution = sol.Solution({'state': {}, 'task': {}})
        super().__init__(instance, solution)

    def solve(self, options=None):
        l = self.get_domains_sets()
        ub = self.get_bounds()
        first_period = self.instance.get_param('start')
        last_period = self.instance.get_param('end')
        consumption = self.instance.get_tasks('consumption')
        requirement = self.instance.get_tasks('num_resource')
        rut_init = self.instance.get_initial_state("used")
        num_resource_maint = aux.fill_dict_with_default(self.instance.get_total_fixed_maintenances(), l['periods'])
        maint_capacity = self.instance.get_param('maint_capacity')
        max_usage = self.instance.get_param('max_used_time')
        min_usage = self.instance.get_param('min_usage_period')
        duration = self.instance.get_param('maint_duration')
        cluster_data = self.instance.get_cluster_constraints()
        c_candidates = self.instance.get_cluster_candidates()

        # shortcut functions
        def dist(t1, t2):
            # t_2 - t_1 + 1
            return self.instance.get_dist_periods(t1, t2) + 1

        def acc_dist(t_1, t_2, tp=None):
            if tp is None:
                tp = t_2
            # sum_{t = t_1 -1}^{t_2} tp - t
            # return (t_2 - t_1 + 1) * (2 * tp - t_1 - t_2 + 2) / 2
            return dist(t_1, t_2) * (dist(t_1, tp)  + dist(t_2, tp))/2

        shift = self.instance.shift_period
        prev = self.instance.get_prev_period

        # In order to break some symmetries, we're gonna give a
        # (different) price for each assignment:
        price_assign = {(a, v): 0 for v in l['tasks'] for a in l['candidates'][v]}
        if options.get('noise_assignment', True):
            price_assign = {(a, v): rn.random() for v in l['tasks'] for a in l['candidates'][v]}

        price_rut_end = options.get('price_rut_end', 1)

        # Sometimes we want to force variables to be integer.
        var_type = pl.LpContinuous
        if options.get('integer', False):
            var_type = pl.LpInteger

        # VARIABLES:
        # binary:
        # mission assignment
        start_T = pl.LpVariable.dicts(name="start_T", indexs=l['avtt2'], lowBound=0, upBound=1, cat=pl.LpInteger)
        # maintenance cycle
        start_M = pl.LpVariable.dicts(name="start_M", indexs=l['att_maints'], lowBound=0, upBound=1, cat=pl.LpInteger)

        # numeric:
        # avg remaining usage hours in cycle
        rem_M = pl.LpVariable.dicts(name="rem_M", indexs=l['att_cycles'], lowBound=0, upBound=ub['rut'], cat=var_type)

        # slack variables:
        slack_vt = {tup: 0 for tup in l['vt']}
        slack_at = {tup: 0 for tup in l['at']}
        slack_kt_hours = {tup: 0 for tup in l['kt']}
        slack_kt_num = {tup: 0 for tup in l['kt']}
        slack_t = {tup: 0 for tup in l['periods']}

        slack_p = options.get('slack_vars')
        if slack_p == 'Yes':
            slack_vt = pl.LpVariable.dicts(name="slack_vt", lowBound=0, indexs=l['vt'], cat=var_type)
            slack_at = pl.LpVariable.dicts(name="slack_at", lowBound=0, indexs=l['at'], cat=var_type)
            slack_kt_hours = pl.LpVariable.dicts(name="slack_kt", lowBound=0, indexs=l['kt'], cat=var_type)
        elif slack_p is int:
            # first X months only
            first_months = self.instance.get_next_periods(first_period, slack_p)
            _vt = [(v, t) for v, t in l['vt'] if t in first_months]
            _kt = [(k, t) for k, t in l['kt'] if t in first_months]
            slack_vt = pl.LpVariable.dicts(name="slack_vt", lowBound=0, indexs=_vt, cat=var_type)
            slack_t = pl.LpVariable.dicts(name="slack_t", lowBound=0, indexs=first_months, cat=var_type)
            slack_kt_num = pl.LpVariable.dicts(name="slack_kt", lowBound=0, indexs=_kt, cat=var_type)

        # MODEL
        model = pl.LpProblem("MFMP_v0002", pl.LpMinimize)

        # OBJECTIVE:
        # maints
        model += pl.lpSum(start_M[a, t1, t2]
                          for a, t1, t2 in l['att_maints_no_last']) * max_usage \
                 + pl.lpSum(rem_M[a, pos] for a, pos in l['att_cycles'])/10 \
                 + 1 * pl.lpSum(assign_st * price_assign[a, v]
                                for (a, v, t, t2), assign_st in start_T.items()) + \
                 1000000 * pl.lpSum(slack_vt.values()) + \
                 1000 * pl.lpSum(slack_at.values()) + \
                 10000 * pl.lpSum(slack_kt_num.values()) + \
                 1000 * pl.lpSum(slack_kt_hours.values()) + \
                 1000000 * pl.lpSum(slack_t.values())

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
                pl.lpSum(start_M[a, t1, t2] for a in c_candidates[k]
                         for (t1, t2) in l['tt_maints_at'].get((a, t), [])
                         ) <= num + slack_kt_num.get((k, t), 0)

        # # ##################################
        # # Usage time
        # # ##################################
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

                # Count the mean remaining hours at each cycle
                _vars_tup = [
                    (start_T[a, v, t11, t22],
                     acc_dist(t11, t22, p2) * (consumption[v] - min_usage)
                     )
                    for v, t11, t22 in _vtt2]
                _vars_tup.append((start_M[a, t1, t2], d_p1_p2 * limit))
                _vars_tup.append((rem_M[a, pos], -d_p1_p2))
                _constant = min_usage * acc_dist(p1, p2, p2) - d_p1_p2 *  limit
                model += pl.LpAffineExpression(_vars_tup, constant=_constant) <= 0

        # ##################################
        # Maintenances
        # ##################################

         # NOTE: we are now assuming one maintenance assignment (be it simple or double)

        # For small horizons, we want to add a an upper limit on maintenances
        # in case the previous constraint is not active for that resource
        for a, tt_list in l['tt_maints_a'].items():
            # if not l['t_a_M_ini'].get(a, []):
            model += pl.lpSum(start_M[a, t1, t2] for t1, t2 in tt_list) == 1

        # max number of maintenances:
        for t in l['periods']:
            at1t1_list = l['att_maints_t'].get(t, [])
            if not len(at1t1_list):
                continue
            model += pl.lpSum(start_M[a, t1, t2] for (a, t1, t2) in at1t1_list) + \
                     num_resource_maint[t] <= maint_capacity + slack_t.get(t, 0)

        # ##################################
        # SOLVING
        # ##################################

        # SOLVING
        config = conf.Config(options)
        result = config.solve_model(model)

        if result != 1:
            print("Model resulted in non-feasible status: {}".format(result))
            return None
        print('model solved correctly')

        _task = {}
        task_periods = tl.TupList(self.vars_to_tups(start_T))
        for a, v, t1, t2 in task_periods:
            for t in self.instance.get_periods_range(t1, t2):
                _task[a, t] = v

        # we store the start of maintenances and tasks in the same place
        _start = self.vars_to_tups(start_T).filter([0, 1, 2]).to_dict(result_col=1, is_list=False)
        _start_M_aux = self.vars_to_tups(start_M)
        starts1_M = _start_M_aux.filter([0, 1]).unique2()
        starts2_M = _start_M_aux.filter_list_f(lambda x: x[2] != last_period).filter([0, 2]).unique2()
        _start_M = {k: 'M' for k in starts1_M + starts2_M}
        _start.update(_start_M)

        _rut = {}
        _rem = sd.SuperDict.from_dict(rem_M).apply(lambda k, v: v.value())
        # _rem = {}
        fixed_maints_horizon = l['at_maint'].filter_list_f(lambda x: first_period <= x[1] <= last_period)
        _state = {tup: 'M' for tup in fixed_maints_horizon}
        _state.update({(a, t2): 'M' for (a, t) in _start_M for t2 in l['t2_at1'][(a, t)]})

        solution_data_pre = {
            'state': _state,
            'task': _task,
            'aux': {
                'start': _start,
                'rut': _rut,
                'rem': _rem,
            }
        }
        solution_data_pre = sd.SuperDict.from_dict(solution_data_pre)

        solution_data = {k: v.to_dictdict()
                         for k, v in solution_data_pre.items() if k != "aux"}
        solution_data['aux'] = {k: v.to_dictdict()
                                for k, v in solution_data_pre['aux'].items()}
        solution = sol.Solution(solution_data)
        self.solution = solution
        return solution


    def get_bounds(self):
        param_data = self.instance.get_param()
        task_data = self.instance.get_tasks()

        # maximal bounds on continuous variables:
        max_used_time = param_data['max_used_time']  # mu. in hours of usage
        maint_duration = param_data['maint_duration']
        max_elapsed_time = param_data['max_elapsed_time'] + maint_duration # me. in periods
        consumption = aux.get_property_from_dic(task_data, 'consumption')  # rh. hours per period.
        num_resources = len(self.instance.get_resources())
        num_periods = len(self.instance.get_periods())
        max_num_maint = num_resources*num_periods/maint_duration
        ret_total = max_elapsed_time*num_resources
        rut_total = max_used_time*num_resources

        return {
            'ret': math.floor(max_elapsed_time),
            'rut': math.floor(max_used_time),
            'used_max': math.ceil(max(consumption.values())),
            'num_maint': math.floor(max_num_maint),
            'ret_end': math.ceil(ret_total),
            'rut_end': math.ceil(rut_total)
        }

    @staticmethod
    def vars_to_tups(var):
        # because of rounding approximations; we need to check if its bigger than half:
        # we check if the var is None in case the solver doesn't return a value
            # (this does not happen very often)
        return tl.TupList(tup for tup in var if
                          var[tup].value() is not None and
                          var[tup].value() > 0.5)

    def get_domains_sets(self):
        states = ['M']

        param_data = self.instance.get_param()

        # periods
        first_period, last_period = param_data['start'], param_data['end']
        periods = self.instance.get_periods_range(first_period, last_period)
        period_0 = self.instance.get_prev_period(param_data['start'])
        periods_0 = [period_0] + periods
        periods_pos = {periods[pos]: pos for pos in range(len(periods))}
        previous = {period: periods_0[periods_pos[period]] for period in periods}

        # tasks
        task_data = self.instance.get_tasks()
        tasks = list(task_data.keys())
        start_time = self.instance.get_tasks('start')
        end_time = self.instance.get_tasks('end')
        min_assign = self.instance.get_tasks('min_assign')
        candidates = self.instance.get_task_candidates()
        # candidates = aux.get_property_from_dic(task_data, 'candidates')

        # resources
        resources_data = self.instance.get_resources()
        resources = list(resources_data.keys())
        duration = param_data['maint_duration']
        max_elapsed = param_data['max_elapsed_time'] + duration
        min_elapsed = param_data['min_elapsed_time'] + duration
        # previous_states = \
        #     sd.SuperDict.from_dict(aux.get_property_from_dic(resources_data, 'states')).\
        #         to_dictup().to_tuplist().tup_to_start_finish()
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
        at_mission_m_horizon = at_mission_m.filter_list_f(lambda x: first_period <= x[2] <= last_period )
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
                    if periods_pos[t] < ret_init_adjusted[a]
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
               periods_pos[t1] <= periods_pos[t2] < periods_pos[t1] + duration])
        # when I could have started maintenance (t2s) to be still in maintenance in period t1
        t2_at1 = att.to_dict(result_col=2, is_list=True)
        # start-assignment options for task assignments.
        avtt = tl.TupList([(a, v, t1, t2) for (a, v, t1) in avt for t2 in t_v[v] if
                periods_pos[t1] <= periods_pos[t2] < periods_pos[t1] + min_assign[v]
                ])
        # Start-stop options for task assignments.
        avtt2 = tl.TupList([(a, v, t1, t2) for (a, v, t1) in avt for t2 in t_v[v] if
                            (periods_pos[t2] >= periods_pos[t1] + min_assign[v] - 1) or
                            (periods_pos[t2] >= periods_pos[t1] and t2 == last_period)
                ])
        # For Start-stop options, during fixed periods, we do not care of the minimum time assignment.
        avtt2_fixed = tl.TupList([(a, v, t1, t2) for (a, v, t1) in avt for t2 in t_v[v] if
                            (periods_pos[t2] >= periods_pos[t1]) and
                            ((a, v, t1) in at_mission_m_horizon or
                             (a, v, t2) in at_mission_m_horizon)
                ])
        avtt2.extend(avtt2_fixed)
        att_m = tl.TupList([(a, t1, t2) for (a, t1) in at_free_start for t2 in periods
                 if periods_pos[t1] < periods_pos[t2] < periods_pos[t1] + min_elapsed
                 ])

        # first maintenance starts possibilities because of initial state of aircraft
        at_M_ini = tl.TupList([(a, t) for (a, t) in at_free_start
                    if ret_init[a] <= len(periods)
                    if ret_init_adjusted[a] <= periods_pos[t] <= ret_init[a]
                    ])

        # this is the domain for the maintenance m_itt variable
        # we want all t1, t2 combinations such as t1 and t2 make possible cycle combinations.
        # without using the last period as a start of a new cycle (as a convention)
        # since we are only assuming max 1 assignment, we need to take out the possibilities that leave
        # more than max_elapsed after it
        # only allow maintenance starts that follow the initial state (at_M_ini)
        att_maints_no_last = tl.TupList((a, t1, t2) for (a, t1) in at_M_ini for t2 in periods
                                if (periods_pos[t1] + min_elapsed <= periods_pos[t2] < periods_pos[t1] + max_elapsed)
                                and len(periods) - max_elapsed <= periods_pos[t2]
                                and t2 < last_period
                                )
        # also, we want to permit incomplete cycles that finish in the last period.
        # the additional possibilities are very similar to the previous ones
        # but with the last_period instead of t2
        _t2 = last_period
        att_maints = att_maints_no_last + \
                     tl.TupList((a, t1, _t2) for (a, t1) in at_M_ini if
                                periods_pos[t1] + min_elapsed <= periods_pos[_t2] < periods_pos[t1] + max_elapsed
                                )
        att_maints = tl.TupList(att_maints)

        # at_cycles are three times for each combination of maints possibility
        # to represent the "before, during and after" of the maintenance cycles
        att_cycles = tl.TupList((a, n) for a in resources for n in range(3))

        att_M = att_maints.filter_list_f(lambda x: periods_pos[x[1]] + max_elapsed < len(periods))
        # this is the TTT_t set.
        # periods that are maintenance periods because of having assign a maintenance
        attt_maints = tl.TupList((a, t1, t2, t) for a, t1, t2 in att_maints for t in t2_at1.get((a, t1), []))
        attt_maints += tl.TupList((a, t1, t2, t) for a, t1, t2 in att_maints for t in t2_at1.get((a, t2), [])
                                  if t2 < last_period)
        attt_maints = attt_maints.unique2()

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
        t1_avt2 = avtt.to_dict(result_col=2, is_list=True)
        t_at_M = att_M.to_dict(result_col=2, is_list=True).fill_with_default(at, [])
        t_a_M_ini = at_M_ini.to_dict(result_col=1, is_list=True)
        tt_maints_at = attt_maints.to_dict(result_col=[1, 2], is_list=True)
        att_maints_t = attt_maints.to_dict(result_col=[0, 1, 2], is_list=True)
        tt_maints_a = att_maints.to_dict(result_col=[1, 2], is_list=True)

        vtt2_a = avtt2.to_dict(result_col=[1, 2, 3]).apply(lambda _, v: tl.TupList(v))
        vtt2_a_after_t = {(a, t): vtt2_a[a].filter_list_f(lambda x: x[1] >= t) for a in resources for t in periods}
        vtt2_a_before_t = {(a, t): vtt2_a[a].filter_list_f(lambda x: x[2] <= t) for a in resources for t in periods}
        vtt2_between_att = {(a, t1, t2): vtt2_a_after_t[a, t1].intersect(vtt2_a_before_t[a, t2])
                            for a in resources for pos1, t1 in enumerate(periods) for t2 in periods[pos1:]}

        return {
         'periods'          :  periods
        ,'period_0'         :  period_0
        ,'periods_0'        :  periods_0
        ,'periods_pos'      :  periods_pos
        ,'previous'         :  previous
        ,'tasks'            :  tasks
        ,'candidates'       :  candidates
        ,'resources'        :  resources
        ,'states'           :  states
        ,'vt'               :  vt
        ,'avt'              :  avt
        ,'at'               :  at
        ,'at_maint'         :  at_maint
        ,'at_mission_m'     : at_mission_m
        ,'ast'              :  ast
        ,'at_start'         :  list(at_start + at_free_start)
        ,'at0'              :  at0
        ,'att'              :  att
        ,'a_t'              :  a_t
        ,'a_vt'             :  a_vt
        ,'v_at'             :  v_at
        ,'t1_at2'           :  t1_at2
        ,'at1_t2'           :  at1_t2
        ,'t2_at1'           :  t2_at1
        ,'at_avail'         : list(at_free + at_mission)
        ,'t2_avt1'          : t2_avt1
        ,'t1_avt2'          : t1_avt2
        ,'avtt'             : avtt
        , 'avtt2'           : avtt2
        , 'tt2_avt'         : tt2_avt
        , 'att_m'           : att_m
        , 't_at_M'          : t_at_M
        , 'att_M'           : att_M
        , 'at_m_ini'        : at_m_ini
        , 't_a_M_ini'       : t_a_M_ini
        , 'kt'              : kt
        , 'att_maints'      : att_maints
        , 'att_cycles'      : att_cycles
        , 'att_maints_no_last': att_maints_no_last
        , 'tt_maints_at'    : tt_maints_at
        , 'att_maints_t'    : att_maints_t
        , 'tt_maints_a'     : tt_maints_a
        , 'vtt2_between_att': vtt2_between_att
        }



if __name__ == "__main__":
    pass
    pass
import math
import pulp as pl
import package.auxiliar as aux
import package.config as conf
import package.solution as sol
import package.experiment as exp
import package.tuplist as tl
import package.superdict as sd
import random as rn


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
        cluster_data = self.instance.get_cluster_constraints()
        c_candidates = self.instance.get_cluster_candidates()

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
        start_T = pl.LpVariable.dicts(name="start_T", indexs=l['avtt2'], lowBound=0, upBound=1, cat=pl.LpInteger)
        # task = pl.LpVariable.dicts(name="task", indexs=l['avt'], lowBound=0, upBound=1, cat=pl.LpInteger)
        start_M = pl.LpVariable.dicts(name="start_M", indexs=l['at_start'], lowBound=0, upBound=1, cat=pl.LpInteger)

        # numeric:
        rut = pl.LpVariable.dicts(name="rut", indexs=l['at0'], lowBound=0, upBound=ub['rut'], cat=var_type)
        usage = pl.LpVariable.dicts(name="usage", indexs=l['at'], lowBound=0, upBound=ub['used_max'], cat=var_type)

        # objective function:
        num_maint = pl.LpVariable(name="num_maint", lowBound=0, upBound=ub['num_maint'], cat=var_type)
        rut_obj_var = pl.LpVariable(name="rut_obj_var", lowBound=0, upBound=ub['rut_end'], cat=var_type)

        if options.get('mip_start'):
            main_starts = self.get_maintenance_periods()
            min_usage = self.instance.get_param('min_usage_period')

            # Initialize values:
            for tup in start_M:
                start_M[tup].setInitialValue(0)

            # for tup in task:
            #     task[tup].setInitialValue(0)

            for tup in start_T:
                start_T[tup].setInitialValue(0)

            for a, t in l['at']:
                usage[a, t].setInitialValue(min_usage)

            number_maint = 0
            for (a, t, t2) in main_starts:
                if (a, t) in l['at_start']:
                    # we check this because of fixed maints
                    start_M[a, t].setInitialValue(1)
                    number_maint += 1
                periods = self.instance.get_periods_range(t, t2)
                for p in periods:
                    if (a, p) in usage:
                        # we check because of previous assignments
                        usage[a, p].setInitialValue(0)

            start_periods = self.get_task_periods()
            task_usage = self.instance.get_tasks('consumption')
            for (a, t, v, t2) in start_periods:
                if (a, v, t) in start_T:
                    start_T[a, v, t].setInitialValue(1)
                periods = self.instance.get_periods_range(t, t2)
                for p in periods:
                    # if (a, v, p) in task:
                    #     task[a, v, p].setInitialValue(1)
                    if (a, p) in usage:
                        usage[a, p].setInitialValue(task_usage[v])

            rut_data = self.set_remaining_usage_time('rut')
            for a, date_info in rut_data.items():
                for t, v in date_info.items():
                    if (a, t) in rut:
                        rut[a, t].setInitialValue(v)

            num_maint.setInitialValue(number_maint)

            if options.get('fix_start', False):
                # vars_to_fix = [start_M]
                # vars_to_fix = [start_T, task, start_M, rut, usage, {0: rut_obj_var}, {0: num_maint}]
                vars_to_fix = [start_T, start_M, rut, usage, {0: rut_obj_var}, {0: num_maint}]
                for _vars in vars_to_fix:
                    for var in _vars.values():
                        var.fixValue()

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
        # if options.get('integer', False):
        #     objective = pl.LpVariable(name="objective", cat=var_type)
        #     model += objective
        #     model += objective >= num_maint * max_usage - rut_obj_var
        # else:
        model +=  num_maint * max_usage + \
                  - price_rut_end * rut_obj_var + \
                  1 * pl.lpSum(assign_st * price_assign[a, v]
                               for (a, v, t, t2), assign_st in start_T.items()) + \
                  1000000 * pl.lpSum(slack_vt.values()) +\
                  1000 * pl.lpSum(slack_at.values()) +\
                  10000 * pl.lpSum(slack_kt_num.values()) + \
                  1000 * pl.lpSum(slack_kt_hours.values()) + \
                  1000000 * pl.lpSum(slack_t.values())

        # To try Kozanidis objective function:
        # we sum the rut for all periods (we take out the periods under maintenance)
        # model += - sum(rut[tup] for tup in rut) + num_maint * max_usage * maint_duration

        # CONSTRAINTS:

        # max one task per period or unavailable state:
        for at in l['at']:
            a, t = at
            v_at = l['v_at'].get(at, [])  # possible missions for that "at"
            t1_at2 = l['t1_at2'].get(at, [])  # possible starts of maintenance to be in maintenance status at "at"
            if len(v_at) + len(t1_at2) == 0:
                continue
            model += pl.lpSum(start_T[a, v, t1, t2] for v in v_at for (t1, t2) in l['tt2_avt'][a, v, t]) + \
                     pl.lpSum(start_M[a, _t] for _t in t1_at2 if (a, _t) in l['at_start']) + \
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
            t1_t2_list = l['tt2_avt'][avt]
            model += pl.lpSum(start_T[a, v, t1, t2] for t1, t2 in t1_t2_list) >= 1

        # # ##################################
        # Clusters
        # ##################################

        # minimum availability per cluster and period
        for (k, t), num in cluster_data['num'].items():
            model += \
                pl.lpSum(start_M[(a, _t)] for (a, _t) in l['at1_t2'][t]
                         if (a, _t) in l['at_start']
                         if a in c_candidates[k]) <= num + slack_kt_num.get((k, t), 0)

        # Each cluster has a minimum number of usage hours to have
        # at each period.
        for (k, t), hours in cluster_data['hours'].items():
            model += pl.lpSum(rut[a, t] for a in c_candidates[k] if (a, t) in l['at']) \
                     >= hours - slack_kt_hours.get((k, t), 0)

        # ##################################
        # Usage time
        # ##################################

        # usage time calculation per month
        for at in l['at']:
            a, t = at
            v_list = l['v_at'][at]
            model += usage[at] >= pl.lpSum(start_T[a, v, t1, t2] * consumption[v]
                                           for v in v_list for (t1, t2) in l['tt2_avt'][a, v, t])

        for at in l['at']:
            # if resource in maintenance at the start, we do not enforce this.
            if at in l['at_maint']:
                continue
            a, t = at
            t1_at2 = l['t1_at2'].get(at, [])
            model += usage[at] >= min_usage * (1 - pl.lpSum(start_M[a, _t] for _t in t1_at2)) - slack_at[at]

        # remaining used time calculations:
        for at in l['at']:
            # apparently it's much faster NOT to sum the maintenances
            # in the two rut calculations below
            a, t = at
            at_prev = a, l['previous'][t]
            model += rut[at] <= rut[at_prev] - usage[at] + ub['rut'] * start_M.get(at, 0)
            model += rut[at] >= ub['rut'] * start_M.get(at, 0)

        # calculate the rut, only if it has a weight:
        if price_rut_end:
            model += pl.lpSum(rut[(a, last_period)] for a in l['resources']) == rut_obj_var

        for a in l['resources']:
            model += rut[a, l['period_0']] == rut_init[a]

        # ##################################
        # Maintenances
        # ##################################

        # # we cannot do two maintenances too close one from the other:
        for att in l['att_m']:
            a, t1, t2 = att
            model += start_M[a, t1] + start_M[a, t2] <= 1

        # we cannot do two maintenances too far apart one from the other:
        # (we need to be sure that t2_list includes the whole horizon to enforce it)
        for (a, t1), t2_list in l['t_at_M'].items():
            model += pl.lpSum(start_M[a, t2] for t2 in t2_list) >= start_M[a, t1]

        # if we need a maintenance inside the horizon, we enforce it
        for a, t_list in l['t_a_M_ini'].items():
            model += pl.lpSum(start_M.get((a, t), 0) for t in t_list) >= 1

        # count the number of maintenances:
        model += num_maint == pl.lpSum(start_M[(a, _t)] for (a, _t) in l['at_start'])

        # max number of maintenances:
        for t in l['periods']:
            model += pl.lpSum(start_M[a, _t] for (a, _t) in l['at1_t2'][t] if (a, _t) in l['at_start']) + \
                     num_resource_maint[t] <= maint_capacity + slack_t.get(t, 0)

        # model += start_M['1', '2018-01'] == 1

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
        task_periods = tl.TupList(aux.vars_to_tups(start_T))
        for a, v, t1, t2 in task_periods:
            for t in self.instance.get_periods_range(t1, t2):
                _task[a, t] = v

        # we store the start of maintenances and tasks in the same place
        _start = tl.TupList(aux.vars_to_tups(start_T)).filter([0, 1, 2]).to_dict(result_col=1, is_list=False)
        _start_M = {k: 'M' for k in aux.vars_to_tups(start_M)}
        _start.update(_start_M)

        # aux.vars_to_tups(slack_kt_hours)

        _rut = {t: rut[t].value() for t in rut}
        fixed_maints_horizon = l['at_maint'].filter_list_f(lambda x: first_period <= x[1] <= last_period)
        _state = {tup: 'M' for tup in fixed_maints_horizon}
        _state.update({(a, t2): 'M' for (a, t) in _start_M for t2 in l['t2_at1'][(a, t)]})

        solution_data_pre = {
            'state': _state,
            'task': _task,
            'aux': {
                'start': _start,
                'rut': _rut,
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
            # 'used_min': math.ceil(min_usage)
        }

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
        # start-assignment options for task assignments.
        avtt = tl.TupList([(a, v, t1, t2) for (a, v, t1) in avt for t2 in t_v[v] if
                periods_pos[t1] <= periods_pos[t2] < periods_pos[t1] + min_assign[v]
                ])
        # Start-stop options for task assignments.
        avtt2 = tl.TupList([(a, v, t1, t2) for (a, v, t1) in avt for t2 in t_v[v] if
                            (periods_pos[t2] >= periods_pos[t1] + min_assign[v] - 1) or
                            (t2 == last_period)
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
        att_maints = tl.TupList([(a, t1, t2) for (a, t1) in at_free_start for t2 in periods
                                 if (periods_pos[t1] + min_elapsed <= periods_pos[t2] < periods_pos[t1] + max_elapsed)
                                 or (len(periods) - periods_pos[t1]) <= min_elapsed
                                 ])
        att_M = att_maints.filter_list_f(lambda x: periods_pos[x[1]] + max_elapsed < len(periods))
        at_M_ini = tl.TupList([(a, t) for (a, t) in at_free_start
                    if ret_init[a] <= len(periods)
                    if ret_init_adjusted[a] <= periods_pos[t] <= ret_init[a]
                    ])
        avtt2t = tl.TupList(
            [(a, v, t1, t2, t) for (a, v, t1, t2) in avtt2 for t in self.instance.get_periods_range(t1, t2)]
        )
        tt2_avt = avtt2t.to_dict(result_col=[2, 3]).fill_with_default(avt, [])

        a_t = at.to_dict(result_col=0, is_list=True)
        a_vt = avt.to_dict(result_col=0, is_list=True)
        v_at = avt.to_dict(result_col=1, is_list=True).fill_with_default(at, [])
        at1_t2 = att.to_dict(result_col=[0, 1], is_list=True)
        t1_at2 = att.to_dict(result_col=1, is_list=True).fill_with_default(at, [])
        t2_at1 = att.to_dict(result_col=2, is_list=True)
        t2_avt1 = avtt.to_dict(result_col=3, is_list=True)
        t1_avt2 = avtt.to_dict(result_col=2, is_list=True)
        t_at_M = att_M.to_dict(result_col=2, is_list=True)
        t_a_M_ini = at_M_ini.to_dict(result_col=1, is_list=True)


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
        , 'at_m_ini'        : at_m_ini
        , 't_a_M_ini'       : t_a_M_ini
        , 'kt'              : kt
        , 'att_maints'      : att_maints
        }


if __name__ == "__main__":
    pass
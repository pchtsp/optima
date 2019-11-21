import solvers.model as md
import pulp as pl
import solvers.config as conf
import package.solution as sol
import pytups.superdict as sd
import random as rn


######################################################
# @profile

class ModelANOR(md.Model):

    def __init__(self, instance, solution=None):
        super().__init__(instance, solution)
        self.task = {}
        self.usage = {}


    def solve(self, options=None):
        self.domains = l = self.get_domains_sets(options)
        ub = self.get_variable_bounds()
        first_period, last_period = self.instance.get_start_end()
        consumption = sd.SuperDict.from_dict(self.instance.get_tasks('consumption'))
        requirement = self.instance.get_tasks('num_resource')
        rut_init = self.instance.get_initial_state("used")
        num_resource_maint = \
            sd.SuperDict.from_dict(self.instance.get_total_fixed_maintenances()).\
            fill_with_default(keys=l['periods'])
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

        # Sometimes we want to force variables to be integer or continuous
        normally_continuous = pl.LpContinuous
        normally_integer = pl.LpInteger
        integer_problem = options.get('integer', False)
        relaxed_problem = options.get('relax', False)
        if integer_problem:
            normally_continuous = pl.LpInteger
        elif relaxed_problem:
            normally_integer = pl.LpContinuous

        # VARIABLES:
        # binary:
        self.start_T = start_T = pl.LpVariable.dicts(name="start_T", indexs=l['avt'], lowBound=0, upBound=1, cat=normally_integer)
        self.task = task = pl.LpVariable.dicts(name="task", indexs=l['avt'], lowBound=0, upBound=1, cat=normally_integer)
        self.start_M = start_M = pl.LpVariable.dicts(name="start_M", indexs=l['at_start_maint'], lowBound=0, upBound=1, cat=normally_integer)

        # numeric:
        self.rut = rut = pl.LpVariable.dicts(name="rut", indexs=l['at0'], lowBound=0, upBound=ub['rut'], cat=normally_continuous)
        self.usage = usage = pl.LpVariable.dicts(name="usage", indexs=l['at'], lowBound=0, upBound=ub['used_max'], cat=normally_continuous)
        # last_maint = pl.LpVariable.dicts(name="last_maint", indexs=l['resources'], lowBound=0, upBound=len(l['periods']), cat=normally_continuous)

        # slack variables:
        price_slack_kts = {s: (p+1)*50 for p, s in enumerate(l['slots'])}
        price_slack_ts = {s: (p+1)*1000 for p, s in enumerate(l['slots'])}
        price_slack_kts_h = {s: (p + 2)**2 for p, s in enumerate(l['slots'])}

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
        model = pl.LpProblem("MFMP_v0002", pl.LpMinimize)

        # try to make the second maintenance the most late possible
        period_pos = self.instance.data['aux']['period_i']
        # the following costs tries to imitate the penalization for doing a
        # second maintenance
        cost_maint = l['att_maints'].\
            take([0, 2]).\
            to_dict(None).\
            kapply(lambda k: period_pos[k[1]]).\
            fill_with_default(l['at_start_maint'])

        # OBJECTIVE:
        # maints
        objective = \
            pl.lpSum(price_assign[a, v] * task for (a, v, t), task in start_T.items()) + \
            + 10 * pl.lpSum(price_slack_kts[s] * slack for (k, t, s), slack in slack_kts.items()) \
            + 10 * pl.lpSum(price_slack_kts_h[s] * slack for (k, t, s), slack in slack_kts_h.items()) \
            + 10 * pl.lpSum(price_slack_ts[s] * slack for (t, s), slack in slack_ts.items()) \
            - pl.lpSum(start_M[a, t] * cost_maint[a, t] for a, t in l['at_start_maint']) \
            + pl.lpSum(start_M.values())* period_pos[last_period] \
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
            t1_at2 = l['t1_at2'].get(at, [])  # possible starts of maintenance to be in maintenance status at "at"
            if len(v_at) + len(t1_at2) == 0:
                continue
            model += pl.lpSum(task[a, v, t] for v in v_at) + \
                     pl.lpSum(start_M[a, _t] for _t in t1_at2 if (a, _t) in l['at_start_maint']) + \
                     (at in l['at_maint']) <= 1

        # ##################################
        # Tasks and tasks starts
        # ##################################

        # num resources:
        for (v, t), a_list in l['a_vt'].items():
            model += pl.lpSum(task[a, v, t] for a in a_list) >= requirement[v] - slack_vt.get((v, t), 0)

        # definition of task start:
        # if we have a task now but we didn't before: we started it
        for avt in l['avt']:
            a, v, t = avt
            avt_prev = a, v, l['previous'][t]
            if t != first_period:
                model += start_T[avt] >= task[avt] - task.get(avt_prev, 0)
            else:
                # we check if we have the assignment in the previous fixed period.
                model += start_T[avt] >= task[avt] - (avt_prev in l['at_mission_m'])

        # definition of task start (2):
        # if we start a task in at least one earlier period, we need to assign a task
        for (a, v, t2), t1_list in l['t1_avt2'].items():
            avt2 = a, v, t2
            model += task.get(avt2, 0) >= \
                     pl.lpSum(start_T.get((a, v, t1), 0) for t1 in t1_list)

        # at the beginning of the planning horizon, we may have fixed assignments of tasks.
        # we need to fix the corresponding variable.
        for avt in l['at_mission_m']:
            if avt in task:
                # sometimes the assignments are in the past
                task[avt].setInitialValue(1)
                task[avt].fixValue()

        # ##################################
        # Clusters
        # ##################################

        # minimum availability per cluster and period
        for (k, t), num in cluster_data['num'].items():
            model += \
                pl.lpSum(start_M[(a, _t)] for (a, _t) in l['at1_t2'][t]
                         if (a, _t) in l['at_start_maint']
                         if a in c_candidates[k]) <= num + pl.lpSum(slack_kts[k, t, s] for s in l['slots'])

        # Each cluster has a minimum number of usage hours to have
        # at each period.
        for (k, t), hours in cluster_data['hours'].items():
            model += pl.lpSum(rut[a, t] for a in c_candidates[k] if (a, t) in l['at']) >= hours - \
                     pl.lpSum(slack_kts_h[k, t, s] for s in l['slots'])

        # ##################################
        # Usage time
        # ##################################

        # usage time calculation per month
        for avt in l['avt']:
            a, v, t = avt
            at = a, t
            model += usage[at] >= task[avt] * consumption[v]

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

        # if we have had a maintenance just before the planning horizon
        # we cant't have one at the beginning:
        # we can formulate this as constraining the combinations of maintenance variables.
        # (already done)
        # for at in l['at_m_ini']:
        #     model += start_M[at] == 0

        # if we need a maintenance inside the horizon, we enforce it
        for a, t_list in l['t_a_M_ini'].items():
            model += pl.lpSum(start_M.get((a, t), 0) for t in t_list) >= 1

        # max number of maintenances:
        for t in l['periods']:
            model += pl.lpSum(start_M[a, _t] for (a, _t) in l['at1_t2'][t] if (a, _t) in l['at_start_maint']) + \
                     num_resource_maint[t] <= \
                     maint_capacity + pl.lpSum(slack_ts.get((t, s), 0) for s in l['slots'])

        # we need to cap the number of maintenances to 2.
        # so it is equivalent to the present model.
        a_t_start_maint = l['at_start_maint'].to_dict(result_col=1)
        for a, _periods in a_t_start_maint.items():
            model += pl.lpSum(start_M.get((a, t), 0) for t in _periods) <= 2

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
    #
    def fill_initial_solution(self):

        """
        :param variables: variable dictionary to unpack
        :param vars_to_fix: possible list of variables to fix. List of dicts assumed)
        :return:
        """
        # raise NotImplementedError('Not yet!')
        l = self.domains
        if not len(l):
            raise ValueError('Model has not been solved yet. No domains are generated')

        first = self.instance.get_param('start')
        last = self.instance.get_param('end')
        main_starts = self.get_maintenance_periods()
        min_usage = self.instance.get_param('min_usage_period')

        start_M = self.start_M
        task = self.task
        start_T = self.start_T
        rut = self.rut
        usage = self.usage

        # Initialize values:
        for tup in start_M:
            start_M[tup].setInitialValue(0)

        for tup in task:
            task[tup].setInitialValue(0)

        for tup in start_T:
            start_T[tup].setInitialValue(0)

        for a, t in l['at']:
            usage[a, t].setInitialValue(min_usage)

        # number_maint = 0
        for (a, t, t2) in main_starts:
            if (a, t) in l['at_start_maint']:
                # we check this because of fixed maints
                start_M[a, t].setInitialValue(1)

            # periods = self.instance.get_periods_range(t, t2)
            # for p in periods:
            #     if (a, p) in usage:
            #         # we check because of previous assignments
            #         usage[a, p].setInitialValue(0)

        start_periods = self.get_task_periods()
        # task_usage = self.instance.get_tasks('consumption')
        for (a, t, v, t2) in start_periods:
            if (a, v, t) in start_T:
                start_T[a, v, t].setInitialValue(1)
            periods = self.instance.get_periods_range(t, t2)
            for p in periods:
                if (a, v, p) in task:
                    task[a, v, p].setInitialValue(1)
        #         if (a, p) in usage:
        #             usage[a, p].setInitialValue(task_usage[v])
        #
        # rut_data = self.set_remaining_usage_time('rut')
        # for a, date_info in rut_data.items():
        #     for t, v in date_info.items():
        #         if (a, t) in rut:
        #             rut[a, t].setInitialValue(v)

    def get_solution(self):

        l = self.domains
        if not len(l):
            raise ValueError('Model has not been solved yet. No domains are generated')

        first_period = self.instance.get_param('start')
        last_period = self.instance.get_param('end')

        _task = self.vars_to_tups(self.task).to_dict(result_col=1, is_list=False)

        # we store the start of maintenances and tasks in the same place
        _start = self.vars_to_tups(self.start_T).to_dict(result_col=1, is_list=False)
        _start_M = {k: 'M' for k in self.vars_to_tups(self.start_M)}
        _start.update(_start_M)

        fixed_maints_horizon = l['at_maint'].vfilter(lambda x: first_period <= x[1] <= last_period)
        _state = {tup: 'M' for tup in fixed_maints_horizon}
        _state.update({(a, t2): 'M' for (a, t) in _start_M for t2 in l['t2_at1'][(a, t)]})

        solution_data_pre = {
            'state': _state,
            'task': _task,
            'aux': {
                'start': _start,
                'rut': self.vars_to_dicts(self.rut),
                'slack_kts_h': self.vars_to_dicts(self.slack_kts_h),
                'slack_kts': self.vars_to_dicts(self.slack_kts),
                'slack_ts': self.vars_to_dicts(self.slack_ts)

            }
        }
        solution_data_pre = sd.SuperDict.from_dict(solution_data_pre)

        solution_data = {k: v.to_dictdict()
                         for k, v in solution_data_pre.items() if k != "aux"}
        solution_data['aux'] = {k: v.to_dictdict()
                                for k, v in solution_data_pre['aux'].items()}
        solution = sol.Solution(solution_data)
        return solution

if __name__ == "__main__":
    pass
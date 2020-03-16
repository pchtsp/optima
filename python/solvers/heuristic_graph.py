import pytups.superdict as sd
import pytups.tuplist as tl

import solvers.heuristics as heur
import solvers.model as mdl
import solvers.config as conf
import solvers.heuristics_maintfirst as heur_maint

import package.solution as sol
import package.instance as inst

import patterns.create_patterns as cp
import patterns.node as nd

import data.data_input as di

import pulp as pl
import operator as op
import random as rn
import math
import time
import logging as log
import multiprocessing as multi
import pandas as pd


class GraphOriented(heur.GreedyByMission, mdl.Model):
    {'aux': {'graphs': {'RESOURCE': {'graph': {}, 'refs': {}, 'refs_inv': {}, 'source': {}, 'sink': {}}}}}

    # graph=g, refs=refs, refs_inv=refs.reverse(), source=source, sink=sink

    def __init__(self, instance, solution=None):

        heur.GreedyByMission.__init__(self, instance, solution)
        self.instance.data['aux']['graphs'] = sd.SuperDict()

    def initialise_graphs(self, options):
        multiproc = options['multiprocess']
        resources = self.instance.get_resources()
        if not multiproc:
            for r in resources:
                graph_data = cp.get_graph_of_resource(instance=self.instance, resource=r)
                self.instance.data['aux']['graphs'][r] = graph_data
            return
        results = {}
        with multi.Pool(processes=multiproc) as pool:
            for r in resources:
                _instance = inst.Instance.from_instance(self.instance)
                results[r] = pool.apply_async(cp.get_graph_of_resource, [_instance, r])
            for r, result in results.items():
                self.instance.data['aux']['graphs'][r] = result.get(timeout=10000)
        return


    def solve(self, options):
        """
         Solves an instance using the metaheuristic.

         :param options: dictionary with options
         :return: solution to the planning problem
         :rtype: :py:class:`package.solution.Solution`
         """
        temperature = options.get('temperature', 1)
        cooling = options.get('cooling', 0.995)
        max_time = options.get('timeLimit', 600)
        max_iters = options.get('max_iters', 99999999)
        max_iters_initial = options.get('max_iters_initial', 10)
        big_window = options.get('big_window', False)
        num_max = options.get('num_max', 10000)
        options_repair = di.copy_dict(options)
        options_repair = sd.SuperDict.from_dict(options_repair)
        options_repair['timeLimit'] = options.get('timeLimit_cycle', 10)

        # 1. get an initial solution.
        if self.solution is None or max_iters_initial:
            first_solve = heur_maint.MaintenanceFirst(self.instance, self.solution)
            first_solve.get_objective_function = self.get_objective_function
            options_fs = {**options, **dict(max_iters=max_iters_initial, assign_missions=True)}
            first_solve.solve(options_fs)
            self.set_solution(first_solve.best_solution)

        # initialise logging, seed and solution status
        self.set_log_config(options)
        self.initialise_solution_stats()
        self.initialise_seed(options)
        self.initialise_graphs(options)

        # 2. repair solution
        time_init = time.time()
        i = 0
        errors = sd.SuperDict()
        while True:
            # if rn.random() > 0.5 or not errors:
            #     change = self.get_candidate_random()
            # elif rn.random() > 0.5:
            #     change = self.get_candidates_tasks(errors)
            # else:
            #     change = self.get_candidates_cluster(errors)
            # if not change:
            #     continue
            change = self.get_candidate_random()
            if big_window:
                change = self.get_candidate_all()
            log.info('Repairing periods {start} => {end} for resources: {resources}'.format(**change))
            patterns = {r: self.get_pattern_options_from_window(r, change['start'], change['end'], num_max=num_max) for r in change['resources']}
            patterns = sd.SuperDict(patterns).vfilter(lambda v: len(v) > 0)
            if not len(patterns):
                continue
            self.solve_repair(patterns, options_repair)

            objective, status, errors = self.analyze_solution(temperature, True)
            num_errors = errors.to_lendict().values_tl()
            num_errors = sum(num_errors)

            # sometimes, we go back to the best solution found
            if objective > self.best_objective and rn.random() < 0.01:
                self.set_solution(self.best_solution)
                objective = self.prev_objective = self.best_objective
                log.info('back to best solution: {}'.format(self.best_objective))

            if status in self.status_worse:
                temperature *= cooling

            # self.get_objective_function()

            clock = time.time()
            time_now = clock - time_init

            log.info("time={}, iteration={}, temperaure={}, current={}, best={}, errors={}".
                     format(round(time_now), i, round(temperature, 4), objective, self.best_objective, num_errors))
            i += 1

            if not self.best_objective or i >= max_iters or time_now > max_time:
                break

        return sol.Solution(self.best_solution)

        pass

    def get_graph_data(self, resource):
        return self.instance.data['aux']['graphs'][resource]

    def get_source_node(self, resource):
        return self.instance.data['aux']['graphs'][resource]['source']

    def get_sink_node(self, resource):
        return self.instance.data['aux']['graphs'][resource]['sink']

    def get_objective_function(self, error_cat=None):
        """
        Calculates the objective function for the current solution.

        :param sd.SuperDict error_cat: possibility to take a cache of errors
        :return: objective function
        :rtype: int
        """
        if error_cat is None:
            error_cat = self.check_solution().to_lendict()
        num_errors = sum(error_cat.values())

        # we count the number of maintenances and their distance to the end
        first, last = self.instance.get_first_last_period()
        _dist = self.instance.get_dist_periods
        maintenances = \
            self.get_maintenance_periods(). \
                take(1). \
                vapply(_dist, last)

        return num_errors * 1000 + sum(maintenances)

    def filter_patterns(self, node2, patterns):
        """
        gets feasible patterns with the resource's maintenance cycle

        :param node2:
        :param patterns:
        :return:
        """
        first, last = self.instance.get_first_last_period()
        _shift = self.instance.shift_period
        next_maint = self.get_next_maintenance(node2.resource, _shift(node2.period_end, 1), {'M'})
        if next_maint is None:
            _period_to_look = last
        else:
            # If there is a next maintenances, we filter patterns depending on the last rut
            _period_to_look = _shift(next_maint, -1)
        rut_cycle = self.get_remainingtime(node2.resource, _period_to_look, 'rut', maint=self.M)
        rut = self.get_remainingtime(node2.resource, node2.period_end, 'rut', maint=self.M)
        if rut is None:
            # it could be we are at the dummy node at the end.
            # here, we would not care about filtering
            return patterns
        min_rut = rut - rut_cycle
        return patterns.vfilter(lambda v: v[-2].rut[self.M] >= min_rut)

    def apply_pattern(self, pattern):
        resource = pattern[0].resource
        _next = self.instance.get_next_period
        _prev = self.instance.get_prev_period
        start = _next(pattern[0].period_end)
        end = _prev(pattern[-1].period)

        deleted_tasks = self.clean_assignments_window(resource, start, end, 'task')
        deleted_maints = self.clean_assignments_window(resource, start, end, 'state_m')

        # We apply all the assignments
        # Warning, here I'm ignoring the two extremes[1:-1]
        added_maints = tl.TupList()
        added_tasks = tl.TupList()
        for node in pattern[1:-1]:
            if node.type == nd.TASK_TYPE:
                added_tasks += self.apply_task(node)
            elif node.type == nd.MAINT_TYPE:
                added_maints += self.apply_maint(node)

        # Update rut and ret.
        # for this we need to join all added and deleted things:
        all_modif = \
            tl.TupList(deleted_tasks + deleted_maints + added_maints + added_tasks). \
                unique2().sorted()
        if all_modif:
            first_date_to_update = all_modif[0][0]
        else:
            first_date_to_update = start
        times = ['rut']
        if added_maints or deleted_maints:
            times.append('ret')
        for t in times:
            for m in self.instance.get_rt_maints(t):
                self.set_remaining_time_in_window(resource, m, first_date_to_update, end, t)
        return True

    def apply_maint(self, node):
        result = tl.TupList()
        for period in self.instance.get_periods_range(node.period, node.period_end):
            self.solution.data.set_m('state_m', node.resource, period, node.assignment, value=1)
            result.add(period, node.assignment)
        return result

    def apply_task(self, node):
        result = tl.TupList()
        for period in self.instance.get_periods_range(node.period, node.period_end):
            self.solution.data.set_m('task', node.resource, period, value=node.assignment)
            result.add(period, node.assignment)
        return result

    def date_to_node(self, resource, date, use_rt=True):

        # We check what the resource has in this period:
        assignment, category = self.solution.get_period_state_category(resource, date)
        if category is None:
            # if nothing, we format as an EMPTY node
            period_start = date
            period_end = date
        else:
            # if there is something, we look for the limits of the assignment
            # to create the correct node.
            period_start = self.get_limit_of_assignment(resource, date, category)
            period_end = self.get_limit_of_assignment(resource, date, category, get_last=True)

        if category is None:
            assignment = ''
            _type = nd.EMPTY_TYPE
        elif category == 'task':
            _type = nd.TASK_TYPE
        elif category == 'state_m':
            _type = nd.MAINT_TYPE
            # maintenances have a different format for assignments
            assignment = assignment.keys_l()[0]
        else:
            _type = nd.DUMMY_TYPE

        if use_rt:
            maints = self.instance.get_maintenances().vapply(lambda v: 1)
            # we want to have the status of the resource in the moment the assignment ends
            # because that will be the status on the node
            _defaults = dict(resource=resource, period=period_end)
            _rts = \
                sd.SuperDict(ret=maints, rut=maints). \
                    to_dictup(). \
                    kapply(lambda k: dict(**_defaults, time=k[0], maint=k[1])). \
                    vapply(lambda v: self.get_remainingtime(**v)). \
                    to_dictdict()
        else:
            _rts = sd.SuperDict(rut=None, ret=None)
        # we return the state of type: period, period_end, assignment, type, rut, ret
        state = sd.SuperDict(period=period_start, period_end=period_end,
                             assignment=assignment, type=_type, **_rts)
        return cp.state_to_node(self.instance, resource=resource, state=state)

    def get_pattern_options_from_window(self, resource, date1, date2, num_max=10000, **kwargs):
        """
        This method is accessed by the repairing method.
        To know the options of assignments

        :param resource:
        :param date1:
        :param date2:
        :return:
        """
        node1 = self.date_to_node(resource, date1, use_rt=True)
        node2 = self.date_to_node(resource, date2, use_rt=False)
        patterns = cp.nodes_to_patterns(node1=node1, node2=node2, **self.get_graph_data(resource), **kwargs)
        # we need to filter them to take out the ones that compromise the post-window periods
        p_filtered = self.filter_patterns(node2, patterns)
        if len(p_filtered) > num_max:
            return rn.sample(p_filtered, num_max)
        return p_filtered

    def get_pattern_from_window(self, resource, date1, date2):
        _range = self.instance.get_periods_range
        nodes = tl.TupList([self.date_to_node(resource, p, use_rt=True) for p in _range(date1, date2)])
        return nodes.unique2().sorted(key=lambda k: k.period)

    def set_remaining_time_in_window(self, resource, maint, start, end, rt):
        _shift = self.instance.shift_period
        _start = start
        maint_data = self.instance.data['maintenances'][maint]
        duration = maint_data['duration_periods']
        while _start <= end:
            # iterate over maintenance periods
            updated_periods = self.update_rt_until_next_maint(resource, _start, maint, rt)
            if not updated_periods:
                # means we're in a maintenance...
                maint_start = _start
            else:
                maint_start = _shift(updated_periods[-1], 1)
            if maint_start > end:
                # we've reached the end, no more updating needed
                break
            # if the maintenance falls inside the period: update the maintenance values
            # and continue the while
            horizon_end = self.instance.get_param('end')
            affected_maints = maint_data['affects']
            maint_end = min(self.instance.shift_period(maint_start, duration - 1), horizon_end)
            periods_maint = self.instance.get_periods_range(maint_start, maint_end)
            for m in affected_maints:
                self.update_time_maint(resource, periods_maint, time='ret', maint=m)
                self.update_time_maint(resource, periods_maint, time='rut', maint=m)
            _start = _shift(maint_end, 1)
        return True

    def clean_assignments_window(self, resource, start, end, key='task'):
        """
        cleans all assignments and returns them.
        :param resource:
        :param start:
        :param end:
        :param key:
        :return: tl.TupList
        """
        # here, we will be take the risk of just deleting everything.
        data = self.solution.data
        fixed_periods = self.instance.get_fixed_periods().to_set()

        delete_assigned = tl.TupList()
        periods = self.instance.get_periods_range(start, end)
        if resource not in data[key]:
            return delete_assigned
        for period in periods:
            if (resource, period) in fixed_periods:
                continue
            info = data[key][resource].pop(period, None)
            if info is not None:
                if key == 'state_m':
                    delete_assigned.add(period, info.keys_l()[0])
                    continue
                delete_assigned.add(period, info)
        return delete_assigned

    def get_assignments_window(self, resource, start, end, key='task'):
        """
        Returns all assignments during the window
        :param resource:
        :param start:
        :param end:
        :param key:
        :return: tl.TupList
        """
        # here, we will be take the risk of just deleting everything.
        data = self.solution.data

        info = tl.TupList()
        periods = self.instance.get_periods_range(start, end)
        if resource not in data[key]:
            return info
        for period in periods:
            assignment = data.get_m(key, resource, period)
            if assignment is None:
                continue
            if key == 'task':
                info.add(resource, -1, period, assignment, nd.TASK_TYPE)
                continue
            for m in assignment:
                info.add(resource, -1, period, m, nd.MAINT_TYPE)
        return info

    def get_domains_sets_patterns(self, options, res_patterns, force=False):
        if self.domains and not force:
            return self.domains
        self.domains = mdl.Model.get_domains_sets(self, options, force=False)
        resources = res_patterns.keys_tl()
        resource = resources[0]
        example_pattern = res_patterns[resource][0]
        start = example_pattern[0].period
        end = example_pattern[-1].period
        old_info = tl.TupList()
        for r in resources:
            for key in ['task', 'state_m']:
                old_info += self.get_assignments_window(resource=r, start=start, end=end, key=key)

        _range = self.instance.get_periods_range
        _dist = self.instance.get_dist_periods
        combos = sd.SuperDict()
        info = tl.TupList()

        rut_or_None = lambda rut: rut[self.M] if rut is not None else None

        for res, _pat_list in res_patterns.items():
            for p, pattern in enumerate(_pat_list):
                combos[res, p] = pattern
                info += tl.TupList(
                    (res, p, period, e.assignment, e.type, rut_or_None(e.rut), pos, _dist(e.period, e.period_end))
                    for e in pattern for pos, period in enumerate(_range(e.period, e.period_end))
                )

        self.domains['combos'] = combos
        self.domains['info'] = info
        self.domains['info_old'] = old_info
        return self.domains

    def solve_repair(self, res_patterns, options):

        # res_patterns is a sd.SuperDict of resource: [list of possible patterns]
        l = self.get_domains_sets_patterns(res_patterns=res_patterns, options=options, force=True)

        # Variables:
        model_vars = self.build_vars()

        if options.get('mip_start'):
            self.fill_initial_solution(model_vars)
        vars_to_fix = options.get('fix_vars', [])
        if vars_to_fix:
            self.fix_variables(model_vars.filter(vars_to_fix, check=False))

        # We all constraints at the same time:
        objective_function, constraints = self.build_model(**model_vars)
        model = pl.LpProblem('MFMP_repair', sense=pl.LpMinimize)
        model += objective_function
        for c in constraints:
            model += c

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
        # tl.TupList(model.variables()).to_dict(None).vapply(pl.value).vfilter(lambda v: v)
        self.solution = self.get_repaired_solution(l['combos'], model_vars['assign'])
        return self.solution

    def get_repaired_solution(self, combos, assign):
        patterns = assign. \
            vapply(pl.value). \
            vfilter(lambda v: v). \
            kapply(lambda k: combos[k]). \
            values_tl()
        for p in patterns:
            self.apply_pattern(p)
        return self.solution

    def draw_graph(self, resource):
        graph_data = self.get_graph_data(resource)
        cp.draw_graph(self.instance, graph_data['graph'], graph_data['refs_inv'])
        return True

    def get_candidates_tasks(self, errors):
        """
        :return: a list of resources, a start time and an end time.
        :rtype: sd.SuperDict
        """
        tasks_probs = \
            errors.get('resources', sd.SuperDict()).\
            to_tuplist().to_dict(result_col=2, indices=[0]).vapply(sum).to_tuplist().sorted()
        if not tasks_probs:
            return sd.SuperDict()
        _tasks, _probs = zip(*tasks_probs)
        task = rn.choices(_tasks, weights=_probs, k=1)[0]
        t_cand = self.instance.get_task_candidates()[task]
        t_info = self.instance.get_tasks()
        start, end = t_info[task]['start'], t_info[task]['end']
        return sd.SuperDict(resources=t_cand, start=start, end=end)

    def get_candidate_random(self):
        _shift = self.instance.shift_period
        periods = self.instance.get_periods()
        resources = self.instance.get_resources().keys_tl().sorted()
        first, last = self.instance.get_first_last_period()
        start = rn.choice(periods)
        size = rn.choice(range(20)) + 1
        end = _shift(start, size)
        if end > last:
            return sd.SuperDict()
        # we choose a subset of resources
        size_sample = rn.choice(range(len(resources))) + 1
        res_to_change = rn.sample(resources, size_sample)
        return sd.SuperDict(resources=res_to_change, start=start, end=end)

    def get_candidate_all(self):
        instance = self.instance
        first, last = instance.get_first_last_period()
        resources = instance.get_resources().keys_tl()
        _shift = self.instance.shift_period
        return sd.SuperDict(resources=resources, start=_shift(first, -1), end=_shift(last))

    def get_candidates_cluster(self, errors):
        """
        :return: a list of candidates [(aircraft, period), (aircraft2, period2)] to free
        :rtype: tl.TupList
        """
        _dist = self.instance.get_dist_periods
        clust_hours = errors.get('hours', sd.SuperDict())
        if not len(clust_hours):
            return sd.SuperDict()
        options = clust_hours.keys_tl().to_dict(1).vapply(lambda v: sd.SuperDict(start=v[0], end=v[-1]))
        opt_probs = options.vapply(lambda v: _dist(v['start'], v['end'])).to_tuplist()
        _cluts, _probs = zip(*opt_probs)
        cluster = rn.choices(_cluts, weights=_probs, k=1)[0]
        c_cand = self.instance.get_cluster_candidates()[cluster]
        return options[cluster]._update(sd.SuperDict(resources=c_cand))

    def build_vars(self):

        l = self.domains
        combos = l['combos']
        _vars = sd.SuperDict()
        assign = pl.LpVariable.dicts(name="assign", indexs=combos, cat=pl.LpBinary)
        _vars['assign'] = sd.SuperDict.from_dict(assign)

        ub = self.get_variable_bounds()
        p_s = {s: p for p, s in enumerate(l['slots'])}

        def make_slack_var(name, domain, ub_func):
            _var = {tup:
                        pl.LpVariable(name="{}_{}".format(name, tup), lowBound=0,
                                      upBound=ub_func(tup), cat=pl.LpContinuous)
                    for tup in domain
                    }
            _var = sd.SuperDict(_var)
            return _var

        _vars['slack_kts'] = make_slack_var('slack_kts', l['kts'],
                                            lambda kts: ub['slack_kts'][kts[2]])
        _vars['slack_kts_h'] = make_slack_var('slack_kts_h', l['kts'],
                                              lambda kts: ub['slack_kts_h'][(kts[0], kts[2])])
        _vars['slack_ts'] = make_slack_var('slack_ts', l['ts'],
                                           lambda ts: ub['slack_ts'][ts[1]])
        _vars['slack_vts'] = make_slack_var('slack_vts', [(*vt, s) for vt in l['vt'] for s in l['slots']],
                                            lambda vts: p_s[vts[2]] * 2 + 1)

        return _vars

    def fill_initial_solution(self, vars):

        l = self.domains
        combos = l['combos']
        resources = combos.keys_tl().take(0).unique()
        res_old_pat = l['info_old'].to_dict(result_col=[2, 3, 4], indices=[0]).vapply(set)

        res_pat_opts = \
            l['info']. \
                vfilter(lambda v: v[3]). \
                to_dict(result_col=[2, 3, 4], indices=[0, 1]). \
                vapply(set). \
                to_dictdict()

        _clean = lambda v: v[0] not in res_pat_opts or v[1] not in res_pat_opts[v[0]]
        empties = l['info'].take([0, 1]).unique2().vfilter(_clean)

        for res, pattern in empties:
            res_pat_opts.set_m(res, pattern, value=set())

        res_pattern = sd.SuperDict()
        for r in resources:
            pats = res_pat_opts.get(r)
            _set2 = res_old_pat.get(r, set())
            for p, _set1 in pats.items():
                dif = _set2 ^ _set1
                if not len(dif):
                    res_pattern[r] = p
                    break

        for v in vars['assign'].values():
            v.setInitialValue(0)

        for k, v in res_pattern.items():
            vars['assign'][k, v].setInitialValue(1)

        return

    def fix_variables(self, vars_to_fix):
        for var_group in vars_to_fix.values():
            for _var in var_group.values():
                _var.fixValue()
        return

    def build_model(self, assign, slack_vts, slack_ts, slack_kts, slack_kts_h):

        l = self.domains
        info = \
            l['info']. \
                to_dict(result_col=[0, 1, 2, 3], indices=[4]). \
                fill_with_default([nd.MAINT_TYPE, nd.TASK_TYPE], default=tl.TupList())
        first, last = self.instance.get_first_last_period()
        combos = l['combos']
        periods = l['info'].take(2).unique().vfilter(lambda v: first <= v <= last).sorted()
        resources = combos.keys_tl().take(0).unique().sorted()
        start = periods[0]
        end = periods[-1]
        old_info = l['info_old']. \
            to_dict(result_col=[0, 1, 2, 3]). \
            fill_with_default([nd.MAINT_TYPE, nd.TASK_TYPE], default=tl.TupList())
        _dist = self.instance.get_dist_periods
        p_s = {s: p for p, s in enumerate(l['slots'])}

        p_t = self.instance.data['aux']['period_i']
        weight_pattern = \
            lambda pattern: \
                sum(_dist(elem.period, last) for elem in pattern
                    if elem.type == nd.MAINT_TYPE)
        assign_p = combos.vapply(weight_pattern)

        def sum_of_slots(variable_slots):
            result_col = len(variable_slots.keys_l()[0])
            indices = range(result_col - 1)
            return \
                variable_slots. \
                    to_tuplist(). \
                    to_dict(result_col=result_col, indices=indices). \
                    vapply(pl.lpSum)

        def sum_of_two_dicts(dict1, dict2):
            return \
                (dict1.keys_tl() + dict2.keys_tl()). \
                    to_dict(None). \
                    kapply(lambda k: dict1.get(k, 0)). \
                    kvapply(lambda k, v: v + dict2.get(k, 0))

        _slack_s_kt = sum_of_slots(slack_kts)
        slack_kts_p = {(k, t, s): (p_s[s] + 1 - 0.001 * p_t[t]) * 50 for k, t, s in l['kts']}

        _slack_s_kt_h = sum_of_slots(slack_kts_h)
        slack_kts_h_p = {(k, t, s): (p_s[s] + 2 - 0.001 * p_t[t]) ** 2 for k, t, s in l['kts']}

        _slack_s_t = sum_of_slots(slack_ts)
        slack_ts_p = {(t, s): (p_s[s] + 1 - 0.001 * p_t[t]) * 1000 for t, s in l['ts']}

        _slack_s_vt = sum_of_slots(slack_vts)
        slack_vts_p = slack_vts.kapply(lambda vts: (p_s[vts[2]] + 1) * 10000)

        # Constraints
        constraints = tl.TupList()

        # objective:
        objective_function = \
            pl.lpSum((assign * assign_p).values_tl()) + \
            100*pl.lpSum((slack_vts * slack_vts_p).values_tl()) + \
            10*pl.lpSum((slack_ts * slack_ts_p).values_tl()) + \
            10*pl.lpSum((slack_kts * slack_kts_p).values_tl()) + \
            10*pl.lpSum((slack_kts_h * slack_kts_h_p).values_tl())

        # one pattern per resource:
        constraints += \
            combos.keys_tl(). \
                vapply(lambda v: (v[0], *v)).to_dict(result_col=[1, 2]). \
                vapply(lambda v: pl.lpSum(assign[vv] for vv in v) == 1). \
                values_tl()

        # ##################################
        # Tasks and tasks starts
        # ##################################
        log.debug("constraints: tasks")
        # num resources:
        # TODO: filter inside the function
        mission_needs = self.check_task_num_resources(deficit_only=False).kfilter(lambda v: start <= v[1] <= end)

        prev_mission_needs = \
            old_info[nd.TASK_TYPE]. \
                to_dict(indices=[3, 2], result_col=[0, 1]). \
                vapply(len)
        t_mission_needs = sum_of_two_dicts(mission_needs, prev_mission_needs)

        p_mission = \
            info[nd.TASK_TYPE]. \
                to_dict(indices=[3, 2], result_col=[0, 1]). \
                fill_with_default(t_mission_needs, default=[])

        constraints += \
            p_mission. \
                kfilter(lambda k: t_mission_needs[k] > 0). \
                vapply(lambda v: [(assign[vv], 1) for vv in v]). \
                kvapply(lambda k, v: [(e, 1) for e in _slack_s_vt[k]]). \
                kvapply(lambda k, v: pl.LpAffineExpression(v, constant=-t_mission_needs[k]) >= 0). \
                values_tl()

        # # ##################################
        # Clusters
        # ##################################
        log.debug("constraints: clusters number")
        res_clusters = self.instance.get_cluster_candidates().list_reverse()
        # minimum availability per cluster and period
        # TODO: filter inside the function
        min_aircraft_slack = self.check_min_available(deficit_only=False)
        prev_aircraft = \
            old_info[nd.MAINT_TYPE]. \
                to_dict(None). \
                vapply(lambda v: res_clusters[v[0]]). \
                to_tuplist(). \
                to_dict(indices=[4, 2], result_col=[0, 1]). \
                vapply(len)

        t_min_aircraft_slack = sum_of_two_dicts(min_aircraft_slack, prev_aircraft)

        p_clustdate = \
            info[nd.MAINT_TYPE]. \
                to_dict(None). \
                vapply(lambda v: res_clusters[v[0]]). \
                to_tuplist(). \
                to_dict(indices=[4, 2], result_col=[0, 1]). \
                fill_with_default(t_min_aircraft_slack, default=[])

        constraints += \
            p_clustdate. \
                vapply(lambda v: pl.lpSum(assign[vv] for vv in v)). \
                kvapply(lambda k, v: v - _slack_s_kt[k] <= t_min_aircraft_slack[k]). \
                values_tl()

        log.debug("constraints: clusters hours 1")
        # Each cluster has a minimum number of usage hours to have
        # at each period.
        # TODO: filter inside the function
        min_hours_slack = \
            self.check_min_flight_hours(recalculate=False, deficit_only=False).\
            vapply(op.mul, -1)

        prevRuts_clustdate = \
            tl.TupList((r, p) for r in resources for p in periods). \
                to_dict(None). \
                vapply(lambda v: res_clusters[v[0]]). \
                to_tuplist(). \
                to_dict(result_col=0, indices=[2, 1]). \
                kvapply(lambda k, v: sum(self.get_remainingtime(p, k[1], 'rut', self.M) for p in v))

        log.debug("constraints: clusters hours 2")
        get_consum = lambda t: self.instance.data['tasks'][t]['consumption']
        row_correct = lambda tup: tup[5] + (tup[7] - tup[6])*get_consum(tup[3]) if tup[4] == nd.TASK_TYPE else tup[5]

        p_clustdate = \
            l['info'].\
                to_dict(None). \
                vapply(lambda v: res_clusters[v[0]]). \
                to_tuplist().\
                vapply(lambda v: (*v, row_correct(v))). \
                to_dict(indices=[8, 2], result_col=[0, 1, 9]). \
                fill_with_default(prevRuts_clustdate, [])

        t_min_hour_slack = sum_of_two_dicts(prevRuts_clustdate, min_hours_slack)

        log.debug("constraints: clusters hours 3")
        constraints += \
            p_clustdate. \
                kfilter(lambda k: t_min_hour_slack.get(k, 0) > 0). \
                vapply(lambda v: [(assign[vv[0], vv[1]], vv[2]) for vv in v]). \
                kvapply(lambda k, v: v + [(e, 1) for e in _slack_s_kt_h[k]]). \
                kvapply(lambda k, v: pl.LpAffineExpression(v, constant=-t_min_hour_slack[k]) >= 0). \
                values_tl()

        log.debug("constraints: maintenances")
        # maintenance capacity
        rem_maint_capacity = self.check_sub_maintenance_capacity(ref_compare=None, periods_to_check=periods)
        type_m = self.instance.get_maintenances('type')
        p_maint_used = \
            info[nd.MAINT_TYPE]. \
                vapply(lambda v: (*v, type_m[v[3]])). \
                to_dict(indices=[4, 2], result_col=[0, 1])

        constraints += \
            p_maint_used. \
                vapply(lambda v: [(assign[vv], 1) for vv in v]). \
                kvapply(lambda k, v: v + [(e, -1) for e in _slack_s_t[k[1]]]). \
                kvapply(lambda k, v: pl.LpAffineExpression(v, constant= -rem_maint_capacity[k]) <= 0). \
                values_tl()

        return objective_function, constraints


if __name__ == '__main__':
    import package.params as pm
    import data.test_data as test_d

    # data_in = test_d.dataset3_no_default()
    data_in = test_d.dataset4()
    instance = inst.Instance(data_in)
    self = GraphOriented(instance)
    _resources = self.instance.get_resources()
    self.draw_graph("12")
    # self.draw_graph("2")
    # data_graph = self.get_graph_data('1')
    # vertices = tl.TupList(data_graph['graph'].vertices()).\
    #     vapply(lambda v: data_graph['refs_inv'][v]).\
    #     vfilter(lambda v: v.rut is None and v.period=='2018-02')
    # vertices[1]
    # vertices[0]
    # vertices[1]

    # self.date_to_node(resource, '2018-12')
    date1 = '2018-02'
    # date2 = '2018-09'
    date2 = '2019-10'

    # resource = '1'
    res_patterns = sd.SuperDict()
    for resource in _resources:
        _patterns = self.get_pattern_options_from_window(resource, date1, date2)
        # res_patterns[resource] = self.filter_patterns(resource, date1, date2, patterns)
        res_patterns[resource] = _patterns

    data = self.solve_repair(res_patterns, options=pm.OPTIONS)
    data = self.solve_repair(res_patterns, options=pm.OPTIONS)
    # data.data

    # solution = self.solve(pm.OPTIONS)

    pass

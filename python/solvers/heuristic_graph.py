import pytups.superdict as sd
import pytups.tuplist as tl

import solvers.heuristics as heur
import solvers.model as mdl
import solvers.config as conf
import solvers.heuristics_maintfirst as heur_maint

import package.solution as sol
import package.instance as inst

import patterns.graphs as gg
import patterns.node as nd

import data.data_input as di

import pulp as pl
import operator as op
import random as rn
import time
import logging as log
import multiprocessing as multi
import pandas as pd
import os
import numpy as np

class GraphOriented(heur.GreedyByMission, mdl.Model):
    {'aux': {'graphs': {'RESOURCE': 'gg.DAG'}}}

    def __init__(self, instance, solution=None):

        heur.GreedyByMission.__init__(self, instance, solution)
        self.instance.data['aux']['graphs'] = sd.SuperDict()
        self.solution_store = tl.TupList()

    def initialise_graphs(self, options):
        path_cache = options.get('cache_graph_path', '')
        if path_cache and os.path.exists(path_cache):
            self.import_graph_data(path_cache)
            return
        multiproc = options['multiprocess']
        resources = self.instance.get_resources()
        if not multiproc:
            for r in resources:
                log.debug('Creating graph for resource: {}'.format(r))
                graph_data = gg.graph_factory(instance=self.instance, resource=r, options=options)
                self.instance.data['aux']['graphs'][r] = graph_data
            return
        results = sd.SuperDict()
        _data = sd.SuperDict()
        with multi.Pool(processes=multiproc) as pool:
            for r in resources:
                _instance = inst.Instance.from_instance(self.instance)
                results[r] = pool.apply_async(gg.graph_factory, [_instance, r, options])
            for r, result in results.items():
                _data[r] = result.get(timeout=10000)
        self.instance.data['aux']['graphs'] = _data
        if path_cache:
            os.mkdir(path_cache)
            self.export_graph_data(path_cache)
        return

    def sub_problem_mip(self, change, options):
        """
        always two or three phases:
        1. (optional) sample patterns
        2. get a list of patterns to apply
        3. apply patterns
        :param change: candidate
        :param options: config
        :return: a solution
        """
        patterns = self.get_patterns_from_window(change, options)
        if not patterns:
            return None
        patterns =  self.solve_repair(patterns, options)
        for p in patterns:
            self.apply_pattern(p)
        return self.solution

    def sub_problem_shortest(self, change, options):
        """
        always two or three phases:
        1. get a list of patterns to apply
        2. apply patterns
        :param change: candidate
        :param options: config
        :return: a solution
        """
        # errors.get('resources', sd.SuperDict())
        _func = lambda resource: \
            self.prepare_data_to_get_patterns(resource, change['start'], change['end'], cutoff=1)
        res_pattern = \
            sd.SuperDict().\
            fill_with_default(change['resources']).\
            kapply(_func)
        _shift = self.instance.shift_period
        _range = self.instance.get_periods_range
        for k, v in res_pattern.items():
            start = _shift(v['node1'].period_end, 1)
            end = _shift(v['node2'].period, -1)
            deleted_tasks = self.clean_assignments_window(k, start, end, 'task')
            deleted_maints = self.clean_assignments_window(k, start, end, 'state_m')
            self.update_ret_rut(k, start, end, deleted_maints)
            errors = self.check_solution(recalculate=False, assign_missions=True,
                                         list_tests=['resources', 'hours', 'capacity'])
            pattern = self.get_graph_data(k).nodes_to_pattern2(**v, errors=errors)
            self.apply_pattern(pattern)

        return self.solution

    def get_patterns_from_window(self, change, options):
        multiproc = options['multiprocess']
        resources = change['resources']
        args = (change['start'], change['end'], options['num_max'], options.get('cutoff', 200))
        if not multiproc or not options.get('multiproc_patterns', False):
            patterns = \
                sd.SuperDict().fill_with_default(change['resources']).\
                kapply(self.get_pattern_options_from_window, *args)
            return patterns.vfilter(lambda v: len(v) > 0)
        results = sd.SuperDict()
        _data = sd.SuperDict()
        with multi.Pool(processes=multiproc) as pool:
            data = {}
            for r in resources:
                data[r] = self.prepare_data_to_get_patterns(r, *args)
                # we need to filter them to take out the ones that compromise the post-window periods
                results[r] = pool.apply_async(self.get_graph_data(r).nodes_to_patterns2, kwds=data[r])
            for r, result in results.items():
                try:
                   _data[r] = result.get(timeout=100)
                except multi.TimeoutError:
                    _data[r] = []
        return _data.vfilter(lambda v: len(v) > 0)

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
        big_window = options.get('big_window', False)

        options_repair = di.copy_dict(options)
        options_repair = sd.SuperDict(options_repair)
        options_repair['timeLimit'] = options.get('timeLimit_cycle', 10)

        max_iters_initial = options.get('max_iters_initial', 10)
        max_patterns_initial = options.get('max_patterns_initial', 0)
        timeLimit_initial = options.get('timeLimit_initial', options_repair['timeLimit'])

        # set clock!
        time_init = time.time()

        # initialise logging, seed
        self.set_log_config(options)
        self.initialise_seed(options)
        log.info("Initialise graphs")
        self.initialise_graphs(options)

        # 1. get an initial solution.
        log.info("Initial solution.")
        initial_opts = dict(max_iters=max_iters_initial,
                            assign_missions=True,
                            num_max=max_patterns_initial,
                            timeLimit=timeLimit_initial)
        options_fs = {**options_repair, **initial_opts}
        if self.solution is None or max_iters_initial:
            first_solve = heur_maint.MaintenanceFirst(self.instance, self.solution)
            first_solve.get_objective_function = self.get_objective_function
            first_solve.solve(options_fs)
            self.set_solution(first_solve.best_solution)
        elif max_patterns_initial:
            ch = self.get_candidate_all()
            patterns = self.get_patterns_from_window(ch, options_fs)
            patterns = self.solve_repair(patterns, options_fs)
            for p in patterns:
                self.apply_pattern(p)

        # initialise solution status
        self.initialise_solution_stats()

        # 2. repair solution
        log.info("Solving phase.")
        i = 0
        errors = sd.SuperDict()
        while i < max_iters:

            # choose a subproblem for the iteration
            subproblem, sub_options = self.choose_subproblem(options)
            if rn.random() > 0.5 or not errors:
                change = self.get_candidate_random(sub_options)
            else:
                func_options = sd.SuperDict(
                    resources=self.get_candidates_tasks,
                     hours=self.get_candidates_cluster,
                     capacity=self.get_candidate_capacity
                     )
                options_filtered = func_options.keys_tl().intersect(errors).sorted()
                opt = rn.choice(options_filtered)
                change = func_options[opt](errors, sub_options)
            if big_window:
                change = self.get_candidate_all()
            if not change:
                continue
            log.info('Repairing periods {start} => {end} for resources: {resources}'.format(**change))
            solution = subproblem(change, sub_options)
            if solution is None:
                continue
            kwargs = dict(assign_missions=True, list_tests=['resources', 'hours', 'capacity'])
            objective, status, errors = self.analyze_solution(temperature, **kwargs)
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
            self.solution_store.append(self.copy_solution(exclude_aux=True))

            log.info("time={}, iteration={}, temperaure={}, current={}, best={}, errors={}".
                     format(round(time_now), i, round(temperature, 4), objective,
                            self.best_objective, num_errors))
            i += 1

            if not self.best_objective or i >= max_iters or time_now > max_time:
                break

        return sol.Solution(self.best_solution)

        pass

    def choose_subproblem(self, options):
        sb_func = dict(mip=self.sub_problem_mip, short=self.sub_problem_shortest)
        subproblem_choice = options.get('subproblem')
        if not subproblem_choice:
            return self.sub_problem_mip, options
        sb_prob = sd.SuperDict(subproblem_choice).get_property('weight').to_tuplist().sorted()
        _opts, _probs = zip(*sb_prob)
        choice = rn.choices(_opts, _probs)[0]
        return sb_func[choice], {**options, **subproblem_choice[choice]}

    def get_graph_data(self, resource):
        return self.instance.data['aux']['graphs'][resource]

    def export_graph_data(self, path):
        instance = self.instance
        for r in instance.get_resources():
            self.get_graph_data(r).to_file(path=path)

    def import_graph_data(self, path):
        raise NotImplementedError("Needs to be adapted to graph from_file")
        # instance = self.instance
        # resources = self.instance.get_resources()
        # source = resources.kapply(lambda k: nd.get_source_node(instance, k))
        # sink = resources.kapply(lambda k: nd.get_sink_node(instance, k))
        # sink_source = \
        #     sd.SuperDict(source=source, sink=sink).\
        #     to_dictup().\
        #     to_tuplist().\
        #     to_dict(result_col=2, indices=[1, 0], is_list=False).\
        #     to_dictdict()
        # data = \
        #     resources.\
        #     kapply(lambda k: cp.from_file(path=path, resource=k))
        # format_node = lambda res, v: sd.SuperDict(v)._update(dict(instance=instance, resource=res))
        # for res in resources:
        #     keys = tl.TupList(data[res]['refs_inv']).take(0)
        #     values = tl.TupList(data[res]['refs_inv']).take(1).vapply(lambda v: format_node(res, v)).vapply(lambda v: nd.Node(**v))
        #     data[res]['refs_inv'] = sd.SuperDict(zip(keys, values))
        #     data[res]['refs'] = sd.SuperDict(zip(values, keys))
        #
        # data.update(sink_source)
        # self.instance.data['aux']['graphs'] = data

    def get_objective_function(self, error_cat=None):
        """
        Calculates the objective function for the current solution.

        :param sd.SuperDict error_cat: possibility to take a cache of errors
        :return: objective function
        :rtype: int
        """
        if error_cat is None:
            error_cat = self.check_solution(list_tests=['resources', 'hours', 'capacity']).to_lendict()
        weights = sd.SuperDict(resources=10000, hours=1000, capacity=10000)
        # TODO: there is an issue with elapsed being sometimes 0 in the last period
        a = {'elapsed', 'usage', 'dist_maints'} & error_cat.keys()
        num_errors = sum((error_cat*weights).values())

        # we count the number of maintenances and their distance to the end
        first, last = self.instance.get_first_last_period()
        _dist = self.instance.get_dist_periods
        maintenances = \
            self.get_maintenance_periods(). \
                take(1). \
                vapply(_dist, last)

        return num_errors * 1000 + sum(maintenances)

    def filter_node2(self, node2):
        """

        :param node2: node2 of candidate
        :return: a function that returns True if a node complies with node2 rut and ret constraints
        """
        rut = self.get_remainingtime(node2.resource, node2.period_end, 'rut', maint=self.M)
        if rut is None:
            # it could be we are at the dummy node at the end.
            # here, we would not care about filtering
            return None
        ret = self.get_remainingtime(node2.resource, node2.period_end, 'ret', maint=self.M)
        first, last = self.instance.get_first_last_period()
        _shift = self.instance.shift_period
        size = self.instance.get_maintenances('elapsed_time_size')[self.M]
        next_maint = self.get_next_maintenance(node2.resource, _shift(node2.period_end, 1), {'M'})
        if next_maint is None:
            _period_to_look = last
        else:
            # If there is a next maintenances, we filter patterns depending on the last rut
            _period_to_look = _shift(next_maint, -1)
        ret_cycle = self.get_remainingtime(node2.resource, _period_to_look, 'ret', maint=self.M)
        rut_cycle = self.get_remainingtime(node2.resource, _period_to_look, 'rut', maint=self.M)
        min_ret = ret - ret_cycle
        if next_maint is None:
            # if there is no maintenance later, we do not care about time
            max_ret = self.instance.get_maintenances('max_elapsed_time')[self.M]
        else:
            # if there is a maintenance later, we cannot make them too close
            max_ret = min_ret + size
        min_rut = rut - rut_cycle
        _func = lambda node: node.rut[self.M] >= min_rut and \
                             min_ret <= node.ret[self.M] <= max_ret
        return _func

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
        self.update_ret_rut(resource, first_date_to_update, end, added_maints + deleted_maints)
        return True

    def update_ret_rut(self, resource, start, end, modified_mants):
        times = ['rut']
        if modified_mants:
            times.append('ret')
        for t in times:
            for m in self.instance.get_rt_maints(t):
                self.set_remaining_time_in_window(resource, m, start, end, t)

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

    def date_to_state(self, resource, date):

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
        # we return the state of type: period, period_end, assignment, type, rut, ret
        return sd.SuperDict(period=period_start, period_end=period_end,
                             assignment=assignment, type=_type, **_rts)

    def prepare_data_to_get_patterns(self, resource, date1, date2, num_max=10000, cutoff=None, **kwargs):
        node1 = nd.Node.from_state(self.instance, resource=resource,
                                   state=self.date_to_state(resource, date1))
        node2_data = self.date_to_state(resource, date2)
        node2 = nd.Node.from_state(self.instance, resource=resource, state=node2_data)
        mask = self.filter_node2(node2)
        node2_data = {**node2_data, **sd.SuperDict(rut=None, ret=None)}
        dummy_node2  = nd.Node.from_state(self.instance, resource=resource, state=node2_data)
        if cutoff is None:
            min_cutoff = self.get_graph_data(resource).shortest_path(node1=node1, node2=dummy_node2)
            max_cutoff = self.instance.get_dist_periods(date1, date2) + 1
            max_cutoff = max(min_cutoff, max_cutoff)
            cutoff = rn.choice(range(min_cutoff, max_cutoff+1))
        return dict(node1=node1, node2=dummy_node2, max_paths=num_max, cutoff=cutoff, mask=mask, **kwargs)

    def get_pattern_options_from_window(self, resource, date1, date2, num_max=10000, cutoff=None, **kwargs):
        """
        This method is accessed by the repairing method.
        To know the options of assignments

        :param resource:
        :param date1:
        :param date2:
        :return:
        """
        log.debug("resource {}".format(resource))
        data = self.prepare_data_to_get_patterns(resource, date1, date2, num_max, cutoff, **kwargs)
        patterns = self.get_graph_data(resource).nodes_to_patterns2(**data)
        return patterns

    def get_pattern_from_window(self, resource, date1, date2):
        _range = self.instance.get_periods_range
        nodes = \
            tl.TupList([self.date_to_state(resource, p) for p in _range(date1, date2)]).\
                vapply(lambda v: nd.Node.from_state(self.instance, resource, state=v))
        return nodes.unique2().sorted(key=lambda k: k.period)

    def set_remaining_time_in_window(self, resource, maint, start, end, rt):
        _shift = self.instance.shift_period
        _start = start
        maint_data = self.instance.data['maintenances'][maint]
        duration = maint_data['duration_periods']
        horizon_end = self.instance.get_param('end')
        affected_maints = maint_data['affects']
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
            maint_end = min(self.instance.shift_period(maint_start, duration - 1), horizon_end)
            periods_maint = self.instance.get_periods_range(maint_start, maint_end)
            for m in affected_maints:
                self.update_time_maint(resource, periods_maint, time=rt, maint=m)
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
        fixed_periods = self.instance.get_fixed_periods(resource=resource).to_set()

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

        resources = res_patterns.keys_tl()
        resource = resources[0]
        example_pattern = res_patterns[resource][0]
        start = example_pattern[0].period
        end = example_pattern[-1].period
        old_info = tl.TupList()
        for r in resources:
            for key in ['task', 'state_m']:
                old_info += self.get_assignments_window(resource=r, start=start, end=end, key=key)

        InfoOld_type = \
            old_info. \
            to_dict(result_col=[0, 1, 2, 3]). \
            fill_with_default([nd.MAINT_TYPE, nd.TASK_TYPE], default=tl.TupList())

        combos = get_patterns_into_dictup(res_patterns)

        info = get_assignments_from_patterns(self.instance, combos, self.M)

        info_type = \
            info. \
                to_dict(result_col=[0, 1, 2, 3], indices=[4]). \
                fill_with_default([nd.MAINT_TYPE, nd.TASK_TYPE], default=tl.TupList())

        first, last = self.instance.get_start_end()
        l = self.domains = mdl.Model.get_domains_sets(self, options, force=False)
        l['combos'] = sd.SuperDict(combos)
        l['info'] = info
        l['info_type'] = info_type
        l['info_old'] = old_info
        l['infoOld_type'] = InfoOld_type
        l['periods_sub'] = info.take(2).unique().vfilter(lambda v: first <= v <= last).sorted()
        return self.domains

    def solve_repair(self, res_patterns, options):

        # res_patterns is a sd.SuperDict of resource: [list of possible patterns]
        log.debug("Building domains.")
        l = self.get_domains_sets_patterns(res_patterns=res_patterns, options=options, force=True)

        # Variables:
        log.debug("Building variables.")
        model_vars = self.build_vars()

        if options.get('mip_start'):
            log.debug("Filling initial solution.")
            self.fill_initial_solution(model_vars)
        vars_to_fix = options.get('fix_vars', [])
        if vars_to_fix:
            log.debug("Fixing variables.")
            self.fix_variables(model_vars.filter(vars_to_fix, check=False))
            if 'assign' in vars_to_fix:
                model_vars['assign'] = model_vars['assign'].vfilter(lambda v: v.value())
                selected_assign = model_vars['assign'].keys_tl().to_set()
                self.domains['info'] = self.domains['info'].vfilter(lambda v: (v[0], v[1]) in selected_assign)
                self.domains['combos'] = self.domains['combos'].filter(selected_assign)
                self.domains['info_type'] = \
                    self.domains['info']. \
                    to_dict(result_col=[0, 1, 2, 3], indices=[4]). \
                    fill_with_default([nd.MAINT_TYPE, nd.TASK_TYPE], default=tl.TupList())

        # We all constraints at the same time:
        log.debug("Building model.")
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
            import datetime as dt
            model.writeLP(filename=options['path'] + 'formulation_{}.lp'.
                          format(dt.datetime.now().strftime("%Y%m%d%H%MT%S")))
            return None
        print('model solved correctly')
        # tl.TupList(model.variables()).to_dict(None).vapply(pl.value).vfilter(lambda v: v)

        # TODO: we should return the mapping: resource => pattern or list of patterns
        # TODO: we should plug and play the subproblem as another class
        return self.get_repaired_solution(l['combos'], model_vars['assign'])

    def get_repaired_solution(self, combos, assign):
        patterns = assign. \
            vapply(pl.value). \
            vfilter(lambda v: v). \
            kapply(lambda k: combos[k]). \
            values_tl()
        return patterns

    def draw_graph(self, resource, **kwargs):
        self.get_graph_data(resource).draw(**kwargs)
        return True

    def get_window_size(self, options):
        horizon_length = self.instance.get_param('num_period')
        # the horizon includes the fake first and last periods
        # so we add two to the size
        max_size = options.get('max_window_size', horizon_length) + 2
        min_size = min(options.get('min_window_size', 10), max_size)
        return rn.choice(range(min_size, max_size + 1))

    def get_window_from_dates(self, periods, options):
        _shift = self.instance.shift_period
        _range = self.instance.get_periods_range
        _first = min(periods)
        _last = max(periods)
        window_size = self.get_window_size(options)
        first, last = self.instance.get_first_last_period()
        last_plus_one = _shift(last, 1)
        first_minus_one = _shift(first, -1)
        first_option = max(_shift(_first, -window_size), first_minus_one)
        last_option = max(_shift(_last, -window_size), first_minus_one)
        start = rn.choice(_range(first_option, last_option))
        end = min(_shift(start, window_size), last_plus_one)
        return start, end

    def get_subfleet_from_list(self, resources, options):
        size = self.get_subfleet_size(options)
        extra_needed = size - len(resources)
        if extra_needed < 0:
            return rn.sample(resources, size)
        remaining = self.instance.get_resources().keys_tl().set_diff(resources).sorted()
        extra = rn.sample(remaining, extra_needed)
        return resources + extra

    def get_subfleet_size(self, options):
        fleet_size = len(self.instance.get_resources())
        max_size = options.get('max_candidates', fleet_size)
        min_size = min(options.get('min_candidates', 10), max_size)
        return rn.choice(range(min_size, max_size + 1))

    def get_candidates_tasks(self, errors, options):
        """
        :return: a list of resources, a start time and an end time.
        :rtype: sd.SuperDict
        """
        tasks_probs = \
            errors['resources'].\
            to_tuplist().to_dict(result_col=2, indices=[0]).vapply(sum).to_tuplist().sorted()
        _tasks, _probs = zip(*tasks_probs)
        task = rn.choices(_tasks, weights=_probs, k=1)[0]
        # we choose dates
        periods_t = self.instance.get_task_period_list(True)
        start, end = self.get_window_from_dates(periods_t[task], options)
        # we choose a subfleet
        t_cand = self.instance.get_task_candidates()[task]
        res_to_change = self.get_subfleet_from_list(t_cand, options)
        return sd.SuperDict(resources=res_to_change, start=start, end=end)

    def get_candidate_random(self, options):
        # dates
        periods = self.instance.get_periods()
        start, end = self.get_window_from_dates(periods, options)
        # we choose a subset of resources
        resources = self.instance.get_resources().keys_tl().sorted()
        res_to_change = self.get_subfleet_from_list(resources, options)
        return sd.SuperDict(resources=res_to_change, start=start, end=end)

    def get_candidate_all(self):
        instance = self.instance
        first, last = instance.get_first_last_period()
        resources = instance.get_resources().keys_tl().sorted()
        _shift = self.instance.shift_period
        return sd.SuperDict(resources=resources, start=_shift(first, -1), end=_shift(last))

    def get_candidate_capacity(self, errors, options):
        cap_periods = errors['capacity'].keys_tl().take(1)
        start, end = self.get_window_from_dates(cap_periods, options)
        resources = self.instance.get_resources().keys_tl().sorted()
        res_to_change = self.get_subfleet_from_list(resources, options)
        return sd.SuperDict(resources=res_to_change, start=start, end=end)

    def get_candidates_cluster(self, errors, options):
        """
        :return: a list of candidates [(aircraft, period), (aircraft2, period2)] to free
        :rtype: tl.TupList
        """
        _dist = self.instance.get_dist_periods
        periods_cluster = errors['hours'].keys_tl().to_dict(1).vapply(sorted)
        # we choose a cluster
        opt_probs = periods_cluster.vapply(len).to_tuplist()
        _cluts, _probs = zip(*opt_probs)
        cluster = rn.choices(_cluts, weights=_probs, k=1)[0]
        # we choose dates
        start, end = self.get_window_from_dates(periods_cluster[cluster], options)
        # we choose a subfleet
        c_cand = self.instance.get_cluster_candidates()[cluster]
        res_to_change = self.get_subfleet_from_list(c_cand, options)
        return sd.SuperDict(resources=res_to_change, start=start, end=end)

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

    def get_constraints_maints(self, assign, _slack_s_t):

        l = self.domains
        periods = l['periods_sub']
        old_info = l['infoOld_type']
        info = l['info_type']
        rem_maint_capacity = self.check_sub_maintenance_capacity(ref_compare=None, periods_to_check=periods)
        type_m = self.instance.get_maintenances('type')

        prevMaints_usage = \
            old_info[nd.MAINT_TYPE]. \
                vapply(lambda v: (*v, type_m[v[3]])). \
                to_dict(indices=[4, 2]).vapply(len)
        t_max_maint_slack = sum_of_two_dicts(prevMaints_usage, rem_maint_capacity)
        p_maint_used = \
            info[nd.MAINT_TYPE]. \
                vapply(lambda v: (*v, type_m[v[3]])). \
                to_dict(indices=[4, 2], result_col=[0, 1])

        return \
            p_maint_used. \
                vapply(lambda v: [(assign[vv], 1) for vv in v]). \
                kvapply(lambda k, v: v + _slack_s_t[k[1]]). \
                kvapply(lambda k, v: pl.LpConstraint(v, rhs= t_max_maint_slack[k], sense=pl.LpConstraintLE)). \
                values_tl()

    def get_constraints_tasks(self, assign, _slack_s_vt):

        # num resources:
        l = self.domains
        periods = l['periods_sub']
        old_info = l['infoOld_type']
        info = l['info_type']
        mission_needs = self.check_task_num_resources(deficit_only=False, periods=periods)

        prev_mission_needs = \
            old_info[nd.TASK_TYPE]. \
            to_dict(indices=[3, 2], result_col=[0, 1]). \
            vapply(len)
        t_mission_needs = sum_of_two_dicts(mission_needs, prev_mission_needs)

        p_mission = \
            info[nd.TASK_TYPE]. \
                to_dict(indices=[3, 2], result_col=[0, 1])

        return \
            p_mission. \
                kfilter(lambda k: t_mission_needs.get(k, 0) > 0). \
                kvapply(lambda k, v: [(assign[vv], 1) for vv in v] + _slack_s_vt[k]). \
                kvapply(lambda k, v: pl.LpConstraint(v, rhs=t_mission_needs[k], sense=pl.LpConstraintGE)). \
                values_tl()

    def get_constraints_number(self, assign, _slack_s_kt):

        # minimum availability per cluster and period
        l = self.domains
        periods = l['periods_sub']
        old_info = l['infoOld_type']
        info = l['info_type']
        res_clusters = self.instance.get_cluster_candidates().list_reverse()

        min_aircraft_slack = self.check_min_available(deficit_only=False, periods=periods)
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

        return \
            p_clustdate. \
                vapply(lambda v: pl.lpSum(assign[vv] for vv in v)). \
                kvapply(lambda k, v: v - _slack_s_kt[k] <= t_min_aircraft_slack[k]). \
                values_tl()

    # @profile
    def get_constraints_hours_df(self, assign, _slack_s_kt_h):

        log.debug("constraints: clusters hours 1")
        l = self.domains
        periods = l['periods_sub']
        combos = l['combos']
        resources = combos.keys_tl().take(0).unique().sorted()
        res_clusters = self.instance.get_cluster_candidates().list_reverse()

        # Each cluster has a minimum number of usage hours to have
        # at each period.
        min_hours_slack = \
            self.check_min_flight_hours(recalculate=False, deficit_only=False, periods=periods).\
            vapply(op.mul, -1)

        prevRuts_clustdate = \
            tl.TupList((r, p) for r in resources for p in periods). \
                to_dict(None). \
                vapply(lambda v: res_clusters[v[0]]). \
                to_tuplist(). \
                to_dict(result_col=0, indices=[2, 1]). \
                kvapply(lambda k, v: sum(self.get_remainingtime(p, k[1], 'rut', self.M) for p in v))

        t_min_hour_slack = sum_of_two_dicts(prevRuts_clustdate, min_hours_slack).vfilter(lambda v: v >0)
        consumption = self.instance.get_tasks('consumption').to_tuplist().to_list()
        _slack_s_kt_h_pd = pd.Series(data=_slack_s_kt_h.values_l(),
                                     index=pd.MultiIndex.from_tuples(_slack_s_kt_h.keys_l(),
                                                                     names=['clust', 'period'])).rename('slack')
        t_min_hour_slack_pd = \
            pd.Series(data = t_min_hour_slack.values_l(),
                      index=pd.MultiIndex.from_tuples(t_min_hour_slack.keys_l(),
                                                      names=['clust', 'period']))\
                .rename('min_hour')
        assign_pd = pd.DataFrame.from_records(assign.to_tuplist().to_list(),
                                              columns=['res', 'pat', '_var'])
        consumption_pd = pd.DataFrame.from_records(consumption, columns=['assign', 'consum'])

        res_clusters_tl = res_clusters.to_tuplist().to_list()
        res_clusters_pd = pd.DataFrame(res_clusters_tl, columns=['res', 'clust'])

        log.debug("constraints: clusters hours 2")
        info = l['info']
        varList_grouped = self.get_constraints_hours_df_step2(
            assign_pd, consumption_pd, info.to_list(), res_clusters_pd
        )
        # cache = varList_grouped.to_dict()
        # cache = sd.SuperDict(cache).to_tuplist().to_set()
        #
        # _path = '/home/pchtsp/Downloads/cache/'
        # assign_pd.to_csv(_path + 'assign_pd.csv', index=False)
        # consumption_pd.to_csv(_path + 'consumption_pd.csv', index=False)
        # res_clusters_pd.to_csv(_path + 'res_clusters_pd.csv', index=False)
        # info.to_csv(_path + 'info.csv')
        # t_min_hour_slack_pd.to_csv(_path + 't_min_hour_slack.csv', header=True)
        # _slack_s_kt_h_pd.apply(pd.Series).stack().apply(lambda v: v[0]).to_csv(_path + 'slack_s_kt_h.csv', header=True)

        # result = varList_grouped.to_dict()
        # result = sd.SuperDict(result).to_tuplist().to_set()
        # dif = result ^ cache

        # #
        log.debug("constraints: clusters hours 3")

        final_table = pd.concat([varList_grouped, t_min_hour_slack_pd, _slack_s_kt_h_pd], axis=1, join='inner')
        final_table['vars'] = final_table.var_rut + final_table.slack

        result = final_table.apply(
            lambda x: pl.LpConstraint(x.vars, rhs=x.min_hour, sense=pl.LpConstraintGE),
            axis=1)
        return result.to_dict()

    def get_constraints_hours_df_step2(self, assign_pd, consumption_pd, info, res_clusters_pd):
        ff = pd.DataFrame(info,
                          columns=['res', 'pat', 'period', 'assign', 'type', 'rut', 'pos', 'tot'])
        merged_data = ff[~ff.rut.isna()]. \
            merge(res_clusters_pd, on='res'). \
            merge(consumption_pd, on='assign', how='left'). \
            merge(assign_pd, on=['res', 'pat'], how='left')
        merged_data.loc[merged_data.type == nd.TASK_TYPE, 'rut'] = merged_data.rut + (
                    merged_data.tot - merged_data.pos) * merged_data.consum
        merged_data = merged_data.loc[merged_data.rut > 0]
        merged_data['var_rut'] = list(zip(merged_data._var, merged_data.rut))
        varList_grouped = merged_data.groupby(['clust', 'period'])['var_rut'].apply(list)
        return varList_grouped

    def get_constraints_hours(self, assign, _slack_s_kt_h):

        log.debug("constraints: clusters hours 1")
        l = self.domains
        periods = l['periods_sub']
        combos = l['combos']
        resources = combos.keys_tl().take(0).unique().sorted()
        res_clusters = self.instance.get_cluster_candidates().list_reverse()

        # Each cluster has a minimum number of usage hours to have
        # at each period.
        min_hours_slack = \
            self.check_min_flight_hours(recalculate=False, deficit_only=False, periods=periods).\
            vapply(op.mul, -1)

        prevRuts_clustdate = \
            tl.TupList((r, p) for r in resources for p in periods). \
                to_dict(None). \
                vapply(lambda v: res_clusters[v[0]]). \
                to_tuplist(). \
                to_dict(result_col=0, indices=[2, 1]). \
                kvapply(lambda k, v: sum(self.get_remainingtime(p, k[1], 'rut', self.M) for p in v))

        t_min_hour_slack = sum_of_two_dicts(prevRuts_clustdate, min_hours_slack).vfilter(lambda v: v >0)

        log.debug("constraints: clusters hours 2")
        get_consum = lambda t: self.instance.data['tasks'][t]['consumption']
        row_correct = lambda tup: tup[5] + (tup[7] - tup[6]) * get_consum(tup[3]) if tup[4] == nd.TASK_TYPE else tup[5]
        p_clustdate = \
            l['info']. \
                vfilter(lambda v: v[5] is not None). \
                to_dict(None). \
                vapply(lambda v: res_clusters[v[0]]). \
                to_tuplist().\
                vapply(lambda v: (*v, row_correct(v))). \
                vfilter(lambda v: v[9] > 0). \
                to_dict(indices=[8, 2], result_col=[0, 1, 9])

        log.debug("constraints: clusters hours 3")
        return \
            p_clustdate. \
                kfilter(lambda k: t_min_hour_slack.get(k, 0) > 0). \
                kvapply(lambda k, v: [(assign[vv[0], vv[1]], vv[2]) for vv in v] + _slack_s_kt_h[k]). \
                kvapply(lambda k, v: pl.LpConstraint(v, rhs=t_min_hour_slack[k], sense=pl.LpConstraintGE))

    def build_model(self, assign, slack_vts, slack_ts, slack_kts, slack_kts_h):

        l = self.domains
        first, last = self.instance.get_first_last_period()
        combos = l['combos']
        _dist = self.instance.get_dist_periods
        p_s = {s: p for p, s in enumerate(l['slots'])}

        p_t = self.instance.data['aux']['period_i']
        weight_pattern = \
            lambda pattern: \
                sum(_dist(elem.period, last) for elem in pattern
                    if elem.type == nd.MAINT_TYPE)
        assign_p = combos.vapply(weight_pattern)

        def sum_of_slots(variable_slots, coef=1):
            result_col = len(variable_slots.keys_l()[0])
            indices = range(result_col - 1)
            return \
                variable_slots. \
                    to_tuplist(). \
                    to_dict(result_col=result_col, indices=indices). \
                    vapply(lambda v: zip(v, [coef]*len(v))).\
                    vapply(list)

        _slack_s_kt = sum_of_slots(slack_kts)
        slack_kts_p = {(k, t, s): (p_s[s] + 1 - 0.001 * p_t[t]) * 50 for k, t, s in l['kts']}

        _slack_s_kt_h = sum_of_slots(slack_kts_h)
        slack_kts_h_p = {(k, t, s): (p_s[s] + 2 - 0.001 * p_t[t]) ** 2 for k, t, s in l['kts']}

        _slack_s_t = sum_of_slots(slack_ts, coef=-1)
        slack_ts_p = {(t, s): (p_s[s] + 1 - 0.001 * p_t[t]) * 1000 for t, s in l['ts']}

        _slack_s_vt = sum_of_slots(slack_vts)
        slack_vts_p = slack_vts.kapply(lambda vts: (p_s[vts[2]] + 1) * 10000)

        # Constraints
        constraints = tl.TupList()

        # objective:
        objective_function = \
            pl.lpSum((assign * assign_p).values_tl()) + \
            1000*pl.lpSum((slack_vts * slack_vts_p).values_tl()) + \
            10*pl.lpSum((slack_ts * slack_ts_p).values_tl()) + \
            10*pl.lpSum((slack_kts * slack_kts_p).values_tl()) + \
            10*pl.lpSum((slack_kts_h * slack_kts_h_p).values_tl())

        # one pattern per resource:
        constraints += \
            combos.keys_tl(). \
                vapply(lambda v: (v[0], *v)).to_dict(result_col=[1, 2]). \
                vapply(lambda v: ((assign[vv], 1) for vv in v)). \
                vapply(lambda v: pl.LpConstraint(v, rhs=1, sense=pl.LpConstraintEQ)). \
                values_tl()

        # ##################################
        # Tasks and tasks starts
        # ##################################
        log.debug("constraints: tasks")
        constraints += self.get_constraints_tasks(assign, _slack_s_vt)

        # # ##################################
        # Clusters
        # ##################################
        log.debug("constraints: clusters number")
        constraints += self.get_constraints_number(assign, _slack_s_kt)

        log.debug("constraints: clusters hours")
        cons = self.get_constraints_hours_df(assign, _slack_s_kt_h)
        # cons2 = self.get_constraints_hours_df(assign, _slack_s_kt_h)
        # a_c = sd.SuperDict(cons).vapply(lambda v: ('c', v.constant)).to_tuplist().to_set()
        # b_c = sd.SuperDict(cons2).vapply(lambda v: ('c', v.constant)).to_tuplist().to_set()
        # aa = sd.SuperDict.from_dict(cons).to_dictdict().to_dictup().to_tuplist().to_set() | a_c
        # bb = sd.SuperDict.from_dict(cons2).to_dictdict().to_dictup().to_tuplist().to_set() | b_c
        # dif = aa ^ bb
        # aa - bb
        # bb - aa
        constraints += list(cons.values())

        log.debug("constraints: maintenances")
        # maintenance capacity
        constraints += self.get_constraints_maints(assign, _slack_s_t)

        return objective_function, constraints


def sum_of_two_dicts(dict1, dict2):
    return \
        (dict1.keys_tl() + dict2.keys_tl()). \
            to_dict(None). \
            kapply(lambda k: dict1.get(k, 0) + dict2.get(k, 0))


def get_patterns_into_dictup(res_patterns):
    return \
        {(res, p): pattern
         for res, _pat_list in res_patterns.items()
         for p, pattern in enumerate(_pat_list)
         }


def get_assignments_from_patterns(instance, combos, maint='M'):
    _range = instance.get_periods_range
    _dist = instance.get_dist_periods
    rut_or_None = lambda rut: rut[maint] if rut is not None else None
    _range_backup = sd.SuperDict(combos).to_tuplist().\
        take(2).unique2().\
        vapply(lambda v: (v.period, v.period_end)).unique2().\
        to_dict(None).\
        vapply(lambda v: _range(*v))

    info = tl.TupList(
        (res, p, period, e.assignment, e.type, rut_or_None(e.rut), pos, _dist(e.period, e.period_end))
        for (res, p), pattern in combos.items()
        for e in pattern for pos, period in enumerate(_range_backup[e.period, e.period_end])
    )
    return info


if __name__ == '__main__':

    pass

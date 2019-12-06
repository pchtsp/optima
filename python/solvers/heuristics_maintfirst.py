import pytups.superdict as sd
import pytups.tuplist as tl
import os
import package.solution as sol
import data.data_input as di
import numpy as np
import solvers.heuristics as heur
import random as rn
import math
import time
import logging as log


class MaintenanceFirst(heur.GreedyByMission):

    # statuses for detecting a new worse solution
    status_worse = {2, 3}
    # statuses for detecting if we accept a solution
    status_accept = {0, 1, 2}

    def __init__(self, instance, solution=None):

        super().__init__(instance, solution)

        pass

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
        #  we add to num_errors the number of non empty assignments.
        all_maints = self.solution.data['state_m']
        num_periods_maint = all_maints.to_lendict().values()
        num_errors += sum(num_periods_maint)/10
        # finally, we add as a less important criteria:
        # the time between each VS maintenance and the end period.
        instance = self.instance
        # TODO: Hard coding of VS
        VS_maints = all_maints.to_dictup().to_tuplist().vfilter(lambda v: v[2]=='VS')
        first, last = instance.get_first_last_period()
        _dist = lambda v: instance.get_dist_periods(v, last)
        sum_of_dates = VS_maints.vapply(lambda v: _dist(v[1]))
        scale = instance.data['resources'].len()*_dist(first)
        num_errors += sum(sum_of_dates)/scale

        return num_errors

    @staticmethod
    def set_log_config(options):
        """
        Sets logging according to options
        :param options: options dictionary
        :return: None
        """
        level = log.INFO
        if options.get('debug', False):
            level = log.DEBUG
        logFile = os.path.join(options.get('path'), 'output.log')
        logFormat = '%(asctime)s %(levelname)s:%(message)s'
        open(logFile, 'w').close()
        formatter = log.Formatter(logFormat)

        # to file:
        file_log_handler = log.FileHandler(logFile, 'a')
        file_log_handler.setFormatter(formatter)

        # to command line
        stderr_log_handler = log.StreamHandler()
        stderr_log_handler.setFormatter(formatter)

        outputs = {'file': file_log_handler, 'console': stderr_log_handler}
        output_choices = options.get('log_output', ['file'])

        _log = log.getLogger()
        _log.handlers = [v for k, v in outputs.items() if k in output_choices]

        # option to add a custom handler:
        custom_handler = options.get('log_handler')
        if custom_handler:
            custom_handler.setFormatter(formatter)
            _log.handlers.append(custom_handler)

        _log.setLevel(level)

    def initialise_solution_stats(self):
        """
        Initializes caches of best and past solution
        :return: None
        """
        self.previous_solution = self.best_solution = self.solution.data
        errs = self.check_solution()
        error_cat = errs.to_lendict()
        self.prev_objective = self.get_objective_function(error_cat)
        self.best_objective = self.prev_objective

    def initialise_seed(self, options):
        seed = options.get('seed')
        if not seed:
            seed = rn.random()*10000
            options['seed'] = seed
        rn.seed(int(seed))
        np.random.seed(int(seed))

    def solve(self, options):
        """
        Solves an instance using the metaheuristic.

        :param options: dictionary with options
        :return: solution to the planning problem
        :rtype: :py:class:`package.solution.Solution`
        """

        self.options = options

        # extract all config from options
        max_iters = options.get('max_iters', 99999999)
        max_time = options.get('timeLimit', 600)
        cooling = options.get('cooling', 0.995)
        num_change_probs = options.get('num_change', [1])
        temperature = options.get('temperature', 1)
        assign_missions = options.get('assign_missions', False)
        num_change_pos = [n + 1 for n, _ in enumerate(num_change_probs)]

        # initialise logging, seed and solution status
        self.set_log_config(options)
        self.initialise_solution_stats()
        self.initialise_seed(options)

        time_init = time.time()
        maints = self.instance.get_maintenances('priority')

        # M maintenances should always be first
        maints, priori = zip(*sorted(maints.items(), key=lambda x: x[1]))

        i = 0
        while True:
            # 1. assign maints for all aircraft around horizon limits
            for m in maints:
                self.assign_missing_maints_to_aircraft(m)

            if assign_missions:
                # 2. assign missions for all aircraft
                self.assign_missions()
                # 3. try to correct min_assign
                self.over_assign_missions()

            # check quality of solution, store best.
            objective, status, errors = self.analyze_solution(temperature, assign_missions)
            num_errors = errors.to_lendict().values_tl()
            num_errors = sum(num_errors)

            # 3. check if feasible. If not, un-assign (some/ all) or move maints
            num_change = rn.choices(num_change_pos, num_change_probs)[0]
            candidates = self.get_candidates(k=num_change, errors=errors)
            for candidate in candidates:
                self.free_resource(candidate)

            # sometimes, we go back to the best solution found
            if objective > self.best_objective and rn.random() < 0.01:
                self.set_solution(self.best_solution)
                objective = self.prev_objective = self.best_objective
                log.info('back to best solution: {}'.format(self.best_objective))

            if status in self.status_worse:
                temperature *= cooling
            clock = time.time()
            time_now = clock - time_init

            log.info("time={}, iteration={}, temperaure={}, current={}, best={}, errors={}".
                     format(round(time_now), i, round(temperature, 4), objective, self.best_objective, num_errors))
            i += 1

            if not self.best_objective or i >= max_iters or time_now > max_time:
                break

        return sol.Solution(self.best_solution)

    def analyze_solution(self, temperature, assign_missions=False):
        """
        Compares solution quality with previous and best.
        Updates the previous solution (always).
        Updates the current solution based on acceptance criteria (Simulated Annealing)
        Updates the best solution when relevant.

        :return: (status int, error dictionary)
        :rtype: tuple
        """
        # This commented function validates if I'm updating correctly rut and ret.
        # self.check_consistency()
        # errors = self.get_inconsistency()
        errs = self.check_solution(recalculate=False, assign_missions=assign_missions)
        error_cat = errs.to_lendict()
        log.debug("errors: {}".format(error_cat))
        objective = self.get_objective_function(error_cat)

        # status
        # 0: best,
        # 1: improved,
        # 2: not-improved + undo,
        # 3: not-improved + not undo.
        if objective > self.prev_objective:
            # solution worse than previous
            status = 3
            if self.previous_solution and rn.random() > \
                    math.exp((self.prev_objective - objective) / temperature / 50):
                # we were unlucky: we go back to the previous solution
                status = 2
                self.set_solution(self.previous_solution)
                errs = self.check_solution(recalculate=False)
                log.debug('back to previous solution: {}'.format(self.prev_objective))
                objective = self.prev_objective
        else:
            # solution better than previous
            status = 1
            if objective < self.best_objective:
                # best solution found
                status = 0
                self.best_objective = objective
                log.info('best solution found: {}'.format(objective))
                self.best_solution = self.copy_solution()
        if status in self.status_accept:
            self.previous_solution = self.copy_solution()
            self.prev_objective = objective
        return objective, status, errs

    def get_candidates(self, errors, k=5, assign_missions=False):
        """
        Samples candidates from many sources (mainly errors) from the current solution.

        :param sd.SuperDict errors: dictionary with all errors.
        :param int k: maximum number of candidates to choose
        :return: chosen candidates
        """
        if assign_missions:
            candidates_tasks = self.get_candidates_tasks(errors)
        else:
            candidates_tasks = []
        candidates_maints = self.get_candidates_maints(errors)
        candidates_cluster = self.get_candidates_cluster(errors)
        candidates_dist_maints = self.get_candidates_dist_maints(errors)
        candidates_size_maints = self.get_candidates_size_maints(errors)
        candidates_min_assign = self.get_candidates_min_max_assign(errors)
        candidates_rut = self.get_candidates_rut(errors)
        candidates_merge = self.get_candidates_merge_consecutive()
        more_candidates_merge = self.get_candidates_merge_non_consecutive()

        candidates = candidates_tasks + candidates_maints + \
                     candidates_cluster + candidates_dist_maints + \
                     candidates_min_assign + candidates_rut + \
                     candidates_size_maints + candidates_merge + \
                     more_candidates_merge
        if not len(candidates):
            return []
        # we add a random resource.
        resources = list(self.instance.data['resources'].keys())
        candidates.extend([(c, None) for c in rn.choices(resources, k=1)])
        # we only select a few (or 1) resource to change
        candidates_filter = rn.choices(candidates, k=min(k, len(candidates)))

        # we want to guarantee we only change the same resource once per iteration:
        ress, dates = [t for t in zip(*candidates_filter)]
        res, indices = np.unique(ress, return_index=True)
        candidates_n = [t for t in zip(res, np.array(dates)[indices])]

        return candidates_n

    def get_candidates_cluster(self, errors):
        """
        :return: a list of candidates [(aircraft, period), (aircraft2, period2)] to free
        :rtype: tl.TupList
        """
        clust_hours = errors.get('hours', sd.SuperDict())
        if not len(clust_hours):
            return []
        clust_hours = clust_hours.to_tuplist().tup_to_start_finish(self.instance.compare_tups)
        c_cand = self.instance.get_cluster_candidates()
        return [(rn.choice(c_cand[c]), d) for c, d, q, d in clust_hours]

    def get_candidates_tasks(self, errors):
        """
        :return: a list of candidates [(aircraft, period), (aircraft2, period2)] to free
        :rtype: tl.TupList
        """
        tasks_probs = errors.get('resources', sd.SuperDict()).to_tuplist()
        t_cand = self.instance.get_task_candidates()
        candidates_to_change = [(rn.choice(t_cand[t]), d) for (t, d, n) in tasks_probs]
        return candidates_to_change

    def get_candidates_min_max_assign(self, errors):
        """
        :return: a list of candidates [(aircraft, period), (aircraft2, period2)] to free
        :rtype: tl.TupList
        """
        return errors.get('min_assign', sd.SuperDict()).to_tuplist().filter([0, 1])

    def get_candidates_dist_maints(self, errors):
        """
        :return: a list of candidates [(aircraft, period), (aircraft2, period2)] to free
        :rtype: tl.TupList
        """
        bad_maints = errors.get('dist_maints', sd.SuperDict())
        return [(r, t) for m, r, t1, t2 in bad_maints for t in [t1, t2]]

    def get_candidates_size_maints(self, errors):
        """
        :return: a list of candidates [(aircraft, period), (aircraft2, period2)] to free
        :rtype: tl.TupList
        """
        bad_maints = errors.get('maint_size', sd.SuperDict())
        return bad_maints.keys_l()

    def get_candidates_maints(self, errors):
        """
        :return: a list of candidates [(aircraft, period), (aircraft2, period2)] to free
        :rtype: tl.TupList
        """
        maints_probs = errors.get('elapsed', sd.SuperDict())
        if not len(maints_probs):
            return []
        inst = self.instance
        sol = self.solution
        dur = inst.get_maintenances('duration_periods')
        start_h = inst.get_param('start')
        end_h = inst.get_param('end')
        resources = inst.get_resources().keys()
        maint_dates = [(k[0], k[2]) for k, v in maints_probs.items() if v == -1]

        # TODO: consider changing this to pandas or numpy
        start = [(m, max(start_h, inst.shift_period(d, -dur[m]))) for m, d in maint_dates]
        end = [(m, min(end_h, inst.shift_period(d, dur[m]))) for m, d in maint_dates]
        periods = [(st[0], d) for (st, end) in zip(start, end) for d in inst.get_periods_range(st[1], end[1])]
        periods = sorted(set(periods))
        candidates = []
        for res in resources:
            for m, p in periods:
                states = sol.get_period_state(res, p, 'state_m')
                if states and m in states:
                    candidates.append((res, p))
                    break
        return candidates

    def get_candidates_merge_consecutive(self):
        """
        :return: a list of candidates [(aircraft, period), (aircraft2, period2)] to free
        :rtype: tl.TupList
        """
        next = self.instance.get_next_period
        data = self.solution.data['state_m']
        def consec_list(_list):
            filt = []
            for el, el2 in zip(_list, _list[1:]):
                if el2[0] != next(el[0]):
                    continue
                for tup in [el, el2]:
                    # only add each side if there is only one maintenance.
                    # small chance to try to change a union of maintenances
                    if len(tup[1]) == 1 or rn.random() > 0.7:
                        filt.append(tup[0])
            return filt

        return data.\
            vapply(lambda v: sorted(v.items())).\
            vapply(consec_list).\
            to_tuplist()

    def get_candidates_merge_non_consecutive(self):
        """
        1. We get maintenances that do not depend on calendar (ret).
        1. If they are alone in the planning horizon, they are a candidate.

        :return: a list of candidates [(aircraft, period), (aircraft2, period2)] to free
        :rtype: tl.TupList
        """
        data = self.solution.data['state_m']
        max_elapsed = self.instance.get_maintenances('max_elapsed_time')
        maints_no_elapsed = max_elapsed.clean(func=lambda v: v is None)
        # TODO: this can be made more efficient if we filter before making tuplist
        result =\
            data.to_dictup().\
            to_tuplist().\
            to_dict(result_col=2, indices=[0, 1]).\
            clean(func=lambda v: len(v)==1).\
            clean(func=lambda v: v[0] in maints_no_elapsed)
        return tl.TupList(result)

    def get_candidates_rut(self, errors):
        """
        :return: a list of candidates [(aircraft, period), (aircraft2, period2)] to free
        :rtype: tl.TupList
        """
        ct = self.instance.compare_tups
        maints_probs_st = \
            errors.get('usage', sd.SuperDict()).\
                to_tuplist().\
            to_start_finish(ct, pp=2)
        candidates = [(r, d) for m, r, d, p, e in maints_probs_st]
        return candidates

    def get_random_periods(self, ref_period):
        """
        Given a reference date, a period around it is created
        :param str ref_period: period (month)
        :return:
        :rtype: list
        """
        inst = self.instance
        first_h, last_h = inst.get_param('start'), inst.get_param('end')
        distance1 = round((rn.random()) * 6)
        distance2 = round((rn.random()) * 6)
        first = max(self.instance.shift_period(ref_period, -distance1), first_h)
        second = min(last_h, self.instance.shift_period(ref_period, distance2))
        return self.instance.get_periods_range(first, second)

    def get_start_maints_period(self, resource, periods, fixed_periods):
        """
        Gets the start period for the first non-fixed check that falls
        inside `periods` and for `resource`, for each maintenance.
        We assume that only one check for each maintenance is found in the given periods.

        :param str resource: resource to look maintenances
        :param list periods: periods where maintenance are looked for
        :param list fixed_periods: list of tuples (resource, period) with fixed periods
        :return: list of (period, maintenance).
        :rtype: list
        """
        maint_duration = self.instance.get_maintenances('duration_periods')
        sol = self.solution
        if resource not in sol.data['state_m']:
            return []

        # We don't count the maintenances that are in the first period... for now
        # but we register them as "found" so as not to
        maint_starts = []
        found = set()
        # we check if maint starts at period
        period_0 = periods[0]
        for period in periods:
            if (resource, period) in fixed_periods:
                continue
            states = sol.get_period_state(resource, period, 'state_m')
            if states is None:
                continue
            # if we already registered it, don't bother
            states = states.keys() - found
            # for each possible maint...
            for m in states:
                # register it as found
                found.add(m)
                if period == period_0 and maint_duration[m] > 1:
                    # we do not add long maintenances that are in the first period
                    # to guarantee that it's a maintenance start
                    continue
                maint_starts.append((period, m))

        return maint_starts

    def free_resource(self, candidate):
        """
        Empties totally or partially the assignments
        (maintenances and tasks) for a resource around some period.

        :param tuple candidate: a (resource, period) combination
        :return: 1 if success, 0 if failure
        :rtype: bool
        """
        data = self.solution.data
        places = [
            data['state_m'],
            data['task'],
            data['aux']['start']
        ]
        resource, date = candidate

        # three options:

        if rn.random() <= self.options['prob_free_aircraft']:
            # 1. we change one aircraft for its whole time
            log.debug('freeing resource: {}'.format(resource))
            for pl in places:
                if resource in pl:
                    pl[resource] = sd.SuperDict()
            self.initialize_resource_states(resource=resource)
            return 1

        if date is None:
            # if no date, we create a random one
            periods = self.instance.get_periods()
            date = rn.choice(periods)

        periods = self.get_random_periods(ref_period=date)

        if rn.random() <= self.options['prob_free_periods']:
            # 2. we change the periods for all resources
            # at least for tasks
            return self.free_periods(periods)

        # 3. we do a local edition of the resource
        return self.free_resource_periods(resource, periods)

    def free_periods(self, periods):
        """
        Takes a list of periods and removes all assignments (maintenances and tasks)
        for all aircraft between those two periods.
        Finally, updates aircraft status accordingly.

        :param list periods: list of periods to liberate
        :return: 1 if success, 0 if failure
        :rtype: int
        """
        if not periods:
            return 0

        data = self.solution.data
        first_period = periods[0]

        fixed_periods = self.instance.get_fixed_periods()

        # deactivate tasks
        resource_del_task = set()
        res_period_task_delete = tl.TupList()
        for resource, res_data in data['task'].items():
            for period in periods:
                if (resource, period) in fixed_periods:
                    continue
                task = res_data.pop(period, None)
                if task is not None:
                    resource_del_task.add(resource)
                    res_period_task_delete.add(resource, period, task)

        maints = self.instance.get_maintenances()
        resource_del_maint = set()
        res_period_maint_delete = tl.TupList()
        for resource in data['state_m']:
            maints_to_delete = self.get_start_maints_period(resource, periods, fixed_periods)
            for period, maint in maints_to_delete:
                log.debug('{} loses {} maint: {}'.format(resource, maint, period))
                duration = maints[maint]['duration_periods']
                for _period in self.instance.get_next_periods(period, duration):
                    self.del_maint(resource, _period, maint=maint)
                resource_del_maint.add(resource)
                res_period_maint_delete.add(resource, period, maint)

        if not (len(resource_del_task) or len(res_period_maint_delete)):
            return 0

        affects = self.instance.get_maintenances('affects')
        res_period_state = res_period_maint_delete
        res_period_state = [(r, p, _s) for r, p, s in res_period_state for _s in affects[s]]
        # TODO: add tasks set before computing
        # missions count as M and VS maintenances in this logic
        res_maint_first_period = \
            tl.TupList(res_period_state).\
            to_dict(result_col=1).\
            vapply(sorted).vapply(lambda v: v[0])

        for (resource, m), period in res_maint_first_period.items():
            for rt in self.instance.get_maint_rt(m):
                if rt == 'ret' and (resource, m) not in res_maint_first_period:
                    continue
                self.update_rt_until_next_maint(resource, period, m, rt)

        # TODO: update correctly in cases of deletion of a long maintenance
        return 1

    def free_resource_periods(self, resource, periods):
        """
        Takes a list of periods and removes all assignments (maintenances and tasks)
        for all aircraft between those two periods.
        Finally, updates aircraft status accordingly.

        :param str resource: resource to free
        :param list periods: list of periods to free
        :return: 1 if success, 0 if failure
        :rtype: bool
        """
        if not periods:
            return 0

        data = self.solution.data
        first_period = periods[0]
        maints = self.instance.get_maintenances()
        fixed_periods = self.instance.get_fixed_periods()

        # deactivate tasks
        delete_tasks = delete_maints = 0
        if resource in data['task']:
            for period in periods:
                if (resource, period) not in fixed_periods:
                    data['task'][resource].pop(period, None)
                    delete_tasks = 1
        # we have several options for deactivating tasks:
        # 1. we eliminate all existing maintenances.
        # 2. we move existing maintenances
        maints_found = self.get_start_maints_period(resource, periods, fixed_periods)
        for period, maint in maints_found:
            if rn.random() < self.options['prob_delete_maint']:
                # We can directly take out the maintenances
                log.debug('{} loses {} maint: {}'.format(resource, maint, period))
                duration = maints[maint]['duration_periods']
                for period in self.instance.get_next_periods(period, duration):
                    self.del_maint(resource, period, maint=maint)
                delete_maints = 1
            else:
                # Or we can just move them
                log.debug('move maint: {}, {}, {}'.format(resource, period, maint))
                # the moving of maintenances updates ret and rut by its own.
                # because it's more complicated
                self.move_maintenance(resource, period, maint=maint)

        # if we did not delete anything: we exit
        if not (delete_tasks or delete_maints):
            return 0

        times = ['rut']
        if delete_maints:
            times.append('ret')
        for t in times:
            for m in self.instance.get_rt_maints(t):
                self.update_rt_until_next_maint(resource, first_period, m, t)
        return 1

    def assign_missing_maints_to_aircraft(self, maint):
        """
        Iterates over all aircraft and assigns maintenances `maint` when missing.

        :param maint: maintenance to assign checks
        :return: None
        """
        inst = self.instance
        first = inst.get_param('start')
        maint_data = inst.data['maintenances'][maint]
        max_ret = maint_data['max_elapsed_time']
        remaining_time_size = \
            inst.get_resources().\
            vapply(lambda v: maint_data['elapsed_time_size'])
        if not max_ret:
            # if we have a VS, we need to re-adapt the size of hour window.
            rt = "rut"
            min_usage = \
                inst.get_resources('min_usage_period').\
                get_property('default').\
                vapply(lambda v: v if v else 10)
            remaining_time_size = min_usage.\
                vapply(lambda v: maint_data['used_time_size'] / v).\
                vapply(math.floor)
        else:
            rt = "ret"
        errors = set()
        it = 0
        # we're not supposed to reach 1000.
        while it < 1000:
            it += 1
            remaining = self.get_remainingtime(time=rt, maint=maint)
            maint_candidates = []
            for res, info in remaining.items():
                if res in errors:
                    continue
                for period, ret in info.items():
                    if ret > 0 or period < first:
                        # if there's no need of maintenance we skip it
                        continue
                    # This tuple has three components:
                    # (resource, first month to assign maintenance, month with troubles)
                    tup = res, \
                          max(first, inst.shift_period(period, -remaining_time_size[res] + 1)), \
                          max(first, period)
                    maint_candidates.append(tup)
                    break
            if not len(maint_candidates):
                break
            # TODO: this could be shuffled?
            maint_candidates.sort(key=lambda x: len(inst.get_periods_range(x[1], x[2])))
            for resource, start, end in maint_candidates:
                result = self.get_and_assign_maintenance(resource=resource,
                                                         maint_need=start,
                                                         max_period=end,
                                                         which_maint='random',
                                                         maint=maint)
                if not result:
                    # we failed, don't continue to try this resource
                    errors.add(resource)
        return

    def over_assign_missions(self):
        """
        Finds violations on minimum assignments for missions
        and corrects that by adding missions where needed.

        :return: None
        """
        min_assign_err = self.check_min_max_assignment()
        task_periods = self.instance.get_task_period_list(True)
        added = 0
        tasks = self.instance.get_tasks()
        for (res, start, end, state), target in min_assign_err.items():
            # if we assign too much: don't over-assign.
            # Also if it's a maintenance problem
            if target < 0 or state not in tasks:
                continue
            task = state
            task_periods_t = set(task_periods[task])

            modifs = [-1, 1]
            it = 0
            while len(modifs) and it < 1000:
                it += 1
                modif = modifs.pop()
                if modif > 0:
                    candidate_period = self.instance.shift_period(end, modif)
                    modif += 1
                else:
                    candidate_period = self.instance.shift_period(start, modif)
                    modif -= 1
                if candidate_period not in task_periods_t:
                    continue
                if not self.solution.is_resource_free(res, candidate_period):
                    continue
                last_period = self.find_assign_task(res, candidate_period, candidate_period, task)
                if last_period == candidate_period:
                    added += 1
                    modifs.append(modif)
                if added >= target:
                    break

    def move_maintenance(self, resource, start, maint, modif=None):
        """
        Moves a maintenance that starts at start to the left or right a number of periods.

        :param str resource: resource to move maintenance
        :param str start: start of maintenance to move
        :param str maint: maintenance to move
        :return:
        """
        inst = self.instance
        first, last = [inst.get_param(p) for p in ['start', 'end']]
        maint_data = inst.data['maintenances'][maint]
        affected_maints = maint_data['affects']
        duration = maint_data['duration_periods']
        _max = maint_data['max_elapsed_time']
        time_size = maint_data['elapsed_time_size']
        if _max is not None:
            _min = _max - time_size
        else:
            # this is the case of VS visits
            _min = 0
            # _max = inst.get_param('num_period') * 2
        end = inst.shift_period(start, duration - 1)
        if modif is None:
            modif = rn.randint(-math.ceil(duration/2), math.ceil(duration/2))
        ret = self.get_remainingtime(resource, start, 'ret', maint=maint)
        if ret is not None:
            new_ret = ret + modif
        else:
            # this is the case of VS visits
            new_ret = 0

        # we check if the move is legal with respect to ret
        if _max is not None and not (_min <= new_ret <= _max):
            return None
        if modif > 0:
            periods_to_add = inst.get_next_periods(end, modif + 1)
            periods_to_take = inst.get_next_periods(start, modif + 1)
            # the first one is not relevant.
            periods_to_add.pop(0)
            periods_to_take.pop()
        elif modif < 0:
            periods_to_add = inst.get_next_periods(start, -modif + 1, previous=True)
            periods_to_take = inst.get_next_periods(end, -modif + 1, previous=True)
            # the last one is not relevant.
            periods_to_add.pop()
            periods_to_take.pop(0)
        else:
            return None
        if periods_to_add[0] < first or periods_to_add[-1] > last:
            return None

        for period in periods_to_add:
            # we check if it's possible to move...
            # for that we check availability and ret only.
            if not self.solution.is_resource_free(resource, period):
                return None

        # we arrived here: we're assigning a maintenance:
        self.set_state_and_clean(resource, maint, periods_to_add)
        for m in affected_maints:
            for rt in self.instance.get_maint_rt(m):
                self.update_time_maint(resource, periods_to_add, time=rt, maint=m)

        # and deleting a maintenance, also:
        for period in periods_to_take:
            self.del_maint(resource, period, maint)
        if modif > 0:
            # we need to modify the previous periods.
            for m in affected_maints:
                for time in self.instance.get_maint_rt(m):
                    self.update_rt_until_next_maint(resource, periods_to_take[0], m, time)
        # now, for the next periods, we always need to update:
        maint_end = inst.shift_period(end, modif)
        start_update_rt = self.instance.get_next_period(maint_end)
        for m in affected_maints:
            for time in self.instance.get_maint_rt(m):
                self.update_rt_until_next_maint(resource, start_update_rt, m, time)
        return start

    def assign_missions(self):
        """
        Assigns all remaining missions to aircraft, if possible.
        :return: None
        """
        rem_resources = self.check_task_num_resources().to_dictdict()
        tasks = rem_resources.keys_l()
        rn.shuffle(tasks)
        for task in tasks:
            self.fill_mission(task, assign_maints=False, max_iters=5, rem_resources=rem_resources)

if __name__ == "__main__":
    import package.params as pm
    import package.instance as inst

    directory = pm.PATHS['data'] + 'examples/201811092041/'
    model_data = di.load_data(os.path.join(directory, 'data_in.json'))
    options = di.load_data(os.path.join(directory, 'options.json'))
    experiments = []
    seeds = [rn.random()*1000 for i in range(5)]
    for s in seeds:
        options['seed'] = s
        instance = inst.Instance(model_data)
        heur_obj = MaintenanceFirst(instance)
        solution = heur_obj.solve(options)
        experiments.append(heur_obj)
    # a = [exp1.check_solution() for exp1 in experiments]
    a = 1
    pass

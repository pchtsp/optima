import package.superdict as sd
import os
import package.data_input as di
import numpy as np
import package.instance as inst
import package.solution as sol
import package.heuristics as heur
import random as rn
import math
import ujson
import time
import logging as log


class MaintenanceFirst(heur.GreedyByMission):

    def __init__(self, instance, solution=None):

        super().__init__(instance, solution)

        pass

    def get_solution(self):
        data = self.solution.data
        data_copy = ujson.loads(ujson.dumps(data))
        return sd.SuperDict.from_dict(data_copy)

    def set_solution(self, data):
        data = ujson.loads(ujson.dumps(data))
        self.solution.data = sd.SuperDict.from_dict(data)
        return True

    def solve(self, options):
        self.options = options
        seed = options.get('seed')
        if not seed:
            seed = rn.random()*10000
            options['seed'] = seed
        rn.seed(seed)

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
        _log.setLevel(level)


        max_iters = options.get('max_iters', 99999999)
        max_time = options.get('timeLimit', 600)
        cooling = options.get('cooling', 0.995)
        num_change_probs = options.get('num_change', [1])
        num_change_pos = [n + 1 for n, _ in enumerate(num_change_probs)]
        i = 0
        self.best_solution = None
        self.previous_solution = None
        self.prev_errors = 10000
        self.min_errors = 10000
        self.temperature = self.options.get('temperature', 1)
        status_worse = {2, 3}
        status_accept = {0, 1, 2}
        time_init = time.time()
        maints = self.instance.get_maintenances('priority')
        # M maintenances should always be first
        maints, priori = zip(*sorted(maints.items(), key=lambda x: x[1]))
        while True:
            # 1. assign maints for all aircraft around horizon limits
            for m in maints:
                self.assign_missing_maints_to_aircraft(m)

            # 2. assign missions for all aircraft
            self.assign_missions()

            # 3. try to correct min_assign
            self.over_assign_missions()

            # check quality of solution, store best.
            status, errors = self.analyze_solution(status_accept)
            error_cat = errors.to_lendict()
            num_errors = sum(error_cat.values())

            # 3. check if feasible. If not, un-assign (some/ all) or move maints
            num_change = rn.choices(num_change_pos, num_change_probs)[0]
            candidates = self.get_candidates(k=num_change, errors=errors)
            for candidate in candidates:
                self.free_resource(candidate)

            # sometimes, we go back to the best solution found
            if rn.random() < 0.01 and num_errors > self.min_errors and self.best_solution:
                self.set_solution(self.best_solution)
                num_errors = self.prev_errors = self.min_errors
                log.info('back to best solution: {}'.format(self.min_errors))

            if status in status_worse:
                self.temperature *= cooling
            clock = time.time()
            time_now = clock - time_init

            log.info("time={}, iteration={}, temperaure={}, errors={}, best={}".
                     format(round(time_now), i, round(self.temperature, 4), num_errors, self.min_errors))
            i += 1

            if not self.min_errors or i >= max_iters or time_now > max_time:
                break

        return sol.Solution(self.best_solution)

    def analyze_solution(self, status_accept):
        """

        :return: (status, errors)
        """
        errs = self.check_solution(recalculate=False)
        error_cat = errs.to_lendict()
        log.debug("errors: {}".format(error_cat))
        num_errors = sum(error_cat.values())
        # status
        # 0: best,
        # 1: improved,
        # 2: not-improved + undo,
        # 3: not-improved + not undo.
        if num_errors > self.prev_errors:
            # solution worse than previous
            status = 3
            if self.previous_solution and rn.random() > \
                    math.exp((self.prev_errors - num_errors) / self.temperature / 50):
                # we were unlucky: we go back to the previous solution
                status = 2
                self.set_solution(self.previous_solution)
                log.debug('back to previous solution: {}'.format(self.prev_errors))
                num_errors = self.prev_errors
        else:
            # solution better than previous
            status = 1
            if num_errors < self.min_errors:
                # best solution found
                status = 0
                self.min_errors = num_errors
                log.info('best solution found: {}'.format(num_errors))
                self.best_solution = self.get_solution()
        if status in status_accept:
            self.previous_solution = self.get_solution()
            self.prev_errors = num_errors
        return status, errs

    def get_candidates(self, errors, k=5):
        candidates_tasks = self.get_candidates_tasks(errors)
        candidates_maints = self.get_candidates_maints(errors)
        candidates_cluster = self.get_candidates_cluster(errors)
        candidates_dist_maints = self.get_candidates_bad_maints(errors)
        candidates_min_assign = self.get_candidates_min_max_assign(errors)
        candidates_rut = self.get_candidates_rut(errors)
        candidates = candidates_tasks + candidates_maints + \
                     candidates_cluster + candidates_dist_maints + \
                     candidates_min_assign + candidates_rut
        if not len(candidates):
            return []
        # we add a random resource.
        resources = list(self.instance.data['resources'].keys())
        candidates.extend([(c, None) for c in rn.choices(resources, k=1)])
        # we only select a few (or 1) resource to change
        candidates_filter = rn.choices(candidates, k=min(k, len(candidates)))

        # we want to garantee we only change the same resource once per iteration:
        ress, dates = [t for t in zip(*candidates_filter)]
        res, indices = np.unique(ress, return_index=True)
        candidates_n = [t for t in zip(res, np.array(dates)[indices])]

        return candidates_n

    def get_candidates_cluster(self, errors):
        clust_hours = errors.get('hours', sd.SuperDict())
        if not len(clust_hours):
            return []
        clust_hours = clust_hours.to_tuplist().tup_to_start_finish(self.instance.compare_tups)
        c_cand = self.instance.get_cluster_candidates()
        return [(rn.choice(c_cand[c]), d) for c, d, q, d in clust_hours]

    def get_candidates_tasks(self, errors):
        tasks_probs = errors.get('resources', sd.SuperDict()).to_tuplist()
        t_cand = self.instance.get_task_candidates()
        candidates_to_change = [(rn.choice(t_cand[t]), d) for (t, d, n) in tasks_probs]
        return candidates_to_change

    def get_candidates_min_max_assign(self, errors):
        return errors.get('min_assign', sd.SuperDict()).to_tuplist().filter([0, 1])

    def get_candidates_bad_maints(self, errors):
        bad_maints = errors.get('dist_maints', sd.SuperDict())
        return [(r, t) for m, r, t1, t2 in bad_maints for t in [t1, t2]]

    def get_candidates_maints(self, errors):
        maints_probs = errors.get('elapsed', sd.SuperDict())
        if not len(maints_probs):
            return []
        start_h = self.instance.get_param('start')
        end_h = self.instance.get_param('end')
        duration = self.instance.get_param('maint_duration')
        resources = self.instance.get_resources().keys()
        dates = [k[2] for k, v in maints_probs.items() if v == -1]
        start = [max(start_h, self.instance.shift_period(d, -duration)) for d in dates]
        end = [min(end_h, self.instance.shift_period(d, duration)) for d in dates]
        periods = [d for (st, end) in zip(start, end) for d in self.instance.get_periods_range(st, end)]
        periods = sorted(set(periods))
        data = self.solution.data
        candidates = []
        for res in resources:
            for p in periods:
                if data['state'].get(res, {}).get(p, '') == 'M':
                    candidates.append((res, p))
                    break
        # candidates = [res for p in periods for res in resources
        #              if data['state'].get(res, {}).get(p, '') == 'M']
        # pick 2?
        return candidates

    def get_candidates_rut(self, errors):
        ct = self.instance.compare_tups
        maints_probs_st = \
            errors.get('usage', sd.SuperDict()).\
                to_tuplist().\
            tup_to_start_finish(ct=ct, pp=2)
        candidates = [(r, d) for m, r, d, p, e in maints_probs_st]
        return candidates

    def get_random_periods(self, ref_period):
        """
        Given a reference date, a period around it is created
        :param ref_period: period (month)
        :return:
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
        gets the maintenance start period for the first maintenance that falls
        inside "periods" for resource. That is not fixed
        :param resource:
        :param periods:
        :param fixed_periods:
        :return:
        """
        sol = self.solution
        delete_maint = []
        for period in periods[1:]:
            if (resource, period) in fixed_periods:
                continue
            state = sol.get_period_state(resource, period)
            if state is not None:
                delete_maint.append((period, state))
        # If the first period is a maintenance: we are not sure it starts there
        # if delete_maint == periods[0]:
        #     prev_period = self.instance.get_prev_period(delete_maint)
        #     # we need to search for the first month
        #     while data['state'][resource].get(prev_period) == 'M':
        #         delete_maint = prev_period
        #         prev_period = self.instance.get_prev_period(delete_maint)
        return delete_maint

    def free_resource(self, candidate):
        """
        This function empties totally or partially the assignments
        (maintenances, tasks) made to a resource.
        :param candidate: a resource, date tuple
        :return:
        """
        # a = self.get_status(candidate[0])
        # a[:50]
        data = self.solution.data
        places = [
            data['state'],
            data['task'],
            data['aux']['start']
        ]
        resource, date = candidate

        # three options:

        if rn.random() <= self.options['prob_free_aircraft']:
            # we change one aircraft for its whole time
            log.debug('freeing resource: {}'.format(resource))
            for pl in places:
                if resource in pl:
                    pl[resource] = sd.SuperDict()
            self.initialize_resource_states(resource=resource)
            return 1

        log.debug('freeing resource: {}'.format(candidate))
        if date is None:
            # if no date, we create a random one
            periods = self.instance.get_periods()
            date = rn.choice(periods)

        periods = self.get_random_periods(ref_period=date)

        if rn.random() <= self.options['prob_free_periods']:
            # 2. we change the periods for all resources
            # at least for tasks
            self.free_periods(periods)

        # 3. we do a local edition of the resource
        return self.free_resource_periods(resource, periods)

    def free_periods(self, periods):
        # TODO: change maintenances too??
        data = self.solution.data
        first_period = periods[0]

        fixed_periods = self.instance.get_fixed_periods()

        # deactivate tasks
        delete_tasks = maint_found = 0
        for resource in data['task']:
            for period in periods:
                if (resource, period) not in fixed_periods:
                    data['task'][resource].pop(period, None)
                    delete_tasks = 1

        if not (delete_tasks or maint_found):
            return 0

        times = ['rut']
        if maint_found:
            times.append('ret')
        resources = self.instance.get_resources()
        maints = self.instance.get_maintenances()
        for resource in resources:
            for m in maints:
                for t in times:
                    # only update remaining hours where there is no maintenances
                    self.update_rt_until_next_maint(resource, first_period, m, t)
        return 1

    def free_resource_periods(self, resource, periods):
        if not periods:
            return 0

        data = self.solution.data
        first_period = periods[0]
        maints = self.instance.get_maintenances()
        fixed_periods = self.instance.get_fixed_periods()

        # deactivate tasks
        delete_tasks = 0
        if resource in data['task']:
            for period in periods:
                if (resource, period) not in fixed_periods:
                    data['task'][resource].pop(period, None)
                    delete_tasks = 1

        # we have several options for deactivating tasks:
        # 1. we eliminate all existing maintenances.
        # 2. we move existing maintenances
        # TODO: adapt to more maints
        maints_found = []
        if resource in data['state']:
            maints_found = self.get_start_maints_period(resource, periods, fixed_periods)
        for period, maint in maints_found:
            if rn.random() < self.options['prob_delete_maint']:
                # We can directly take out the maintenances
                    duration = maints[maint]['duration_periods']
                    for period in self.instance.get_next_periods(period, duration):
                        self.del_maint(resource, period, maint=maint)
            else:
                # Or we can just move them
                maints_found = self.move_maintenance(resource, maints_found)

        # of we did not delete anything: we exit
        if not (delete_tasks or maints_found):
            return 0

        times = ['rut']
        if maints_found:
            times.append('ret')
        for m in maints:
            for t in times:
                self.update_rt_until_next_maint(resource, first_period, m, t)
        return 1


    def assign_missing_maints_to_aircraft(self, maint='M'):
        first = self.instance.get_param('start')
        last = self.instance.get_param('end')
        maint_data = self.instance.data['maintenances'][maint]
        duration = maint_data['duration_periods']
        elapsed_time_size = maint_data['elapsed_time_size']
        # elapsed_time_size = self.instance.get_param('elapsed_time_size')
        errors = []
        while True:
            rets = self.get_remainingtime(time='ret', maint=maint)
            maint_candidates = []
            for res, info in rets.items():
                if res in errors:
                    continue
                for period, ret in info.items():
                    if ret > 0:
                        continue
                    tup = res, \
                          max(first, self.instance.shift_period(period, -elapsed_time_size + 1)), \
                          period
                    maint_candidates.append(tup)
                    break
            if not len(maint_candidates):
                break
            maint_candidates.sort(key=lambda x: len(self.instance.get_periods_range(x[1], x[2])))
            for resource, start, end in maint_candidates:
                result = self.find_assign_maintenance(resource=resource,
                                                      maint_need=start,
                                                      max_period=end,
                                                      which_maint='random',
                                                      maint=maint)
                if not result:
                    # we failed miserably, don't continue to try this resource
                    errors.append(resource)
        return

    def over_assign_missions(self):
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
            while len(modifs):
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

    def move_maintenance(self, resource, start):
        """
        we move a maintenance that starts at start to the left or right
        :param resource:
        :param start: start of maintenance
        :return:
        """
        # TODO: adapt to multiple maints
        # a = self.get_status(resource)
        # a[a.period >= '2021-03'][:8]
        first, last = [self.instance.get_param(p) for p in ['start', 'end']]
        duration = self.instance.get_param('maint_duration')
        end = self.instance.shift_period(start, duration - 1)
        modif = rn.randint(-3, 3)
        ret = self.get_remainingtime(resource, start, 'ret', maint='M')
        new_ret = ret + modif
        _max = self.instance.get_param('max_elapsed_time')
        _min = _max - self.instance.get_param('elapsed_time_size')
        if new_ret > _max or new_ret < _min:
            return None
        if modif > 0:
            periods_to_add = self.instance.get_next_periods(end, modif + 1)
            periods_to_take = self.instance.get_next_periods(start, modif + 1)
            # the first one is not relevant.
            periods_to_add.pop(0)
            periods_to_take.pop()
        elif modif < 0:
            periods_to_add = self.instance.get_next_periods(start, -modif + 1, previous=True)
            periods_to_take = self.instance.get_next_periods(end, -modif + 1, previous=True)
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
        for period in periods_to_add:
            self.set_state(resource, period)
        self.update_time_maint(resource, periods_to_add, time='ret')
        self.update_time_maint(resource, periods_to_add, time='rut')

        # and deleting a maintenance, also:
        for period in periods_to_take:
            self.del_maint(resource, period)
        return start

    def assign_missions(self):
        rem_resources = self.check_task_num_resources().to_dictdict()
        tasks = rem_resources.keys_l()
        rn.shuffle(tasks)
        for task in tasks:
            self.fill_mission(task, assign_maints=False, max_iters=5, rem_resources=rem_resources)

if __name__ == "__main__":
    import package.params as pm

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
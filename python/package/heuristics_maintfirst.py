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


class MaintenanceFirst(heur.GreedyByMission):

    def __init__(self, instance, solution=None):

        super().__init__(instance, solution)

        pass

    def get_solution(self):
        data = self.solution.data
        return ujson.loads(ujson.dumps(data))

    def set_solution(self, data):
        self.solution.data = ujson.loads(ujson.dumps(data))

    def solve(self, options):
        self.options = options
        seed = options.get('seed')
        if not seed:
            seed = rn.random()*10000
            options['seed'] = seed
        rn.seed(seed)
        max_iters = options.get('max_iters', 100)
        cooling = options.get('cooling', 0.995)
        num_change_prob = options.get('num_change', [1])
        # prob_ch_all = self.options['prob_ch_all']
        temperature = self.options.get('temperature', 1)
        num_change = [n + 1 for n, _ in enumerate(num_change_prob)]
        i = 0
        best_solution = None
        previous_solution = None
        prev_errors = 10000
        min_errors = 10000
        while True:
            # 1. assign maints for all aircraft around horizon limits
            self.assign_missing_maints_to_aircraft()

            # 2. assign missions for all aircraft
            self.assign_missions()

            # 3. try to correct min_assign
            self.over_assign_missions()

            # check quality of solution, store best.
            error_cat = self.check_solution_count(recalculate=False)
            if self.options['print']:
                print("errors: {}".format(error_cat))
            num_errors = sum(error_cat.values())
            if num_errors < min_errors:
                prev_errors = min_errors = num_errors
                print('best solution found: {}'.format(num_errors))
                best_solution = self.get_solution()
            elif num_errors > prev_errors \
                    and rn.random() > math.exp((prev_errors - num_errors)/temperature/50) \
                    and previous_solution:
                self.set_solution(previous_solution)
                num_errors = prev_errors
            else:
                previous_solution = self.get_solution()
                prev_errors = num_errors
            # 3. check if feasible. If not, un-assign (some/ all)
                # maintenances and go to 1
            num_change_iter = rn.choices(num_change, num_change_prob)[0]
            candidates = self.get_candidates(num_change_iter)
            for candidate in candidates:
                self.free_resource(candidate)

            # sometimes, we go back to the best solution found
            if rn.random() < 0.01 and num_errors > min_errors and best_solution:
                self.set_solution(best_solution)
                num_errors = prev_errors = min_errors
                print('back to best solution: {}'.format(min_errors))

            temperature *= cooling
            # prob_ch_all = self.options['prob_ch_all']
            # self.options['prob_ch_all'] = prob_ch_all*cooling

            print("iteration={}, temperaure={}, errors={}, best={}".
                  format(i, temperature, num_errors, min_errors))
            i += 1

            if not min_errors or i >= max_iters:
                break

        return sol.Solution(best_solution)

    def get_candidates(self, k=5):
        candidates_tasks = self.get_candidates_tasks()
        candidates_maints = self.get_candidates_maints()
        candidates_cluster = self.get_candidates_cluster()
        candidates_dist_maints = self.get_candidates_bad_maints()
        candidates_min_assign = self.get_candidates_min_assign()
        candidates = candidates_tasks + candidates_maints + \
                     candidates_cluster + candidates_dist_maints + \
                     candidates_min_assign
        if not len(candidates):
            return []
        resources = sd.SuperDict(self.instance.data['resources']).keys_l()
        candidates_filter = rn.choices(candidates, k=min(k, len(candidates)))
        ress, dates = [t for t in zip(*candidates_filter)]
        res, indices = np.unique(ress, return_index=True)
        candidates_n = [t for t in zip(res, np.array(dates)[indices])]
        candidates_n.extend([(c, None) for c in rn.choices(resources, k=1)])
        return candidates_n

    def get_candidates_cluster(self):
        clust_hours = self.check_min_flight_hours(recalculate=False)
        if not len(clust_hours):
            return []
        clust_hours = clust_hours.to_tuplist().tup_to_start_finish(self.instance.compare_tups)
        c_cand = self.instance.get_cluster_candidates()
        return [(rn.choice(c_cand[c]), d) for c, d, q, d in clust_hours]

    def get_candidates_tasks(self):
        tasks_probs = self.check_task_num_resources().to_tuplist()
        t_cand = self.instance.get_task_candidates()
        candidates_to_change = [(rn.choice(t_cand[t]), d) for (t, d, n) in tasks_probs]
        # print(tasks_probs)
        # for task, date, num in tasks_probs:
        #     cands_init = self.instance.get_task_candidates(task)
        #     cands = rn.choices(cands_init, k=num)
        #     candidates_to_change.extend(cands)
        # return rn.choices(candidates_to_change, k=min(5, len(candidates_to_change)))
        return candidates_to_change

    def get_candidates_min_assign(self):
        return self.check_min_assignment().keys_l()

    def get_candidates_bad_maints(self):
        bad_maints = self.check_min_distance_maints()
        return [(r, t) for r, t1, t2 in bad_maints for t in [t1, t2]]

    def get_candidates_maints(self):
        maints_probs = self.check_elapsed_consumption(recalculate=False)
        if not len(maints_probs):
            return []
        start_h = self.instance.get_param('start')
        end_h = self.instance.get_param('end')
        duration = self.instance.get_param('maint_duration')
        resources = self.instance.get_resources().keys()
        dates = [k[1] for k, v in maints_probs.items() if v == -1]
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

    def get_random_periods(self, ref_period):
        """
        Given a reference date, a period around is it's created
        :param ref_period: period (month)
        :return:
        """
        distance1 = round((rn.random()) * 6)
        distance2 = round((rn.random()) * 6)
        first = self.instance.shift_period(ref_period, -distance1)
        second = self.instance.shift_period(ref_period, distance2)
        return self.instance.get_periods_range(first, second)

    def free_resource(self, candidate):
        """
        This function empties totally or partially the assignments
        (maintenances, tasks) made to a resource.
        :param resource: resource to empty
        :return:
        """
        data = self.solution.data
        places = [
            data['state'],
            data['task'],
            data['aux']['start']
        ]
        resource, date = candidate
        if rn.random() <= self.options['prob_ch_all']:
            if self.options['print']:
                print('freeing resource: {}'.format(resource))
            for pl in places:
                if resource in pl:
                    pl[resource] = {}
            self.initialize_resource_states(resource=resource)
        else:
            if self.options['print']:
                print('freeing resource: {}'.format(candidate))
            if date is None:
                # if no date, we create a random one
                periods = self.instance.get_periods()
                date = rn.choice(periods)
            periods = self.get_random_periods(ref_period=date)
            fixed_periods = self.instance.get_fixed_periods()
            # deactivate tasks
            if resource in data['task']:
                for period in periods:
                    if (resource, period) not in fixed_periods:
                        data['task'][resource].pop(period, None)

            # deactivate states
            if resource not in data['state']:
                return
            delete_maint = None
            for period in periods:
                if data['state'][resource].get(period) == 'M' and \
                                (resource, period) not in fixed_periods:
                    delete_maint = period
                    break
            if delete_maint is not None and delete_maint > periods[0]:
                # we only delete maintenances that start inside the periods
                duration = self.instance.get_param('maint_duration')
                for period in self.instance.get_next_periods(delete_maint, duration):
                    data['state'][resource].pop(period, None)

            # only update remaining hours when no maintenances
            # also, the ret since we're deactivating maintenances
            non_maintenances = self.get_non_maintenance_periods(resource)
            for resource, start, end in non_maintenances:
                periods = self.instance.get_periods_range(start, end)
                for t in ['rut', 'ret']:
                    self.update_time_usage(resource, periods, time=t)


    def assign_missing_maints_to_aircraft(self):
        first = self.instance.get_param('start')
        last = self.instance.get_param('end')
        duration = self.instance.get_param('maint_duration')
        elapsed_time_size = self.instance.get_param('elapsed_time_size')
        errors = []
        while True:
            rets = sd.SuperDict(self.solution.data['aux']['ret'])
            maint_candidates = []
            for res, info in rets.items():
                if res in errors:
                    continue
                for period, ret in info.items():
                    if ret <= 0:
                        tup = res, \
                              max(first, self.instance.shift_period(period, -elapsed_time_size + 1)), \
                              min(last, self.instance.shift_period(period, duration))
                        maint_candidates.append(tup)
                        break
            if not len(maint_candidates):
                break
            maint_candidates.sort(key=lambda x: len(self.instance.get_periods_range(x[1], x[2])))
            for resource, start, end in maint_candidates:
                result = self.find_assign_maintenance(resource=resource,
                                                      maint_need=start,
                                                      max_period=end,
                                                      which_maint='random')
                if not result:
                    # we failed miserably
                    errors.append(resource)
        return

    def over_assign_missions(self):
        first_period = self.instance.get_param('start')
        min_assign = self.instance.get_min_assign()
        min_assign_err_start = sd.SuperDict(self.check_min_assignment())
        task_periods = self.instance.get_task_period_list(True)
        added = 0
        for (res, start), size in min_assign_err_start.items():
            end = self.instance.shift_period(start, size - 1)
            if start >= first_period:
                task = self.solution.data['task'][res][start]
            else:
                task = self.instance.data['resources'][res]['states'][start]
            task_periods_t = set(task_periods[task])
            target = min_assign[task] - size
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
        duration = self.instance.get_param('maint_duration')
        end = self.instance.shift_period(start, duration - 1)
        new_end = self.instance.shift_period(start, end + 1)
        new_start = self.instance.get_prev_period(start)
        options = [(start, new_end), (end, new_start)]
        options_f = []
        for op in options:
            if self.solution.is_resource_free(resource, op[1]):
                options_f.append(op)


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
import package.superdict as sd
import os
import package.data_input as di
import numpy as np
import package.instance as inst
import package.solution as sol
import package.heuristics as heur
import random as rn
import copy


class MaintenanceFirst(heur.GreedyByMission):

    def __init__(self, instance, solution=None):

        super().__init__(instance, solution)

        pass

    def solve(self, options):
        seed = options.get('seed')
        if not seed:
            seed = rn.random()*10000
            options['seed'] = seed
        rn.seed(seed)
        i = 0
        best_solution = None
        min_errors = 10000
        while True:
            # 1. assign maints for all aircraft around horizon limits
            self.assign_missing_maints_to_aircraft()

            # 2. assign missions for all aircraft
            self.assign_missions()

            # 3. check if feasible. If not, un-assign (some/ all)
                # maintenances and go to 1
            candidates = self.get_candidates(1)
            if not len(candidates) or i >= 100:
                break
            for candidate in candidates:
                self.free_resource(candidate)

            # sometimes, we go back to the best solution found
            if rn.random() < 0.05:
                self.solution.data = copy.deepcopy(best_solution)
                print('back to best solution')

            # check quality of solution, store best.
            error_cat = self.check_solution_count()
            if self.options['print']:
                print("errors: {}".format(error_cat))
            num_errors = sum(error_cat.values())
            if num_errors < min_errors:
                min_errors = num_errors
                print('best solution found: {}'.format(num_errors))
                best_solution = copy.deepcopy(self.solution.data)
            prob_ch_all = self.options['prob_ch_all']
            self.options['prob_ch_all'] = prob_ch_all*0.995
            print("iteration={}, prob_ch_all={}, errors={}".format(i, prob_ch_all, num_errors))
            i += 1
        return sol.Solution(best_solution)

    def get_candidates(self, k=5):
        candidates_tasks = self.get_candidates_tasks()
        candidates_maints = self.get_candidates_maints()
        candidates_cluster = self.get_candidates_cluster()
        candidates = set(candidates_tasks + candidates_maints + candidates_cluster)
        if not len(candidates):
            return []
        resources = sd.SuperDict(self.instance.data['resources']).keys_l()
        candidates_filter = rn.choices(sorted(candidates), k=min(k, len(candidates)))
        ress, dates = [t for t in zip(*candidates_filter)]
        res, indices = np.unique(ress, return_index=True)
        candidates_n = [t for t in zip(res, np.array(dates)[indices])]
        # candidates_n.extend([(c, None) for c in rn.choices(resources, k=k)])
        return candidates_n

    def get_candidates_cluster(self):
        clust_hours = self.check_min_flight_hours()
        if not len(clust_hours):
            return []
        clust_hours = clust_hours.to_tuplist().tup_to_start_finish()
        c_cand = self.instance.get_cluster_candidates()
        # clusts = [c for c, _ in clust_hours]
        # print(clust_hours)
        # if len(clust_hours):
        #     a = 1
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
        return rn.choices(candidates_to_change, k=min(5, len(candidates_to_change)))

    def get_candidates_maints(self):
        maints_probs = self.check_elapsed_consumption()
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
                        # print(res, period, ret)
                        tup = res, \
                              max(first, self.instance.shift_period(period, -elapsed_time_size + 1)), \
                              min(last, self.instance.shift_period(period, duration))
                        maint_candidates.append(tup)
                        break
            if not len(maint_candidates):
                break
            # maint_candidates = [(r, s, e) for r, (s, e) in per_maint_start.items()]
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
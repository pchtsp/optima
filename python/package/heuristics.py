import package.auxiliar as aux
import numpy as np
import package.experiment as test
import package.solution as sol
import random as rn
import pytups.tuplist as tl
import logging as log


class GreedyByMission(test.Experiment):

    def __init__(self, instance, solution=None):

        self.options = {'debug': True}

        # if solution given, just initialize and go
        if solution is not None:
            super().__init__(instance, solution)
            return

        # if not, create a mock solution, initialize and fill
        sol_data = {'state': {}, 'task': {}, 'aux': {'ret': {}, 'rut': {}, 'start': {}}}
        solution = sol.Solution(sol_data)
        super().__init__(instance, solution)

        resources = list(self.instance.get_resources())

        for r in resources:
            self.initialize_resource_states(r)

        return

    def solve(self, options):

        self.options.update(options)

        # 1. Choose a mission.
        # 2. Choose candidates for that mission.
        # 3. Start assigning candidates to the mission's months.
            # Here the selection order could be interesting
            #  Assign maintenances when needed to each aircraft.
        # 4. When finished with the mission, repeat with another mission.
        # solution = {}
        # quality = {}
        # for i in range(10):
        check_candidates = self.instance.check_enough_candidates()
        tasks_sorted = sorted(check_candidates.items(), key=lambda x: x[1][1])
        # mission by mission: we find candidates and assign them
        determinist = options.get('determinist', True)
        for task, content in tasks_sorted:
            self.fill_mission(task, determinist)
        # at the end, there can be resources that need maintenance because of ret.
        # for r in range(2):
        needs = [k for k, v in self.check_elapsed_consumption().items() if v == -1]
        for res, period in needs:
            log.debug('resource {} needs a maintenance in {}'.format(res, period))
            self.find_assign_maintenance(res, self.instance.shift_period(period, -1), which_maint='latest')
        # solution[i] = copy.deepcopy(self.solution)
        # quality[i] = self.check_solution()
        return self.solution

    def fill_mission(self, task, assign_maints=True, max_iters=100, rem_resources=None):
        """
        This function assigns all the necessary resources to a given task
        It edits the solution object inside the experiment
        :param task: a task code to satisfy
        :return: nothing
        """
        if rem_resources is None:
            rem_resources = self.check_task_num_resources().to_dictdict()
        if task in rem_resources:
            rem_resources = rem_resources[task]
        else:
            return False
        duration = self.instance.get_param('maint_duration')
        dtype = 'U7'
        candidates = self.instance.get_task_candidates(task=task)
        i = 0
        while len(rem_resources) > 0 and len(candidates) > 0 and i < max_iters:
            i += 1
            # num_maints = {r: self.solution.get_number_maintenances(r) for r in candidates}
            # candidates.sort(key=lambda x: (num_maints[x], resource_num_tasks[x], ret_initial[x]))
            candidate = rn.choice(candidates)
            # get free periods for candidate
            periods_task = \
                self.get_free_starts(candidate, np.fromiter(rem_resources, dtype=dtype))
            if len(periods_task) == 0:
                # consider eliminating the resource from the list.
                # if all its periods are 'used'
                candidates.remove(candidate)
                continue
            maint_need = ''
            # for each period of consecutive months
            # we randomize them to give some more sugar
            rn.shuffle(periods_task)
            for start, end in periods_task:
                # here we do all the checks to see if we can assign tasks
                # and how many periods we can assign it to.
                last_month = self.find_assign_task(candidate, start, end, task)
                for period in self.instance.get_periods_range(start, last_month):
                    # we assign all found periods
                    rem_resources[period] -= 1
                    # delete periods that have 0 rem_resources to assign
                    if rem_resources[period] == 0:
                        rem_resources.pop(period)
                if maint_need == '' and last_month < end:
                    # we register (only) the first need of maintenance:
                    maint_need = self.instance.shift_period(last_month, -duration + 1)
            if maint_need == '' or not assign_maints:
                # if the resource has no need for maintenance yet, we don't attempt one
                # also, if we do not want to assign maintenances: we do not do it
                continue
            log.debug("resource {} needs a maintenance after period {}".format(candidate, maint_need))
            # find soonest period to start maintenance:
            result = self.find_assign_maintenance(candidate, maint_need)
            if not result:
                # the maintenance failed: we pop the candidate because it is most probably useless.
                candidates.remove(candidate)
        return

    def initialize_resource_states(self, resource):
        """
        Assign fixed states, and calculate ret and rut for resource
        :param resource:
        :return:
        """

        period_0 = self.instance.get_prev_period(self.instance.get_param('start'))

        # we preassign initial ruts and rets.
        for time in ['rut', 'ret']:
            initial_state = self.instance.get_initial_state(self.label_rt(time), resource=resource)
            self.set_remainingtime(resource, period_0, time, initial_state[resource])

        # we pre-assign fixed maintenances:
        fixed_states = self.instance.get_fixed_states(resource, filter_horizon=True)
        maint_periods = []
        for _, state, period in fixed_states:
            # if maintenance: we have a special treatment.
            cat = 'task'
            if state == 'M':
                cat = 'state'
                maint_periods.append(period)
            self.set_state(resource, period, state, cat=cat)

        self.update_time_maint(resource, maint_periods, time='ret')
        self.update_time_maint(resource, maint_periods, time='rut')

        # we update non-maintenance periods
        non_maintenances = self.get_non_maintenance_periods(resource)
        times = ['rut', 'ret']
        for _, start, end in non_maintenances:
            # print(resource, start, end)
            periods = self.instance.get_periods_range(start, end)
            for t in times:
                self.update_time_usage(resource, periods, time=t)

    def fix_over_maintenances(self):
        # sometimes the solution includes a 12 period maintenance
        # this function should delete the first of the two periods.
        return

    def find_assign_task(self, resource, start, end, task):
        periods_to_assign = self.check_assign_task(resource, self.instance.get_periods_range(start, end), task)
        info = self.instance.data['tasks'][task]
        min_asign = info['min_assign']
        # min_asign = 1
        last_month_task = info['end']
        size_assign = len(periods_to_assign)
        if not size_assign:
            # if there was nothing assigned: there is nothing to do
            return self.instance.shift_period(start, -1)
        first_period_to_assign = periods_to_assign[0]
        last_period_to_assign = periods_to_assign[-1]
        previous_period = self.instance.shift_period(first_period_to_assign, -1)
        next_period = self.instance.shift_period(last_period_to_assign, 1)
        previous_state = self.solution.data['task'].get(resource, {}).get(previous_period ,'')
        next_state = self.solution.data['task'].get(resource, {}).get(next_period, '')
        if (size_assign < min_asign
            and last_period_to_assign != last_month_task
            and previous_state != task
            and next_state != task):
            # if not enough to assign
            #   (and not the last task month
            # and different from previous state)
            # and different from next state)
            return self.instance.shift_period(start, -1)
        for period in periods_to_assign:
            self.expand_resource_period(self.solution.data['task'], resource, period)
            self.solution.data['task'][resource][period] = task
        # here, the updating of ret and rut is done.
        # It is done until the next maintenance or the end
        next_maint = self.get_next_maintenance(resource, end)
        if next_maint is None:
            periods_to_update = self.instance.get_periods_range(start, self.instance.get_param('end'))
        else:
            periods_to_update = self.instance.get_periods_range(start, next_maint)
        self.update_time_usage(resource, periods=periods_to_update, time='rut')
        # self.update_time_usage(resource, periods=periods_to_update, time='ret')
        return last_period_to_assign

    def find_assign_maintenance(self, resource, maint_need, max_period=None, which_maint='soonest'):
        """
        Tries to find the soonest maintenance in the planning horizon
        for a given resource.
        :param resource: resource to find maintenance
        :param maint_need: date when the resource needs the maintenance
        :param max_period: date when the resource can no longer have the maintenance
        :return:
        """
        # a = self.get_status(resource)
        # a[a.period>= periods_maint[0]][:8]
        if max_period is None:
            max_period = self.instance.get_param('end')
        horizon_end = self.instance.get_param('end')
        duration = self.instance.get_param('maint_duration')
        maint_start = None
        if which_maint == 'latest':
            maint_start = self.get_latest_maint(resource, maint_need)
        elif which_maint == 'random':
            maint_start = self.get_random_maint(resource, maint_need, max_period)
        elif maint_start == 'soonest':
            maint_start = self.get_soonest_maint(resource, maint_need)
        if maint_start is None:
            # this means that we cannot assign tasks BUT
            # we cannot assign maintenance either :/
            # we should then take out the candidate:
            log.debug("resource {} has no candidate periods for maintenance".format(resource))
            return False
        maint_end = min(self.instance.shift_period(maint_start, duration - 1), horizon_end)
        periods_maint = self.instance.get_periods_range(maint_start, maint_end)
        periods_to_update = periods_maint
        next_maint = self.get_next_maintenance(resource, maint_start)
        last_maint_prev = self.get_next_maintenance(resource, maint_start, previous=True)
        if last_maint_prev is not None and last_maint_prev >= maint_need:
            log.debug("resource {} has already a maintenance {} after the need {}.".
                      format(resource, last_maint_prev, maint_need))
            return False
        # TODO: this swap is not checking everything: sometimes make infeasible choices
        if next_maint is not None and False:
            # we need to take out the old one, that happens *after* the new.
            # and choose carefully the periods to update.
            log.debug("resource {} could swap maintenances: {} to {}".format(resource, next_maint, maint_start))
            old_maint_end = self.instance.shift_period(next_maint, duration - 1)
            for period in self.instance.get_periods_range(next_maint, old_maint_end):
                self.del_maint(resource, period)
        log.debug("resource {} will get a maintenance in periods {} -> {}".format(resource, maint_start, maint_end))
        start_update_rt = self.instance.get_next_period(maint_end)
        end_update_rt = self.get_next_maintenance(resource, start_update_rt)
        if end_update_rt is None:
            end_update_rt = horizon_end
        for period in periods_maint:
            self.set_state(resource, period)
        self.update_time_maint(resource, periods_to_update, time='ret')
        self.update_time_maint(resource, periods_to_update, time='rut')
        # it doesn't make sense to assign a maintenance after a maintenance
        if maint_end == self.instance.get_param('end'):
            # we assigned the last day to maintenance:
            # there is nothing to update.
            return True
        self.update_time_usage(resource, self.instance.get_periods_range(start_update_rt, end_update_rt), time='ret')
        self.update_time_usage(resource, self.instance.get_periods_range(start_update_rt, end_update_rt), time='rut')
        return True

    def check_assign_task(self, resource, periods, task):
        """
        Calculates the amount of periods it's possible to assign
        a given task to a resource.
        Based on the usage status of the resource.
        :param resource: candidate to assign a task
        :param periods: periods to try to assign the task
        :param task: task to assign to resource
        :return: subset of continuous periods to assign to the resource
        """
        min_usage = self.instance.get_param('min_usage_period')
        consumption = self.instance.data['tasks'][task]['consumption']
        net_consumption = consumption - min_usage
        start = periods[0]
        end = periods[-1]
        number_periods = len(periods)

        horizon_end = self.instance.get_param('end')
        next_maint = self.get_next_maintenance(resource, end)
        if next_maint is not None:
            before_maint = self.instance.shift_period(next_maint, -1)
            rut = self.solution.data['aux']['rut'][resource][before_maint]
        else:
            rut = self.solution.data['aux']['rut'][resource][horizon_end]
        # ret = self.solution.data['aux']['ret'][resource][end]
        number_periods_ret = self.solution.data['aux']['ret'][resource][start] - 2
        number_periods_rut = int(rut // net_consumption)
        final_number_periods = max(min(number_periods_ret, number_periods_rut, number_periods), 0)
        return periods[:final_number_periods]

    def get_free_periods_maint(self):
        """
        finds the periods where maintenance capacity is not full
        :return: list of periods (month)
        """
        num_in_maint = aux.fill_dict_with_default(self.solution.get_in_maintenance(),
                                                  self.instance.get_periods())
        return [p for p, num in num_in_maint.items() if
                                 num < self.instance.get_param('maint_capacity')]

    def get_maintenance_candidates(self, resource, min_period, max_period):
        periods_to_search = \
            np.intersect1d(self.get_free_periods_maint(),
                           self.get_free_periods_resource(resource))
        free = [(1, period) for period in periods_to_search
                if min_period <= period <= max_period]
        return tl.TupList(free).to_start_finish(self.instance.compare_tups)

    def get_random_maint(self, resource, min_period, max_period):
        """
        Finds a random maintenance from all possible assignments in range
        :param resource: resource (code) to search for maintenance
        :param min_period: period (month) to start searching for date
        :param max_period: period (month) to end searching for date
        :return: period (month) or none
        """
        maint_duration = self.instance.get_param('maint_duration')
        start_to_finish = \
            self.get_maintenance_candidates(
                resource, min_period=min_period, max_period=max_period
            )
        sizes = [(pos, len(self.instance.get_periods_range(st, end)) - maint_duration)
                 for pos, (id, st, end) in enumerate(start_to_finish)]
        maint_options = [(pos, t) for pos, size in sizes if size >= 0 for t in range(size+1)]
        if not len(maint_options):
            return None
        l = rn.choice(maint_options)
        pos1, pos2 = l
        _id, st, end = start_to_finish[pos1]
        return self.instance.shift_period(st, pos2)

    def get_soonest_maint(self, resource, min_period):
        """
        Finds the soonest possible maintenance to assign to a resource.
        :param resource: resource (code) to search for maintenance
        :param min_period: period (month) to start searching for date
        :return: period (month) or none
        """
        maint_duration = self.instance.get_param('maint_duration')
        horizon_end = self.instance.get_param('end')
        start_to_finish = \
            self.get_maintenance_candidates(
                resource, min_period=min_period, max_period=horizon_end
            )
        for (id, st, end) in start_to_finish:
            # the period needs to be as least the size of the maintenance
            # alternative: the end is the end of the horizon.
            if end == horizon_end or len(self.instance.get_periods_range(st, end)) >= maint_duration:
                return st
        return None

    def get_latest_maint(self, resource, max_period):
        """
        Finds the soonest possible maintenance to assign to a resource.
        :param resource: resource (code) to search for maintenance
        :param max_period: period (month) to end searching for date
        :return: period (month) or none
        """
        horizon_start = self.instance.get_param('start')
        maint_duration = self.instance.get_param('maint_duration')
        start_to_finish = \
            self.get_maintenance_candidates(
                resource, min_period=horizon_start, max_period=max_period
            )
        for (id, st, end) in reversed(start_to_finish):
            # the period needs to be as least the size of the maintenance
            # alternative: the end is the end of the horizon.
            if len(self.instance.get_periods_range(st, end)) >= maint_duration:
                return self.instance.shift_period(end, - maint_duration + 1)
        return None

    def get_free_periods_resource(self, resource):
        """
        Finds the list of periods (month) that the resource is available.
        :param resource: resource code
        :return: periods (month)
        """
        # resource = "A100"
        dtype = 'U7'
        union = \
            np.union1d(
            np.fromiter(self.solution.data['task'].get(resource, {}), dtype=dtype),
            np.fromiter(self.solution.data['state'].get(resource, {}), dtype=dtype)
        )
        return np.setdiff1d(
            np.fromiter(self.instance.get_periods(), dtype=dtype),
            union
        )

    def get_free_starts(self, resource, periods):
        # dtype = 'U7'
        candidate_periods = \
            np.intersect1d(
            periods,
            self.get_free_periods_resource(resource)
        )
        if len(candidate_periods) == 0:
            return []

        startend = tl.TupList([(1, p) for p in candidate_periods]).to_start_finish(self.instance.compare_tups)
        return startend.filter([1, 2])

    def update_time_maint(self, resource, periods, time='rut'):
        value = 'max_' + self.label_rt(time) + '_time'
        for period in periods:
            self.set_remainingtime(resource, period, time, self.instance.get_param(value))
        return True

    def del_maint(self, resource, period):
        if resource in self.solution.data['state']:
            return self.solution.data['state'][resource].pop(period, None)

    def set_state(self, resource, period, value='M', cat='state'):
        self.expand_resource_period(self.solution.data[cat], resource, period)
        self.expand_resource_period(self.solution.data['aux']['rut'], resource, period)
        self.expand_resource_period(self.solution.data['aux']['ret'], resource, period)

        self.solution.data[cat][resource][period] = value
        return True

    def get_maintenances(self, resource):
        return self.solution.data['state'].get(resource, {}).keys()

    def get_maintenance_periods_resource(self, resource):
        periods = [(1, k) for k, v in self.solution.data['state'].get(resource, {}).items() if v == 'M']
        result = tl.TupList(periods).to_start_finish(self.instance.compare_tups)
        return result.filter([1, 2])

    def get_next_maintenance(self, resource, min_start, previous=False):
        start_end = self.get_maintenance_periods_resource(resource)
        if len(start_end) > 1:
            start_end.sort(key=lambda x: x[0])
        prev_st = None
        for st, end in start_end:
            if st >= min_start:
                if previous:
                    return prev_st
                return st
            prev_st = st
        if previous:
            return prev_st
        return None


if __name__ == "__main__":
    #  see ../scripts/exec_heur.py
    pass
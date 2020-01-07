import numpy as np
import package.experiment as test
import package.solution as sol
import random as rn
import pytups.tuplist as tl
import logging as log
import pytups.superdict as sd


class GreedyByMission(test.Experiment):

    def __init__(self, instance, solution=None):

        self.options = {'debug': True}

        # if solution given, just initialize and go
        if solution is not None:
            super().__init__(instance, solution)
            for time in ['rut', 'ret']:
                self.set_remaining_usage_time_all(time=time)
            return

        # if not, create a mock solution
        solution = sol.Solution(sd.SuperDict())
        super().__init__(instance, solution)

        resources = list(self.instance.get_resources())

        for r in resources:
            self.initialize_resource_states(r)

        return

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
        dtype = 'U7'
        candidates = self.instance.get_task_candidates(task=task)[task]
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
            # for each period of consecutive months
            # we randomize them to give some more sugar
            rn.shuffle(periods_task)
            for start, end in periods_task:
                # here we do all the checks to see if we can assign tasks
                # and how many periods we can assign it to.
                last_month = self.find_assign_task(candidate, start, end, task)
                periods_assgined = self.instance.get_periods_range(start, last_month)
                if periods_assgined:
                    log.debug("resource {} gets mission assigned between periods {} and {}".
                              format(candidate, start, last_month))
                for period in periods_assgined:
                    # we assign all found periods
                    rem_resources[period] -= 1
                    # delete periods that have 0 rem_resources to assign
                    if rem_resources[period] == 0:
                        rem_resources.pop(period)
        return

    def initialize_resource_states(self, resource):
        """
        Assign fixed states, and calculate ret and rut for resource
        :param str resource:
        :return:
        """

        # we pre-assign fixed maintenances:
        fixed_states = self.instance.get_fixed_states(resource, filter_horizon=True)
        maintenances = self.instance.get_maintenances()
        for _, state, period in fixed_states:
            # if maintenance: we have a special treatment.
            if state in maintenances:
                # self.set_state(resource, period, cat='state', value=state)
                self.set_state(resource, period, state, cat='state_m', value=1)
            else:
                self.set_state(resource, period, cat='state', value=state)

        for time in ['rut', 'ret']:
            self.set_remaining_usage_time_all(time=time, resource=resource)

    def fix_over_maintenances(self):
        # sometimes the solution includes a 12 period maintenance
        # this function should delete the first of the two periods.
        return

    def find_assign_task(self, resource, start, end, task):
        """

        :param str resource:
        :param str start: period
        :param str end: period
        :param str task:
        :return:
        """
        _range = self.instance.get_periods_range(start, end)
        periods_to_assign = self.check_assign_task(resource, _range, task)
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
            tup = [resource, period]
            self.solution.data['task'].tup_to_dicts(tup=tup, value=task)
        # here, the updating of ret and rut is done.
        # It is done until the next maintenance or the end
        next_maint = self.get_next_maintenance(resource, end)
        if next_maint is None:
            periods_to_update = self.instance.get_periods_range(start, self.instance.get_param('end'))
        else:
            periods_to_update = self.instance.get_periods_range(start, next_maint)
        self.update_time_usage_all(resource, periods=periods_to_update, time='rut')
        return last_period_to_assign

    def get_and_assign_maintenance(self, resource, maint_need, max_period=None, which_maint='soonest', maint='M'):
        """
        Tries to find the soonest maintenance in the planning horizon
        for a given resource.
        :param resource: resource to find maintenance
        :param maint_need: date when the resource needs the maintenance
        :param max_period: date when the resource can no longer start the maintenance
        :param which_maint: three techniques to choose the position of the maintenance
        :param maint: maintenance to schedule
        :return:
        """
        # a = self.get_status(resource)
        # a[a.period>= periods_maint[0]][:8]
        if max_period is None:
            max_period = self.instance.get_param('end')
        maint_start = None
        if which_maint == 'latest':
            raise NameError('not implemented')
        elif which_maint == 'random':
            maint_start = self.get_random_maint(resource, maint_need, max_period, maint=maint)
        elif maint_start == 'soonest':
            raise NameError('not implemented')
        if maint_start is None:
            # this means that we cannot assign tasks BUT
            # we cannot assign maintenance either :/
            # we should then take out the candidate:
            log.debug("{} has no candidate for maint {}".
                      format(resource, maint))
            return False
        return self.assign_maintenance(resource, maint, maint_start)

    def update_rt_until_next_maint(self, resource, start_update_rt, maint, time):
        """
        finds next maintenance compatible with the provided maint and
        updates remaining time until the next maintenance
        :param str resource:
        :param str start_update_rt: start the search for next maintenance.
        :param str maint: maintenance type
        :param str time: rut or ret
        :return: updated periods
        """
        horizon_end = self.instance.get_param('end')
        maint_data = self.instance.data['maintenances'][maint]
        depends_on = maint_data['depends_on']
        end_update_rt = self.get_next_maintenance(resource, start_update_rt, maints=set(depends_on))
        if end_update_rt is None:
            end_update_rt = horizon_end
        else:
            end_update_rt = self.instance.get_prev_period(end_update_rt)
        periods_to_update = self.instance.get_periods_range(start_update_rt, end_update_rt)
        self.update_time_usage(resource, periods_to_update, time=time, maint=maint)
        return periods_to_update

    def check_assign_task(self, resource, periods, task):
        """
        Calculates the amount of periods it's possible to assign
        a given task to a resource.
        Based on the usage status of the resource.
        :param str resource: candidate to assign a task
        :param list periods: periods to try to assign the task
        :param str task: task to assign to resource
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
        # TODO: deal with M
        if next_maint is not None:
            before_maint = self.instance.shift_period(next_maint, -1)
            rut = self.get_remainingtime(resource, before_maint, 'rut', 'M')
        else:
            rut = self.get_remainingtime(resource, horizon_end, 'rut', 'M')
        number_periods_ret = self.get_remainingtime(resource, start, 'ret', 'M')
        number_periods_rut = int(rut // net_consumption)
        final_number_periods = max(min(number_periods_ret, number_periods_rut, number_periods), 0)
        return periods[:final_number_periods]

    def get_non_free_periods_maint(self, maint='M', periods_to_check=None):
        """
        finds the periods where maintenance capacity is not full
        :return: list of periods (month)
        """
        _usage = self.instance.data['maintenances'][maint]['capacity_usage']
        _type = self.instance.data['maintenances'][maint]['type']
        type_periods = self.check_sub_maintenance_capacity(
            ref_compare=_usage, periods_to_check=periods_to_check
        )
        periods = []
        if len(type_periods):
            types, periods = zip(*type_periods)
            periods = np.asarray(periods)[np.asarray(types) == _type]
        return periods
        # periods_all = set(self.instance.get_periods())
        # return tl.TupList(periods_all - periods_full)


        # if len(type_periods):
        #     periods_full = np.array(type_periods.keys_l())[:, 1]
        #     periods_full = set(periods_full)
        # return tl.TupList(self.instance.get_periods()).filter_list_f(lambda x: x not in periods_full)
        # num_in_maint = aux.fill_dict_with_default(self.solution.get_in_maintenance(maint),
        #                                           )
        # return [p for p, num in num_in_maint.items() if
        #                          num < self.instance.get_param('maint_capacity')]

    def get_maintenance_candidates(self, resource, min_period, max_period, maint):
        """
        Obtaines candidate periods to start a maintenance.
        :param resource:
        :param min_period:
        :param max_period:
        :param maint:
        :return:
        """
        inst = self.instance
        maint_needed = set(inst.get_periods_range(min_period, max_period))
        maint_not_possible = set(self.get_non_free_periods_maint(maint, maint_needed))
        resource_blocked = set(self.get_blocked_periods_resource(resource))

        periods = maint_needed - maint_not_possible - resource_blocked

        free = [(1, period) for period in periods]
        return tl.TupList(free).to_start_finish(inst.compare_tups)

    def get_random_maint(self, resource, min_period, max_period, maint='M'):
        """
        Finds a random maintenance from all possible assignments in range
        :param str resource: resource (code) to search for maintenance
        :param str min_period: period (month) to start searching for start of maintenance
        :param str max_period: period (month) to end searching for start of maintenance
        :return: period (month) or none
        """
        last_period = self.instance.get_param('end')
        maint_data = self.instance.data['maintenances'][maint]
        duration = maint_data['duration_periods']
        max_end_maint = self.instance.shift_period(max_period, duration - 1)
        start_to_finish = \
            self.get_maintenance_candidates(
                resource, min_period=min_period, max_period=max_end_maint, maint=maint
            )
        if not len(start_to_finish):
            return None
        sizes = {st: len(self.instance.get_periods_range(st, end)) - duration
                 for (id, st, end) in start_to_finish}
        # we get all the possible starts that can happen in each piece of available periods
        maint_options = [self.instance.shift_period(st, t) for st, size in sizes.items()
                         if size >= 0 for t in range(size + 1)]
        last_piece = start_to_finish[-1]
        if last_piece[-1] == last_period:
            incomplete_maints = self.instance.get_periods_range(last_piece[1], max_period)
            maint_options.extend(incomplete_maints)
        if not len(maint_options):
            return None
        probs = np.array([(i+1) for i, _ in enumerate(maint_options)])
        probs = probs / sum(probs)
        start = np.random.choice(a=maint_options, p=probs)
        return start

    def assign_maintenance(self, resource, maint, maint_start):
        """
        Assigns a given maintenance to a resources at a specific time
        :param resource: resource to assign check
        :param maint: maintenance to assign check
        :param maint_start: the start period for the check
        :return: True
        """
        horizon_end = self.instance.get_param('end')
        maint_data = self.instance.data['maintenances'][maint]
        affected_maints = maint_data['affects']
        duration = maint_data['duration_periods']
        maint_end = min(self.instance.shift_period(maint_start, duration - 1), horizon_end)
        periods_maint = self.instance.get_periods_range(maint_start, maint_end)
        log.debug("{} gets {} maint: {} -> {}".
                  format(resource, maint, maint_start, maint_end))
        self.set_state_and_clean(resource, maint, periods_maint)
        for m in affected_maints:
            self.update_time_maint(resource, periods_maint, time='ret', maint=m)
            self.update_time_maint(resource, periods_maint, time='rut', maint=m)
        # it doesn't make sense to assign a maintenance after a maintenance
        if maint_end == self.instance.get_param('end'):
            # we assigned the last day to maintenance:
            # there is nothing to update.
            return True
        start_update_rt = self.instance.get_next_period(maint_end)
        for m in affected_maints:
            for time in self.instance.get_maint_rt(m):
                self.update_rt_until_next_maint(resource, start_update_rt, m, time)
        return True


    def get_free_periods_resource(self, resource):
        """
        Finds the list of periods (month) that the resource is available.
        :param str resource: resource code
        :return: periods (month)
        """
        dtype_date = 'U7'
        blocked = self.get_blocked_periods_resource(resource)
        return np.setdiff1d(
            np.fromiter(self.instance.get_periods(), dtype=dtype_date),
            blocked
        )

    def get_blocked_periods_resource(self, resource):
        """
        All periods where the resource is being used to a task or a maintenance
        :param resource:
        :return: list of periods
        """
        dtype_date = 'U7'
        states = self.solution.data['state_m'].get(resource, {}).items()
        periods_maint = np.array([], dtype = dtype_date)
        if len(states):
            periods_maint, maints = zip(*states)
            periods_maint = np.asarray(periods_maint, dtype = dtype_date)
            filt = ['M' in d.keys() for d in maints]
            periods_maint = periods_maint[filt]
        periods_task = np.fromiter(self.solution.data['task'].get(resource, {}), dtype=dtype_date)

        return list(periods_maint) + list(periods_task)

    def get_free_starts(self, resource, periods):
        """
        Gets a start-stop list of free periods for a resource.
        :param resource: resource to look for
        :param periods: list of periods to look for
        :return:
        """
        candidate_periods = \
            np.intersect1d(
            periods,
            self.get_free_periods_resource(resource)
        )
        if len(candidate_periods) == 0:
            return []

        ct = self.instance.compare_tups
        startend = tl.TupList([(1, p) for p in candidate_periods]).to_start_finish(ct)
        return startend.filter([1, 2])

    def update_time_maint(self, resource, periods, time='rut', maint='M'):
        """
        Updates ret or rut for a resource on certain periods for a given maintenance.
        It sets it to the maximum of that maintenance.
        :param str resource:
        :param list periods:
        :param str time:
        :param str maint:
        :return: True
        """
        value = self.instance.get_max_remaining_time(time=time, maint=maint)
        for period in periods:
            self.set_remainingtime(resource, period, time, value, maint)
        return True

    def del_maint(self, resource, period, maint=None):
        """
        removes a maintenance assign in a resource and a period
        :param resource:
        :param period:
        :param maint: maint to delete. if None: delete all
        :return:
        """
        try:
            periods = self.solution.data['state_m'][resource]
            maints = periods[period]
            if maint is not None:
                maints.pop(maint, None)
            if not len(maints) or maint is None:
                periods.pop(period, None)
        except KeyError:
            pass

    def set_state(self, *args, value, cat):
        """
        :param value: can be the state, can be 1
        :param cat: task, state, state_m
        :param args: normally: [resource, period]. Can also be: [resource, period, maint]
        :return:
        """
        self.solution.data[cat].set_m(*args, value=value)

        return True

    def set_state_and_clean(self, resource, maint, periods):
        """
        :param resource: resource to update
        :param maint: maintenance name to update
        :param periods: periods to update (list)
        :return: True
        """
        maint_data = self.instance.data['maintenances'][maint]
        affected_maints = maint_data['affects']
        for period in periods:
            self.set_state(resource, period, maint, cat='state_m', value=1)
        # Delete auxiliary maintenances if any in relevant periods
        for m in affected_maints:
            if m == maint:
                continue
            for period in periods:
                self.del_maint(resource, period, m)
        return True

    def get_maintenance_periods_resource(self, resource, maint='M'):
        """
        Given a resource and a maintenance, it returns the periods
        where that maint starts
        :param resource: resource to look for
        :param maint: maintenance to look for
        :return:
        """
        periods = [(1, k) for k, v in self.solution.data['state_m'].get(resource, {}).items() if maint in v]
        result = tl.TupList(periods).to_start_finish(self.instance.compare_tups)
        return result.filter([1, 2])

    def get_next_maintenance(self, resource, min_start, maints=None):
        """
        Looks for the inmediate next maintenance start
        :param resource: resource to look
        :param min_start: date to start looking
        :param set maints: maintenances to look for
        :return:
        """
        last = self.instance.get_param('end')
        if maints is None:
            maints = {'M'}
        period = min_start
        while period <= last:
            states = self.solution.get_period_state(resource, period, 'state_m')
            if states is not None and len(states.keys() & maints):
                return period
            period = self.instance.get_next_period(period)
        return None



if __name__ == "__main__":
    #  see ../scripts/exec_heur.py
    pass
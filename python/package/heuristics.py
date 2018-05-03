import package.auxiliar as aux
import package.data_input as di
import numpy as np
import package.tests as test
import package.instance as inst
import package.solution as sol
# import pprint as pp
import pandas as pd
import package.tests as exp


class Greedy(test.Experiment):

    def __init__(self, instance, options=None):

        if options is None:
            options = {}
        solution = sol.Solution({'state': {}, 'task': {}})
        super().__init__(instance, solution)

        self.solution.data['aux'] = {'ret': {}, 'rut': {}}
        first_period = aux.get_prev_month(self.instance.get_param('start'))
        last_period = self.instance.get_param('end')
        resources = list(self.instance.get_resources())

        # we preassign ruts and rets.
        for time in ['rut', 'ret']:
            initial_state = self.instance.get_initial_state(self.label_rt(time))
            for resource in resources:
                self.set_remainingtime(resource, first_period, time, initial_state[resource])

        periods_to_update = {r: self.instance.get_periods() for r in resources}

        # we preassign fixed maintenances:
        fixed_maintenances = self.instance.get_fixed_maintenances(dict_key='resource')
        for resource, periods in fixed_maintenances.items():
            periods_to_update[resource] = \
                aux.get_months(aux.get_next_month(periods[-1]), last_period)

            for period in periods:
                self.set_maint(resource, period)
            self.update_time_maint(resource, periods, time='ret')
            self.update_time_maint(resource, periods, time='rut')

        for t in ['rut', 'ret']:
            for r in resources:
                self.update_time_usage(r, periods_to_update[r], time=t)

        def_options = {'print': True}
        self.options = {**def_options, **options}

    def solve(self):
        # 1. Choose a mission.
        # 2. Choose candidates for that mission.
        # 3. Start assigning candidates to the mission's months.
            # Here the selection order could be interesting
            #  Assign maintenances when needed to each aircraft.
        # 4. When finished with the mission, repeat with another mission.
        check_candidates = self.instance.check_enough_candidates()
        tasks_sorted = sorted(check_candidates.items(), key=lambda x: x[1][1])
        # mission by mission: we find candidates and assign them
        for task, content in tasks_sorted:
            self.fill_mission(task)
        # at the end, there can be resources that need maintenance because of ret.
        needs = [k for k, v in self.check_elapsed_consumption().items() if v == -1]
        for res, period in needs:
            print('resource {} needs a maintenance in {}'.format(res, period))
            self.find_assign_maintenance(res, aux.shift_month(period, -1))
        return self.solution

    def fill_mission(self, task):
        """
        This function assigns all the necessary resources to a given task
        It edits the solution object inside the experiment
        :param task: a task code to satisfy
        :return: nothing
        """
        duration = self.instance.get_param('maint_duration')
        dtype = 'U7'
        # get task candidates: we copy the list.
        task_resources = self.instance.get_task_candidates()
        resource_num_tasks = \
            aux.dict_to_lendict(
                aux.dict_list_reverse(task_resources)
            )
        candidates = task_resources[task]
        candidates.sort(key=lambda x: resource_num_tasks[x])
        # get task periods to satisfy:
        rem_resources = \
            aux.dicttup_to_dictdict(
                self.instance.get_task_period_needs()
            )[task]

        while len(rem_resources) > 0 and len(candidates) > 0:
            candidate = candidates[0]
            # get free periods for candidate
            periods_task = \
                self.get_free_starts(candidate, np.fromiter(rem_resources, dtype=dtype))
            if len(periods_task) == 0:
                # consider eliminating the resource from the list.
                # if all its periods are 'used'
                candidates.pop(0)
                continue
            maint_need = ''
            # for each period of consecutive months
            for start, end in periods_task:
                # here we do all the checks to see if we can assign tasks
                # and how many periods we can assign it to.
                last_month = self.find_assign_task(candidate, start, end, task)
                for period in aux.get_months(start, last_month):
                    # we assign all found periods
                    rem_resources[period] -= 1
                    # delete periods that have 0 rem_resources to assign
                    if rem_resources[period] == 0:
                        rem_resources.pop(period)
                if maint_need == '' and last_month < end:
                    # we register (only) the first need of maintenance:
                    maint_need = aux.shift_month(last_month, -duration+1)
            if maint_need == '':
                # if the resource has no need for maintenance yet, we don't attempt one
                continue
            if self.options['print']:
                print("resource {} needs a maintenance after period {}".format(candidate, maint_need))
            # find soonest period to start maintenance:
            result = self.find_assign_maintenance(candidate, maint_need)
            if not result:
                # the maintenance failed: we pop the candidate because it is most probably useless.
                candidates.pop(0)

        return

    def fix_over_maintenances(self):
        # sometimes the solution includes a 12 period maintenance
        # this function should delete the first of the two periods.
        return

    def find_assign_task(self, resource, start, end, task):
        periods_to_assign = self.check_assign_task(resource, aux.get_months(start, end), task)
        for period in periods_to_assign:
            self.expand_resource_period(self.solution.data['task'], resource, period)
            self.solution.data['task'][resource][period] = task
        last_month = end
        if len(periods_to_assign) == 0:
            # if there was nothing assigned: there is nothing to do
            return aux.shift_month(start, -1)
        if periods_to_assign[-1] != end:
            # if last assigned period corresponds with the end:
            # this means we did not have problems.
            # if not, a maintenance is needed after that period.
            last_month = periods_to_assign[-1]
        # here, the updating of ret and rut is done.
        # It is done until the next maintenance or the end
        next_maint = self.get_next_maintenance(resource, end)
        if next_maint is None:
            periods_to_update = aux.get_months(start, self.instance.get_param('end'))
        else:
            periods_to_update = aux.get_months(start, next_maint)
        self.update_time_usage(resource, periods=periods_to_update, time='rut')
        # self.update_time_usage(resource, periods=periods_to_update, time='ret')
        return last_month

    def find_assign_maintenance(self, resource, maint_need):
        """
        Tries to find the soonest maintenance in the planning horizon
        for a given resource.
        :param resource: resource to find maintenance
        :param maint_need: date when the resources needs the maintenance
        :return:
        """
        horizon_end = self.instance.get_param('end')
        duration = self.instance.get_param('maint_duration')
        maint_start = self.get_soonest_maint(resource, maint_need)
        if maint_start is None:
            # this means that we cannot assign tasks BUT
            # we cannot assign maintenance either :/
            # we should then take out the candidate:
            if self.options['print']:
                print("resource {} has no candidate periods for maintenance".format(resource))
            return False
        maint_end = min(aux.shift_month(maint_start, duration - 1), horizon_end)
        periods_maint = aux.get_months(maint_start, maint_end)
        periods_to_update = periods_maint
        next_maint = self.get_next_maintenance(resource, maint_start)
        last_maint_prev = self.get_next_maintenance(resource, maint_start, previous=True)
        if last_maint_prev is not None and last_maint_prev >= maint_need:
            if self.options['print']:
                print("resource {} has already a maintenance {} after the need {}.".
                      format(resource, last_maint_prev, maint_need))
            return False
        if next_maint is not None:
            # we need to take out the old one, that happens *after* the new.
            # and choose carefully the periods to update.
            # TODO: this swap is not checking everything: sometimes make infeasible choices
            if self.options['print']:
                print("resource {} could swap maintenances: {} to {}".format(resource, next_maint, maint_start))
            old_maint_end = aux.shift_month(next_maint, duration - 1)
            for period in aux.get_months(next_maint, old_maint_end):
                self.del_maint(resource, period)
        if self.options['print']:
            print("resource {} will get a maintenance in periods {} -> {}".format(resource, maint_start, maint_end))
        start_update_rt = aux.get_next_month(maint_end)
        end_update_rt = self.get_next_maintenance(resource, start_update_rt)
        if end_update_rt is None:
            end_update_rt = horizon_end
        for period in periods_maint:
            self.set_maint(resource, period)
        self.update_time_maint(resource, periods_to_update, time='ret')
        self.update_time_maint(resource, periods_to_update, time='rut')
        # it doesn't make sense to assign a maintenance after a maintenance
        if maint_end == self.instance.get_param('end'):
            # we assigned the last day to maintenance:
            # there is nothing to update.
            return True
        self.update_time_usage(resource, aux.get_months(start_update_rt, end_update_rt), time='ret')
        self.update_time_usage(resource, aux.get_months(start_update_rt, end_update_rt), time='rut')
        return True

    def check_assign_task(self, resource, periods, task):
        """
        Calculates the amount of periods it's possible to assign
        a given task to a resource.
        Based on the usage status of the resource.
        :param resource: candidate to assign a task
        :param periods: periods to try to assign the task
        :param task: task to assign to resource
        :return: subset of continous periods to assign to the resource
        """
        consumption = self.instance.data['tasks'][task]['consumption']
        start = periods[0]
        end = periods[-1]
        number_periods = len(periods)

        horizon_end = self.instance.get_param('end')
        next_maint = self.get_next_maintenance(resource, end)
        if next_maint is not None:
            before_maint = aux.shift_month(next_maint, -1)
            rut = self.solution.data['aux']['rut'][resource][before_maint]
        else:
            rut = self.solution.data['aux']['rut'][resource][horizon_end]
        # ret = self.solution.data['aux']['ret'][resource][end]
        number_periods_ret = self.solution.data['aux']['ret'][resource][start] - 2
        number_periods_rut = int(rut // consumption)
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

    def get_soonest_maint(self, resource, min_period, maint_duration=6):
        """
        Finds the soonest possible maintenance to assign to a resource.
        :param resource: resource (code) to search for maintenance
        :param min_period: period (month) starting period to search for date
        :param maint_duration: duration in number periods of the maintenance
        :return: period (month) or none
        """
        horizon_end = self.instance.get_param('end')
        periods_to_search = \
            np.intersect1d(self.get_free_periods_maint(),
                           self.get_free_periods_resource(resource))
        free = [(1, period) for period in periods_to_search if period >= min_period]
        for (id, st, end) in aux.tup_to_start_finish(free):
            # the period needs to be as least the size of the maintenance
            # alternative: the end is the end of the horizon.
            if end == horizon_end or len(aux.get_months(st, end)) >= maint_duration:
                return st
        return None

    def get_free_periods_resource(self, resource):
        """
        Finds the list of periods (month) that the resouce is available.
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
        startend = aux.tup_to_start_finish([(1, p) for p in candidate_periods])
        return aux.tup_filter(startend, [1, 2])

    def update_time_maint(self, resource, periods, time='rut'):
        value = 'max_' + self.label_rt(time) + '_time'
        for period in periods:
            self.set_remainingtime(resource, period, time, self.instance.get_param(value))
        return True

    def del_maint(self, resource, period):
        if resource in self.solution.data['state']:
            return self.solution.data['state'][resource].pop(period, None)

    def set_maint(self, resource, period, value='M'):
        self.expand_resource_period(self.solution.data['state'], resource, period)
        self.expand_resource_period(self.solution.data['aux']['rut'], resource, period)
        self.expand_resource_period(self.solution.data['aux']['ret'], resource, period)

        self.solution.data['state'][resource][period] = value
        return True

    def get_maintenances(self, resource):
        return self.solution.data['state'].get(resource, {}).keys()

    def get_maintenance_periods_resource(self, resource):
        periods = [(1, k) for k, v in self.solution.data['state'].get(resource, {}).items() if v == 'M']
        result = aux.tup_to_start_finish(periods)
        return [(k[1], k[2]) for k in result]

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
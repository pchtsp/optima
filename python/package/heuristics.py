import package.aux as aux
import package.data_input as di
import numpy as np
import package.tests as test
import package.instance as inst
import package.solution as sol


class Greedy(test.Experiment):

    def __init__(self, instance, solution, options=None):
        super().__init__(instance, solution)
        for resource in self.instance.get_resources():
            self.update_time_usage(resource, time='ret')
            self.update_time_usage(resource, time='rut')

        self.options = options

    def fill_mission(self, task):
        dtype = 'U7'
        # get task candidates: we copy the list.
        candidates = self.instance.data['tasks'][task]['candidates'][:]
        # get task periods to satisfy:
        rem_resources = \
            aux.dicttup_to_dictdict(
                self.instance.get_task_period_needs()
            )[task]

        while len(rem_resources) > 0 and len(candidates) > 0:
            # candidate = 'A124'
            candidate = candidates[0]
            # get free periods for candidate
            periods_task = \
                self.get_free_starts(candidate, np.fromiter(rem_resources, dtype=dtype))

            # consider eliminating the resource from the list.
            # if all its periods are 'used'
            if len(periods_task) == 0:
                candidates.pop(0)
                continue
            print('candidate={}\ntask={}\nperiods={}'.format(candidate, task, periods_task))
            maint_period = ''
            # TODO: this is still wrong:
            for start, end in periods_task:
                for period in aux.get_months(start, end):
                    result = self.check_set_task(candidate, period, task)
                    if not result:
                        maint_period = period
                        break
                    rem_resources[period] -= 1
                self.update_time_usage(candidate, period, self.get_next_maintenance(candidate, period), 'rut')

            if maint_period != '':
                # find soonest period to start maintenance:
                maint_start = self.get_soonest_maint(candidate, maint_period)
                if maint_start is None:
                    # this means that we cannot assign tasks BUT
                    # we cannot assign maintenance either :/
                    # we should then take out the candidate:
                    candidates.pop(0)
                    continue
                maint_end = aux.shift_month(maint_start, 5)
                periods_maint = aux.get_months(maint_start,  maint_end)
                for period in periods_maint:
                    self.set_maint(candidate, period)
                self.update_time_maint(maint_start, maint_end)
                next_month = aux.get_next_month(maint_end)
                self.update_time_usage(candidate, next_month, 'ret')
                self.update_time_usage(candidate, next_month, 'rut')

            # delete periods that have 0 rem_resources to assign
            rem_resources = {p: v for p, v in rem_resources.items() if v > 0}

        return

    def get_soonest_maint(self, resource, min_period):
        free = [(1, period) for period in self.get_free_periods(resource) if period >= min_period]
        maint_starts = \
            [st for (id, st, end) in aux.tup_to_start_finish(free)
             if len(aux.get_months(st, end)) >= 6]
        if len(maint_starts) > 0:
            return min(maint_starts)
        return None

    def get_free_periods(self, resource):
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
            self.get_free_periods(resource)
        )
        if len(candidate_periods) == 0:
            return []
        startend = aux.tup_to_start_finish((1, p) for p in candidate_periods)
        return aux.tup_filter(startend, [1, 2])

    def update_time_maint(self, resource, period_start=None, period_end=None, time='rut'):
        # next_period = period_start
        # in_maint = self.get_maintenances(resource)
        value = 'max_' + self.label_rt(time) + '_time'
        for next_period in aux.get_months(period_start, period_end):
            self.set_remainingtime(resource, next_period, time, self.instance.get_param(value))
        return True

    def set_maint(self, resource, period):
        self.expand_resource_period(self.solution.data['state'], resource, period)
        self.expand_resource_period(self.solution.data['aux']['rut'], resource, period)
        self.expand_resource_period(self.solution.data['aux']['ret'], resource, period)

        self.solution.data['state'][resource][period] = 'M'
        return True

    def check_set_task(self, resource, period, task):
        # check
        # ret and rut
        if self.solution.data['aux']['rut'][resource][period] < \
                self.instance.get_tasks('consumption')[task] \
                or self.solution.data['aux']['ret'][resource][period] < 2:
            return False

        # assign:
        self.expand_resource_period(solution.data['task'], resource, period)
        self.solution.data['task'][resource][period] = task
        return True

    def get_maintenances(self, resource):
        return self.solution.data['state'].get(resource, {}).keys()

    def get_maintenance_periods_resource(self, resource):
        periods = [(1, k) for k, v in self.solution.data['state'].get(resource, {}).items() if v == 'M']
        result = aux.tup_to_start_finish(periods)
        return [(k[1], k[2]) for k in result]

    def get_next_maintenance(self, resource, min_start):
        start_end = self.get_maintenance_periods_resource(resource)
        maints = [st for st, end in start_end if st >= min_start]
        if len(maints) > 0:
            return min(maints)
        else:
            return None

def heuristic(instance, options):
    # 1. Choose a mission.
    # 2. Choose candidates for that mission.
    # 3. Start assigning candidates to the mission's months.
        # Here the selection order could be interesting
        #  Assign maintenances when needed to each aircraft.
    # 4. When finished with the mission, repeat with another mission.

    # MAIN VARS:
    solution = sol.Solution({'state': {}, 'task': {}})
    heur = Greedy(instance, solution)
    heur.instance.get_tasks()
    heur.fill_mission('O10')

    return False




if __name__ == "__main__":
    path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712182300/"
    model_data = di.load_data(path + "data_in.json")
    # this was for testing purposes

    instance = inst.Instance(model_data)
    solution = sol.Solution({'state': {}, 'task': {}})
    heur = Greedy(instance, solution)
    # heur.instance.get_tasks()
    # heur.expand_resource_period(heur.solution.data['task'], 'A100', '2017-01')
    tasks = list(heur.instance.get_tasks().keys())
    # tasks = ['O5']
    for task in tasks:
        heur.fill_mission(task)
    heur.solution.print_solution("/home/pchtsp/Downloads/calendar_temp1.html")
    checks = heur.check_solution()
    print([k for k, v in checks.items() if len(v) > 0])
    # {k: v for k, v in checks.items() if len(v) > 0}.keys()
    # checks.keys()
    # path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712190002/"
    # self = Greedy.from_dir(path)
    #
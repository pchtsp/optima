import math
import pulp as pl
import solvers.config as conf
import package.experiment as exp
import pytups.tuplist as tl
import pytups.superdict as sd


class ModelMissions(exp.Experiment):

    def __init__(self, instance, solution):

        super().__init__(instance, solution)

        self.task = sd.SuperDict()
        self.model = None

    def solve(self, options):

        resources = self.instance.get_resources()
        candidates = self.instance.get_task_candidates()
        res_task = sd.SuperDict.from_dict(candidates).to_tuplist()
        tasks = self.instance.get_tasks()
        maintenances = self.instance.get_maintenances()

        # discard task-resources because of M or VS.
        M_and_VS_assignments = \
            self.solution.get_state().\
            keys_tl().\
            vfilter(lambda v: v[2] in {'M', 'VS'})
        resource_busy = M_and_VS_assignments.take([0, 1]).to_dict(1).vapply(set)
        tasks_periods = self.instance.get_task_period_list().to_dict(1).vapply(set)
        # get forbidden resource-task:
        forbidden1 = \
            resource_busy.\
            vapply(lambda v: [t for t, _list in tasks_periods.items()
                              if _list & v]).\
                to_tuplist()
        res_task = res_task.set_diff(forbidden1)

        # prepare cycles for flight hours constraints
        # we do it for each maintenance type (M and VS)
        resource_cycles = sd.SuperDict()
        for maint, check in [('M', {'M'}), ('VS', {'M', 'VS'})]:
            resource_cycles[maint] = \
                self.get_maintenance_periods(state_list=check). \
                to_dict(result_col=[1, 2]). \
                vapply(sorted). \
                fill_with_default(keys=resources, default=[]). \
                vapply(self.get_maintenance_cycles)

        self.task = \
            pl.LpVariable.dicts(name="task",
                                indexs=res_task,
                                lowBound=0, upBound=1,
                                cat=pl.LpInteger)
        self.task = sd.SuperDict.from_dict(self.task)

        self.model = model = pl.LpProblem("MFMP_v0004", pl.LpMinimize)

        # the sum of flight hours that are assigned need to fall in
        # between the hour visits.
        # for each maintenance, resource and cycle: the missions that are available.
        flight_hours = tasks.get_property('consumption')
        hour_limit = maintenances.get_property('max_used_time')
        for maint, limit in hour_limit.items():
            for r, _cycles in resource_cycles[maint].items():
                for cycle in _cycles:
                    start, stop = cycle
                    # tasks that the resource can do
                    # that start inside the cycle.
                    _tasks = res_task.to_dict(1)[r].vfilter(lambda v: start <= v['start'] <= stop)
                    model += pl.lpSum(self.task[r, v]*flight_hours[v] for v in _tasks) <= limit

        # we pre-calculate the remaining fleet per period
        # for each period, we calculate the non assigned aircraft's demand for maintenance.
        usage = maintenances.vfilter(lambda v: v['type']=='2').get_property('capacity_usage')

        # per period, what maintenance operations are scheduled
        maint_period_res = \
            self.solution.get_state().\
                keys_tl().take([1, 2, 0]).\
                vfilter(lambda v: v[1] in usage).\
                to_dict([1, 2])
        # since we know missions, we can calculate the
        # capacity that remains in base.
        capacities = self.instance.get_capacity_calendar().to_dictdict().get('2', {})
        fleet_size = len(resources)
        rem_capacity_period =\
            self.instance.get_total_period_needs().\
            apply(lambda k, v: (fleet_size-v)/fleet_size*capacities[k]).\
            vapply(lambda v: round(v))

        # we calculate, for each period,
        # what are the assignments that affect it.
        res_task_period = \
            res_task. \
                to_dict(None). \
                vapply(lambda v: tasks_periods[v[0]]). \
                vapply(list).to_tuplist().take([2, 1, 0]).\
                to_dict(2).to_dictdict()

        # finally, we create the constraint guaranteeing capacity.
        for t, _capacity in rem_capacity_period.items():
            _res_task = res_task_period[t]
            _res_maint_list = maint_period_res[t]
            model += pl.lpSum((1 - self.task[r, v])*usage[m] for m, r in _res_maint_list
                     for v in _res_task[r]) <= _capacity

        config = conf.Config(options)
        result = config.solve_model(model)
        if result is None:
            return None

        self.solution = self.get_solution()

        return self.solution

    def get_solution(self):

        _range_f = self.instance.get_periods_range
        assignments = self.task.vfilter(pl.value).keys_tl()
        tasks = self.instance.get_tasks()
        _task = sd.SuperDict()
        for a, v in assignments:
            t1, t2 = tasks[v]['start'], tasks[v]['end']
            for t in _range_f(t1, t2):
                _task[a, t] = v

        sol_data = self.copy_solution()
        sol_data['task'] = _task

        return sol.Solution(sol_data)


if __name__ == '__main__':
    import package.params as params
    import data.template_data as tp
    import package.instance as inst
    import package.solution as sol
    path = params.PATHS['data'] + 'template/Test3/'
    model_data = tp.import_input_template(path+ '/template_in.xlsx')
    sol_data = tp.import_output_template(path + 'template_out.xlsx')
    self = ModelMissions(instance = inst.Instance(model_data),
                         solution=sol.Solution(sol_data))
    # self = ModelMissions.from_dir(path)
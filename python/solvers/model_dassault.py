import math
import pulp as pl
import solvers.config as conf
import package.experiment as exp
import pytups.tuplist as tl
import pytups.superdict as sd
import package.solution as sol


class ModelMissions(exp.Experiment):

    def __init__(self, instance, solution):

        super().__init__(instance, solution)

        self.task = sd.SuperDict()
        self.model = None

    def solve(self, options):

        resources = self.instance.get_resources()
        candidates = self.instance.get_task_candidates()
        res_task = sd.SuperDict.from_dict(candidates).to_tuplist().take([1, 0])
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

        # we calculate, for each period,
        # what are the assignments that affect it.
        task_period_rest = \
            res_task. \
                to_dict(None). \
                vapply(lambda v: tasks_periods[v[1]]). \
                vapply(list).to_tuplist().take([2, 0, 1]).\
                to_dict(2)

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

        self.model = pl.LpProblem("MFMP_v0004", pl.LpMinimize)

        # only one active mission assignment per aircraft.
        res_nb_mission = \
            task_period_rest.\
            vfilter(lambda v: len(v)>1).\
            apply(lambda k, v: pl.lpSum(self.task[k[1], _v] for _v in v)).\
                to_tuplist().take([1, 0, 2]).to_dict(2, is_list=False)
        for v in res_nb_mission.values():
            self.model += v <= 1

        # number of resources per mission
        res_task_v = res_task.to_dict(0)
        task_num_resource = self.instance.get_tasks('num_resource')
        res_per_task = res_task_v.apply(lambda k, v: pl.lpSum(self.task[_v, k] for _v in v))
        for k, v in res_per_task.items():
            self.model += v == task_num_resource[k]

        # the sum of flight hours that are assigned need to fall in
        # between the hour visits.
        # for each maintenance, resource and cycle: the missions that are available.
        flight_hours = tasks.get_property('consumption')
        hour_limit = maintenances.get_property('max_used_time').vfilter(lambda v: v)
        res_task_r = res_task.to_dict(1)
        task_start = tasks.get_property('start')
        for maint, limit in hour_limit.items():
            for r, _cycles in resource_cycles[maint].items():
                for cycle in _cycles:
                    start, stop = cycle
                    # tasks that the resource can do
                    # that start inside the cycle.
                    _tasks = res_task_r.get(r)
                    if not _tasks:
                        continue
                    _tasks = _tasks.vfilter(lambda v: start <= task_start[v] <= stop)
                    self.model += pl.lpSum(self.task[r, v]*flight_hours[v] for v in _tasks) <= limit

        # we pre-calculate the remaining fleet per period
        # for each period, we calculate the non assigned aircraft's demand for maintenance.
        usage = maintenances.vfilter(lambda v: v['type'] == '2').get_property('capacity_usage')

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

        # finally, we create the constraint guaranteeing capacity.
        task_period_rest_t = task_period_rest.to_dictdict()
        for t, _res_task in task_period_rest_t.items():
            # we iterate over periods that have at least an active mission.
            if t not in maint_period_res:
                # there are no maintenances in this period.
                # we do not check
                continue
            _res_maint_list = maint_period_res[t]
            _capacity  = rem_capacity_period[t]
            self.model += pl.lpSum((1 - res_nb_mission.get((r, t), 0))*usage[m]
                                   for m, r in _res_maint_list) <= _capacity
        config = conf.Config(options)
        self.model.writeLP(options['path']+'formulation.lp')
        result = config.solve_model(self.model)
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
        sol_data['task'] = _task.to_dictdict()

        return sol.Solution(sol_data)


if __name__ == '__main__':
    import package.params as params
    import data.template_data as tp
    import package.instance as inst
    import data.data_input as di
    import reports.gantt as gantt

    path = params.PATHS['data'] + 'template/Test3/'
    model_data = tp.import_input_template(path+ '/template_in.xlsx')
    sol_data = tp.import_output_template(path + 'template_out.xlsx')
    self = ModelMissions(instance = inst.Instance(model_data),
                         solution=sol.Solution(sol_data))
    params.OPTIONS['solver'] = 'CBC'
    solution = self.solve(params.OPTIONS)
    output_path = params.OPTIONS['path']
    # di.export_data(output_path, self.instance.data, name="data_in", file_type='json', exclude_aux=True)
    di.export_data(output_path, self.instance.data, name="data_in", file_type='json', exclude_aux=True)
    di.export_data(output_path, self.solution.data, name="data_out", file_type='json', exclude_aux=True)
    gantt.make_gantt_from_experiment(path=output_path)
    # solution.get_tasks()
    # self = ModelMissions.from_dir(path)
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
        """
        Solves a mip model given a FMP with fixed maintenances
        :param options:
        :return:
        """

        resources = self.instance.get_resources()
        candidates = self.instance.get_task_candidates()
        res_task = sd.SuperDict.from_dict(candidates).to_tuplist().take([1, 0])
        tasks = self.instance.get_tasks()
        maintenances = self.instance.get_maintenances()
        periods = self.instance.get_periods()

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

        # We delete the tasks, because we need
        # to calculate previous ruts
        self.solution.data['task'] = sd.SuperDict()
        # so they remain coherent, we also clean the new_defaults
        self.solution.data['new_default'] = sd.SuperDict()

        prev = self.instance.get_prev_period
        rut = self.set_remaining_usage_time_all('rut')
        get_rut = lambda r, p: rut['M'][r][p]

        # we get M cycles that dictate waste flight hours
        rem_rut_cycle =\
            resource_cycles['M'].\
            to_tuplist().\
            to_dict(None).\
            vapply(lambda v: get_rut(v[0], prev(v[-1])))

        # VARIABLES
        self.task = \
            pl.LpVariable.dicts(name="task",
                                indices=res_task,
                                lowBound=0, upBound=1,
                                cat=pl.LpInteger)
        self.task = sd.SuperDict.from_dict(self.task)
        self.reduction = pl.LpVariable.dicts('reduction', indices=rem_rut_cycle, lowBound=0)
        self.reduction = sd.SuperDict.from_dict(self.reduction)

        # penalty for over-assigning missions to same aircraft
        slots = [s for s in range(3)]
        _options = [(r, s) for r in resources for s in slots]
        self.num_missions_range = pl.LpVariable.dicts('num_missions_range', indices=_options, lowBound=0, upBound=1)
        self.num_missions_range = sd.SuperDict.from_dict(self.num_missions_range)
        for r in resources:
            self.num_missions_range[r, slots[-1]].upBound = None
        excess_cost = {s: (s+1)**2 - 1 for s in slots}

        # penalty for over-capacity:
        self.excess_capacity = pl.LpVariable.dict('excess_capacity', indices=periods, lowBound=0, upBound=None)
        self.excess_capacity = sd.SuperDict.from_dict(self.excess_capacity)

        self.model = pl.LpProblem("MFMP_v0004", pl.LpMinimize)

        # OBJECTIVE FUNCTION
        # minimize reductions in default hours
        cost_of_excess = self.num_missions_range.kvapply(lambda k, v: v * excess_cost[k[1]]).values()
        self.model += \
            pl.lpSum(self.reduction.values()) + \
            pl.lpSum(cost_of_excess)*100 + \
            pl.lpSum(self.excess_capacity.values())*1000

        # CONSTRAINTS
        # only one active mission assignment per aircraft.
        res_nb_mission = \
            task_period_rest.\
            vfilter(lambda v: len(v) > 1).\
            kvapply(lambda k, v: pl.lpSum(self.task[k[1], _v] for _v in v)).\
                to_tuplist().take([1, 0, 2]).to_dict(2, is_list=False)
        for v in res_nb_mission.values():
            self.model += v <= 1

        # number of resources per mission
        res_task_v = res_task.to_dict(0)
        task_num_resource = self.instance.get_tasks('num_resource')
        res_per_task = res_task_v.kvapply(lambda k, v: pl.lpSum(self.task[_v, k] for _v in v))
        for k, v in res_per_task.items():
            self.model += v == task_num_resource[k]

        # number of missions per resource
        task_per_resource = res_task.to_dict(1).kvapply(lambda k, v: pl.lpSum(self.task[k, _v] for _v in v))
        excess = self.num_missions_range.to_tuplist().take([0, 2]).to_dict(1).vapply(pl.lpSum)
        for r, v in task_per_resource.items():
            self.model += v == excess[r]

        # the sum of flight hours that are assigned need to fall in
        # between the hour limit for that visit.
        # for each maintenance, resource and cycle: the missions that are available.
        flight_hours = tasks.get_property('consumption')
        hour_limit = maintenances.get_property('max_used_time').vfilter(lambda v: v)
        res_task_r = res_task.to_dict(1)
        task_start = tasks.get_property('start')
        task_end = tasks.get_property('end')
        _dist = self.instance.get_dist_periods
        task_total_hours = task_start.kvapply(lambda k, v: _dist(v, task_end[k])*flight_hours[k])
        for maint, limit in hour_limit.items():
            for r, _cycles in resource_cycles[maint].items():
                for start, stop in _cycles:
                    # tasks that the resource can do
                    # that start inside the cycle.
                    _tasks = \
                        res_task_r.get(r, tl.TupList()).\
                        vfilter(lambda v: start <= task_start[v] <= stop)
                    if not _tasks:
                        continue
                    total_hours_missions = pl.lpSum(self.task[r, v]*task_total_hours[v] for v in _tasks)
                    # limit total amount of hours
                    self.model += total_hours_missions <= limit
                    if (r, start, stop) in self.reduction:
                        remaining_hours = rem_rut_cycle[r, start, stop]
                        self.model += self.reduction[r, start, stop] >= total_hours_missions - remaining_hours

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
        remaining_capacity_calendar = self.instance.dassault_remaining_capacity()
        rem_capacity_period2 = remaining_capacity_calendar.to_dictdict().get('2', {})

        # finally, we create the constraint guaranteeing capacity.
        task_period_rest_t = task_period_rest.to_dictdict()
        for t, _res_task in task_period_rest_t.items():
            # we iterate over periods that have at least an active mission.
            if t not in maint_period_res:
                # there are no maintenances in this period.
                # we do not check
                continue
            _res_maint_list = maint_period_res[t]
            _capacity  = rem_capacity_period2[t]
            self.model += pl.lpSum((1 - res_nb_mission.get((r, t), 0))*usage[m]
                                   for m, r in _res_maint_list) - self.excess_capacity[t] <= _capacity

        config = conf.Config(options)
        if options.get('writeMPS', False):
            self.model.writeMPS(filename=options['path'] + 'formulation.mps')
        if options.get('writeLP', False):
            self.model.writeLP(filename=options['path'] + 'formulation.lp')

        result = config.solve_model(self.model)
        if result != 1:
            print("Model resulted in non-feasible status: {}".format(result))
            return None
        print('model solved correctly')
        # self.reduction.vapply(pl.value)
        # self.num_missions_range.vapply(pl.value)
        # self.excess_capacity.vapply(pl.value)
        # we got a solution, now we need to prepare something to cheat the default_usage of
        # aircraft that did missions.

        self.solution = self.get_solution()

        return self.solution

    def get_solution(self):
        """
        Reads the MIP variables contents after solving and generates a solution
        :return: formatted solution
        """

        _range_f = self.instance.get_periods_range
        assignments = self.task.vfilter(pl.value).keys_tl()
        tasks = self.instance.get_tasks()
        task_var = sd.SuperDict()
        for a, v in assignments:
            t1, t2 = tasks[v]['start'], tasks[v]['end']
            for t in _range_f(t1, t2):
                task_var[a, t] = v

        sol_data = self.copy_solution()
        sol_data['task'] = task_var.to_dictdict()

        # we will try only with large maintenances.
        # maint, check in [('M', {'M'}), ('VS', {'M', 'VS'})]
        new_default_hours = sd.SuperDict()
        for maint, check in [('VS', {'M', 'VS'}), ('M', {'M'})]:
            new_default_hours = self.edit_default_hours_cycle(maint, check, task_var, new_default_hours)

        sol_data['new_default'] = new_default_hours
        return sol.Solution(sol_data)

    def edit_default_hours_cycle(self, maint, check, task_solution, new_default_hours):
        """

        :param maint: M or VS
        :param check: {'M'} or {'M', 'VS'}
        :param task_solution: assignments of missions
        :param new_default_hours: modified hours dictionary to update
        :return: new_default_hours
        """

        # we add in aux some information
        # to discount the number of flight hours depending on the mission
        insta = self.instance
        tasks = insta.get_tasks()
        consumption = tasks.get_property('consumption')
        total_consumption = \
            self.instance.get_task_period_list(). \
                to_dict(1).to_lendict(). \
                kvapply(lambda k, v: v * consumption[k])

        resources = insta.get_resources()
        resource_cycles = \
            self.get_maintenance_periods(state_list=check). \
            to_dict(result_col=[1, 2]). \
            vapply(sorted). \
            fill_with_default(keys=resources, default=[]). \
            vapply(self.get_maintenance_cycles)

        # cycles_all = resource_cycles.to_tuplist()
        first, last = insta.get_start_end()
        # we calculate which mission assignment is in which resource cycle.
        task_start = task_solution.to_tuplist().to_dict(result_col=1).vapply(lambda v: v[0])
        cycle_tasks = tl.TupList()
        for (resource, mission), _task_start in task_start.items():
            for start, end in resource_cycles[resource]:
                if start <= _task_start <= end:
                    cycle_tasks.add(resource, start, end, mission)

        # we sum the consumption of missions for each cycle.
        cycle_missionHours = \
            cycle_tasks.to_dict(result_col=3).\
            vapply(lambda v: v.vapply(lambda vv: total_consumption[vv])).\
            vapply(sum)

        _range = lambda st, stp: set(insta.get_periods_range(st, stp))

        # we get the periods the resource is in tasks in the cycle.
        mission_periods = sd.SuperDict()
        for resource, start, stop, v in cycle_tasks:
            cycle = (resource, start, stop)
            if cycle not in mission_periods:
                mission_periods[cycle] = set()
            mission_periods[cycle] |= _range(tasks[v]['start'], tasks[v]['end'])

        # all periods on a cycle
        cycle_allPeriods = cycle_missionHours.kapply(lambda k: _range(k[1], k[2]))

        # we also consider periods already fixed as "missions":
        res_fixedPeriods = new_default_hours.vapply(set)

        # we do the same for the fixed default hours:
        cycle_FixeDefPeriods = \
            cycle_allPeriods.\
            kvapply(lambda k, v: v & res_fixedPeriods.get(k[0], set()))

        # we subtract the task periods from the cycle
        cycle_noFixedPeriods = \
            cycle_allPeriods.\
            kvapply(lambda k, v: v - mission_periods[k] - cycle_FixeDefPeriods[k])

        # we get the total hours available in each cycle:
        rut_init = \
            insta.get_initial_state('used', maint=maint).\
            vapply(lambda v: sd.SuperDict({first: v}))
        max_rut = insta.get_maintenances('max_used_time')[maint]
        cycle_rut = cycle_missionHours.kapply(lambda k: rut_init.get_m(k[0], k[1], default=max_rut))

        cycle_fixedDefHours = \
            cycle_FixeDefPeriods.\
            kvapply(lambda k, v: sum(new_default_hours[k[0]][p] for p in v)).vapply(round, 2)

        # we get the amount of hours the aircraft should have to do
        # to comply with the cycle remaining hours
        cycle_maxDefaultHours = \
            cycle_rut.\
                kvapply(lambda k, v: (v - cycle_missionHours[k] - cycle_fixedDefHours[k]))

        get_default = insta.get_default_consumption
        cycle_defaults = \
            cycle_noFixedPeriods.\
            kvapply(lambda k, v: sum(get_default(k[0], p) for p in v))
        # if we can do more than the default, we leave the default
        cycle_wrong = cycle_maxDefaultHours.kfilter(lambda k: cycle_maxDefaultHours[k] < cycle_defaults[k])
        for _tup, max_default_hours in cycle_wrong.items():
            # we distribute the hours among all periods
            cy_periods = cycle_noFixedPeriods[_tup]
            mean_hours = round(max_default_hours / len(cy_periods), 2)
            resource = _tup[0]
            for period in cy_periods:
                new_default_hours.set_m(resource, period, value=mean_hours)
        return new_default_hours


if __name__ == '__main__':
    import package.params as params
    import data.template_data as tp
    import package.instance as inst
    import data.data_input as di
    import reports.gantt as gantt

    path = params.PATHS['data'] + 'template/Lot5 (3000s)/'
    model_data = tp.import_input_template(path+ '/template_in.xlsx')
    sol_data = tp.import_output_template(path + 'template_out.xlsx')
    self = ModelMissions(instance = inst.Instance(model_data),
                         solution=sol.Solution(sol_data))
    params.OPTIONS['solver'] = 'CBC'
    solution = self.solve(params.OPTIONS)
    output_path = params.OPTIONS['path']
    self.set_remaining_usage_time_all(time='rut')
    self.set_remaining_usage_time_all(time='ret')
    tp.export_output_template(path + 'template_out_missions.xlsx', self)
    # tp.export_output_template(path + 'template_out.xlsx', model_data, self.solution.data)
    # di.export_data(output_path, self.instance.data, name="data_in", file_type='json', exclude_aux=True)
    di.export_data(output_path, self.instance.data, name="data_in", file_type='json', exclude_aux=True)
    di.export_data(output_path, self.solution.data, name="data_out", file_type='json', exclude_aux=True)
    gantt.make_gantt_from_experiment(experiment=self, path=path)
    # solution.get_tasks()
    # self = ModelMissions.from_dir(path)

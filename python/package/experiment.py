# /usr/bin/python3
import numpy as np
import pandas as pd
import data.data_input as di
import data.template_data as dt
import package.solution as sol
import package.instance as inst
import pytups.tuplist as tl
import pytups.superdict as sd
import os
import orloge as ol
import ujson


class Experiment(object):
    """
    These objects represent the unification of both input data and solution.
    Each is represented by their proper objects.
    Methods are especially checks on faisability.
    In the future I may make this object be a sublass of Solution.
    """

    def __init__(self, instance, solution):
        self.instance = instance
        self.solution = solution

    @classmethod
    def from_dir(cls, path, format='json', prefix="data_"):
        files = [os.path.join(path, prefix + f + "." + format) for f in ['in', 'out']]
        if not np.all([os.path.exists(f) for f in files]):
            return None
        instance = di.load_data(files[0])
        solution = di.load_data(files[1])
        return cls(inst.Instance(instance), sol.Solution(solution))

    @classmethod
    def from_template_dir(cls, path, format='xlsx', prefix="template_"):
        files = file_in, file_out = [os.path.join(path, prefix + f + "." + format) for f in ['in', 'out']]
        if not np.all([os.path.exists(f) for f in files]):
            return None
        instance = dt.import_input_template(file_in)
        solution = dt.import_output_template(file_out)
        return cls(inst.Instance(instance), sol.Solution(solution))

    def check_solution_count(self, **params):
        return self.check_solution(**params).to_lendict()

    def check_solution(self, **params):
        func_list = {
            'candidates':  self.check_resource_in_candidates
            ,'state':       self.check_resource_state
            ,'resources':   self.check_task_num_resources
            ,'usage':       self.check_usage_consumption
            ,'elapsed':     self.check_elapsed_consumption
            ,'min_assign':  self.check_min_max_assignment
            ,'available':   self.check_min_available
            ,'hours':       self.check_min_flight_hours
            ,'start_periods': self.check_fixed_assignments
            ,'dist_maints': self.check_min_distance_maints
            ,'capacity': self.check_sub_maintenance_capacity
            ,'maint_size': self.check_maints_size
        }
        result = {k: v(**params) for k, v in func_list.items()}
        return sd.SuperDict({k: v for k, v in result.items() if v})

    # @profile
    def check_sub_maintenance_capacity(self, ref_compare=0, periods_to_check=None, **param):
        # we get the capacity per month
        inst = self.instance
        rem = inst.get_capacity_calendar(periods_to_check)
        first, last = inst.get_param('start'), inst.get_param('end')
        maintenances = inst.get_maintenances()
        types = maintenances.get_property('type')
        # all_types = set(types.values())
        usage = maintenances.get_property('capacity_usage')
        all_states_tuple = self.get_states()
        if periods_to_check is not None:
            periods_to_check = set(periods_to_check)
            all_states_tuple = all_states_tuple.filter_list_f(lambda x: x[1] in periods_to_check)
        else:
            all_states_tuple = all_states_tuple.filter_list_f(lambda x: last >= x[1] >= first)

        if not len(all_states_tuple):
            return []

        # TODO: finish this
        # all_states_tuple_np = np.asarray(all_states_tuple)
        # a = all_states_tuple_np[:, 2]
        # periods_np = all_states_tuple_np[:, 1]
        # types_np = np.zeros_like(a)
        # usage_np = np.zeros(len(a))
        # for k, v in types.items():
        #     types_np[a == k] = v
        # for k, v in usage.items():
        #     usage_np[a == k] = v
        # for _type in all_types:
        #     _values, _groups = self.sum_by_group(usage_np[types_np==_type],
        #                                          periods_np[types_np==_type])
        #

        for res, period, maint in all_states_tuple:
            _type = types[maint]
            _usage = usage[maint]
            rem[_type, period] -= _usage

        # TODO extra_cap= dp.n(X.consum) - 1)
        return rem.clean(func=lambda x: x < ref_compare)

    @staticmethod
    def sum_by_group(values, groups):
        order = np.argsort(groups)
        groups = groups[order]
        values = values[order]
        values.cumsum(out=values)
        index = np.ones(len(groups), 'bool')
        index[:-1] = groups[1:] != groups[:-1]
        values = values[index]
        groups = groups[index]
        values[1:] = values[1:] - values[:-1]
        return values, groups

    def check_task_num_resources(self, strict=False, **params):
        task_reqs = self.instance.get_tasks('num_resource')

        task_assigned = \
            sd.SuperDict.from_dict(self.solution.get_task_num_resources()).\
                fill_with_default(self.instance.get_task_period_list())
        task_under_assigned = {
            (task, period): task_reqs[task] - task_assigned[task, period]
            for (task, period) in task_assigned
        }
        if strict:
            return sd.SuperDict(task_under_assigned).clean(func=lambda x: x != 0)
        return sd.SuperDict(task_under_assigned).clean(func=lambda x: x > 0)

    def check_resource_in_candidates(self, **params):
        task_solution = self.solution.get_tasks()

        task_candidates = self.instance.get_task_candidates()

        bad_assignment = {
            (resource, period): task
            for (resource, period), task in task_solution.items()
            if resource not in task_candidates[task]
        }
        return bad_assignment

    def get_consumption(self):
        hours = self.instance.get_tasks("consumption")
        return {k: hours[v] for k, v in self.solution.get_tasks().items()}

    def set_remainingtime(self, resource, period, time, value, maint='M'):
        """
        This procedure *updates* the remaining time in the aux property of the solution.
        :param str resource:
        :param str period:
        :param str time: ret or rut
        :param float value: remaining time
        :return: True
        """
        tup = [time, maint, resource, period]
        self.solution.data['aux'].set_m(*tup, value=value)
        return True

    def get_remainingtime(self, resource=None, period=None, time='rut', maint='M'):
        try:
            data = self.solution.data['aux'][time][maint]
            if resource is None:
                return data
            data2 = data[resource]
            if period is None:
                return data2
            return data2[period]
        except KeyError:
            return None

    def update_resource_all(self, resource):
        periods_to_update = self.instance.get_periods()
        for t in ['rut', 'ret']:
            self.update_time_usage(resource, periods_to_update, time=t)

    def update_time_usage_all(self, resource, periods, time):
        maints = self.instance.get_maintenances()
        for m in maints:
            self.update_time_usage(resource, periods, time=time, maint=m)
        return True

    def update_time_usage(self, resource, periods, previous_value=None, time='rut', maint='M'):
        """
        :param resource: a resource to update
        :param periods: a list of consecutive periods to update. ordered.
        :param previous_value: optional value for the remaining time before the first period
        :param time: rut or ret depending if it's usage time or elapsed time
        :return: True
        This procedure *updates* the time of each period using set_remainingtime.
        It assumes all periods do not have a maintenance.
        So the periods should be filled with a task or nothing.
        """
        if self.instance.get_max_remaining_time(time, maint) is None or not periods:
            # we do not update maints that do not check this
            # if periods is empty: we have nothing to update
            return True
        if previous_value is None:
            _period = self.instance.get_prev_period(periods[0])
            previous_value = self.get_remainingtime(resource, _period, time, maint=maint)
        for period in periods:
            value = previous_value - self.get_consumption_individual(resource, period, time)
            self.set_remainingtime(resource=resource, period=period,
                                   time=time, value=value, maint=maint)
            previous_value = value
        return True

    def get_consumption_individual(self, resource, period, time='rut'):
        if time == 'ret':
            return 1
        task = self.solution.data['task'].get_m(resource, period)
        if task is None:
            # get default consumption:
            consumption = self.instance.get_default_consumption(resource, period)
        else:
            # get task consumption:
            consumption = self.instance.data['tasks'].get_m(task, 'consumption', default=0)
        return consumption

    def get_non_maintenance_periods(self, resource=None, state_list=None):
        """
        :return: a tuplist with the following structure:
        resource: [(resource, start_period1, end_period1), (resource, start_period2, end_period2), ..., (resource, start_periodN, end_periodN)]
        two consecutive periods being separated by a maintenance operation.
        It's built using the information of the maintenance operations.
        :param resource: if not None, we filter to only provide this resource's info
        """
        # a = self.get_status(resource)
        # a[a.period>='2019-02'][:8]
        first, last = self.instance.get_param('start'), self.instance.get_param('end')

        if resource is None:
            resources = self.instance.get_resources()
        else:
            resources = [resource]
        # we initialize nomaint periods for resources that do not have a single maintenance:
        nonmaintenances = [(r, first, last) for r in resources if r not in maintenances]
        # now, we iterate over all maintenances to add the before and the after
        for res in maintenances:
            maints = sorted(maintenances[res], key=lambda x: x[0])
            first_maint_start = maints[0][0]
            last_maint_end = maints[-1][1]
            if first_maint_start > first:
                first_maint_start_prev = self.instance.get_prev_period(first_maint_start)
                nonmaintenances.append((res, first, first_maint_start_prev))
            for maint1, maint2 in zip(maints, maints[1:]):
                start = self.instance.get_next_period(maint1[1])
                end = self.instance.get_prev_period(maint2[0])
                nonmaintenances.append((res, start, end))
            if last_maint_end != last:
                start = self.instance.get_next_period(last_maint_end)
                nonmaintenances.append((res, start, last))
        return tl.TupList(nonmaintenances)

    def set_start_periods(self):
        """
        This function remakes the start tasks assignments and states (maintenances)
        It edits the aux part of the solution data
        :return: the dictionary that it assigns
        """
        tasks_start = self.get_task_periods()
        states_start = self.get_state_periods()
        all_starts = tasks_start + states_start
        starts = {(r, t): v for (r, t, v, _) in all_starts}

        if 'aux' not in self.solution.data:
            self.solution.data['aux'] = sd.SuperDict()
        self.solution.data['aux']['start'] = sd.SuperDict.from_dict(starts)

        return starts

    def set_remaining_usage_time_all(self, time='rut', resource=None):
        """
        Wrapper around set_remaining_usage_time to do all maints
        :param time:
        :param resource:
        :return:
        """
        maints = self.instance.get_maintenances()
        return {m: self.set_remaining_usage_time(time=time, maint=m, resource=resource)
                for m in maints if time in self.instance.get_maint_rt(m)}

    def set_remaining_usage_time(self, time="rut", maint='M', resource=None):
        """
        This function remakes the rut and ret times for all resources.
        It assumes nothing of state or task.
        It edits the aux part of the solution data
        :param time: ret or rut
        :param maint: type of maintenance
        :param resource: optional filter of resources
        :return: the dictionary that's assigned
        """
        inst = self.instance
        prev_month = inst.get_prev_period(self.instance.get_param('start'))
        # initial = self.instance.get_resources(label)
        initial = inst.get_initial_state(self.instance.label_rt(time), maint=maint, resource=resource)

        # we initialize values for the start of the horizon
        max_rem = inst.get_max_remaining_time(time=time, maint=maint)

        depends_on = inst.data['maintenances'][maint]['depends_on']
        for _res in initial:
            self.set_remainingtime(resource=_res, period=prev_month, time=time, value=initial[_res], maint=maint)

        # we update values during maintenances
        maintenances = self.get_maintenance_periods(state_list=depends_on, resource=resource)
        for _res, start, end in maintenances:
            for period in inst.get_periods_range(start, end):
                self.set_remainingtime(resource=_res, period=period, time=time, value=max_rem, maint=maint)

        # we update values in between maintenances
        non_maintenances = self.get_non_maintenance_periods(resource=resource, state_list=depends_on)
        for _res, start, end in non_maintenances:
            # print(resource, start, end)
            periods = inst.get_periods_range(start, end)
            self.update_time_usage(resource=_res, periods=periods, time=time, maint=maint)

        return self.solution.data['aux'][time][maint]

    def check_usage_consumption(self, **params):
        return self.check_resource_consumption(time='rut', **params)

    def check_elapsed_consumption(self, **params):
        return self.check_resource_consumption(time='ret', **params)

    def check_resource_consumption(self, time='rut', recalculate=True, **params):
        """
        This function (calculates and) checks the "remaining time" for all maintenances
        :param time: calculate rut or ret
        :param recalculate: used cached rut and ret
        :param params: optional. compatibility
        :return: {(maint, resource, period): remaining time}
        """
        if recalculate:
            rt_maint = self.set_remaining_usage_time_all(time=time)
        else:
            rt_maint = self.solution.data['aux'][time]



    def check_resource_state(self, **params):
        task_solution = self.solution.get_tasks()
        state_solution = self.solution.get_state_tuplist().filter([0, 1])

        task_solution_k = np.fromiter(task_solution.keys(),
                                      dtype=[('A', '<U6'), ('T', 'U7')])
        state_solution_k = np.asarray(state_solution,
                                      dtype=[('A', '<U6'), ('T', 'U7')])
        duplicated_states = \
            np.intersect1d(task_solution_k, state_solution_k)

        return [tuple(item) for item in duplicated_states]

    def check_min_max_assignment(self, **params):
        """
        :return: periods were the min assignment (including maintenance)
        in format: (resource, start, end): error.
        if error negative: bigger than max. Otherwise: less than min
        is not respected
        """
        tasks = self.solution.get_tasks().to_tuplist()
        maints = self.solution.get_state_tuplist()
        previous = sd.SuperDict.from_dict(self.instance.get_resources("states")).\
            to_dictup().to_tuplist()
        min_assign = self.instance.get_min_assign()
        max_assign = self.instance.get_max_assign()

        num_periods = self.instance.get_param('num_period')
        ct = self.instance.compare_tups
        all_states = maints + tasks + previous
        all_states_periods = \
            tl.TupList(all_states).\
                sorted(key=lambda v: (v[0], v[2], v[1])).\
                to_start_finish(ct, sort=False)

        first_period = self.instance.get_param('start')
        last_period = self.instance.get_param('end')

        incorrect = {}
        for (resource, start, state, finish) in all_states_periods:
            # periods that finish before the horizon
            # or at the end are not checked
            if finish < first_period or finish == last_period:
                continue
            size_period = len(self.instance.get_periods_range(start, finish))
            if size_period < min_assign.get(state, 1):
                incorrect[resource, start, finish, state] = min_assign[state] - size_period
            elif size_period > max_assign.get(state, num_periods):
                incorrect[resource, start, finish, state] = max_assign[state] - size_period
        return sd.SuperDict(incorrect)

    def check_fixed_assignments(self, **params):
        first_period = self.instance.get_param('start')
        last_period = self.instance.get_param('end')
        state_tasks = self.solution.get_state_tasks().to_list()
        fixed_states = self.instance.get_fixed_states()
        fixed_states_h = fixed_states.\
            filter_list_f(lambda x: first_period <= x[2] <= last_period).\
            filter([0, 2, 1])
        # state_tasks_tab = pd.DataFrame(state_tasks,
        #                                columns=['resource', 'period', 'status'])
        # fixed_states_tab = pd.DataFrame(fixed_states_h,
        #                                 columns=['resource', 'status', 'period'])
        diff_tups = set(fixed_states_h) - set(state_tasks)
        # result = pd.merge(fixed_states_tab, state_tasks_tab, how='left', on=['resource', 'period'])
        return sd.SuperDict({k: 1 for k in diff_tups})
        # return sd.SuperDict({tuple(x): 1 for x in
        #                      result[result.state_x != result.state_y].to_records(index=False)})


    def check_min_available(self, **params):
        """
        :return: periods where the min availability is not guaranteed.
        """
        resources = self.instance.get_resources().keys()
        c_candidates = self.instance.get_cluster_candidates()
        cluster_data = self.instance.get_cluster_constraints()
        maint_periods = tl.TupList(self.get_maintenance_periods()).\
            to_dict(result_col=[1, 2]).fill_with_default(keys=resources, default=[])
        max_candidates = cluster_data['num']
        num_maintenances = sd.SuperDict().fill_with_default(max_candidates.keys())
        for cluster, candidates in c_candidates.items():
            for candidate in candidates:
                for maint_period in maint_periods[candidate]:
                    for period in self.instance.get_periods_range(*maint_period):
                        if (cluster, period) in num_maintenances:
                            num_maintenances[cluster, period] += 1
        over_assigned = sd.SuperDict({k: max_candidates[k] - v for k, v in num_maintenances.items()})
        return over_assigned.clean(func=lambda x: x < 0)

    def check_min_flight_hours(self, recalculate=True, **params):
        """
        :return: periods where the min flight hours is not guaranteed.
        """
        c_candidates = self.instance.get_cluster_candidates()
        cluster_data = self.instance.get_cluster_constraints()
        min_hours = cluster_data['hours']
        if recalculate:
            ruts = self.set_remaining_usage_time(time='rut', maint='M')
        else:
            ruts = self.get_remainingtime(time='rut', maint='M')
        cluster_hours = sd.SuperDict().fill_with_default(min_hours.keys())
        for cluster, candidates in c_candidates.items():
            for candidate in candidates:
                for period, hours in ruts[candidate].items():
                    if period >= self.instance.get_param('start'):
                        cluster_hours[cluster, period] += hours

        hours_deficit = sd.SuperDict({k: v - min_hours[k] for k, v in cluster_hours.items()})
        return hours_deficit.clean(func=lambda x: x < 0)

    def check_maints_size(self, **params):
        maints = self.instance.get_maintenances()
        duration = maints.get_property('duration_periods')
        inst = self.instance
        start, end = inst.get_param('start'), inst.get_param('end')
        m_s_tab_r = pd.DataFrame.from_records(self.get_state_periods().to_list(),
                                            columns=['resource', 'start', 'maint', 'end'])
        def dist_periods(series, series2):
            return pd.Series(self.instance.get_dist_periods(p, p2) for p, p2 in zip(series, series2))

        inside = np.any([m_s_tab_r.start > start, m_s_tab_r.end < end], axis=0)
        m_s_tab = m_s_tab_r[inside]
        m_s_tab['dist'] = dist_periods(m_s_tab.start, m_s_tab.end) + 1
        m_s_tab['duration'] = m_s_tab.maint.map(duration)
        m_s_tab['value'] = m_s_tab.dist - m_s_tab.duration
        error = m_s_tab[m_s_tab.value != 0]
        result = error[['resource', 'start', 'value']].to_records(index=False)
        return tl.TupList(result).to_dict(result_col=2, is_list=False)

    def check_min_distance_maints(self, **params):
        maints = self.instance.get_maintenances()
        elapsed_time_size = maints.get_property('elapsed_time_size').clean(func=lambda x: x is not None)
        first, last = self.instance.get_first_last_period()
        _next = self.instance.get_next_period

        def compare(tup, last_tup, pp):
            return tup[0]!=last_tup[0] or tup[1]!=last_tup[1] or\
                tup[3] > last_tup[3]

        rets = \
            elapsed_time_size.\
            kapply(lambda m: self.get_remainingtime(time='ret', maint=m)).to_dictup()

        # periods where there's been too short of a period between maints
        ret_before_maint = \
            rets.\
            to_tuplist().sorted().\
            to_start_finish(compare_tups=compare, sort=False, pp=2).\
            filter_list_f(lambda x: x[4] < last).\
            filter([0, 1, 2, 4]).\
            to_dict(result_col=None).\
            kapply(lambda k: (rets[k[0], k[1], k[3]], k[0])).\
            clean(func=lambda v: v[0] > elapsed_time_size[v[1]]).\
            vapply(lambda v: v[0])

        # maybe filter resources when getting states:
        ret_before_maint.keys_tl().filter(1).unique2()
        states = self.get_states().to_dict(result_col=2, is_list=False)
        # here we filter the errors to the ones that involve the same
        # maintenance done twice.
        return \
            ret_before_maint.\
            kapply(lambda k: (k[1], _next(k[3]))).\
            vapply(lambda v: states.get(v)).\
            to_tuplist().\
            filter_list_f(lambda x: x[0] == x[4]).\
            filter([0, 1, 2, 3]).\
            to_dict(result_col=None).\
            vapply(lambda v: ret_before_maint[v])



    def get_kpis(self):
        raise NotImplementedError("This is no longer supported")

    def export_solution(self, path, sheet_name='solution'):

        hours = self.instance.get_tasks('consumption')
        tasks_edited = [(t[0], t[1], '{} ({}h)'.format(t[2], hours[t[2]]))
                        for t in tasks]

        table.to_excel(path, sheet_name=sheet_name)
        return table

    def get_states(self, resource=None):
        """
        :param str resource: optional filter
        :return: (resource, period, state) list
        :rtype: tl.TupList
        """
        # in the input data of the instance
        # (although for now this is the case)
        maints_codes = self.instance.get_maintenances()
        previous_states = self.instance.get_prev_states(resource).\
            filter_list_f(lambda x: x[2] in maints_codes)
        states = self.solution.get_state_tuplist(resource)
        previous_states.extend(states)
        return tl.TupList(previous_states).unique2()

    def get_state_periods(self, resource=None):
        """
        :param str resource: optional filter
        :return: (resource, start, end, state) list
        :rtype: tl.TupList
        """
        all_states = self.get_states(resource)
        ct = self.instance.compare_tups
        all_states.sort(key=lambda x: (x[0], x[2], x[1]))
        return all_states.to_start_finish(ct)

    def get_maintenance_periods(self, resource=None, state_list=None):
        """
        :param str resource: optional resource to filter
        :param set state_list: maintenances to filter
        :return:
        :rtype: tl.TupList
        """
        if state_list is None:
            state_list = set('M')
        all_states = self.get_states(resource)
        maintenances = tl.TupList([(k[0], k[1]) for k in all_states if k[2] in state_list])
        ct = self.instance.compare_tups
        return maintenances.to_start_finish(ct)

    def get_task_periods(self, resource=None):
        tasks = set(self.instance.get_tasks().keys())
        previous_tasks = self.instance.get_prev_states(resource). \
            filter_list_f(lambda x: x[2] in tasks)
        ct = self.instance.compare_tups
        tasks_assigned = self.solution.get_tasks().to_tuplist()
        previous_tasks.extend(tasks_assigned)
        return previous_tasks.to_start_finish(ct)

    def get_maintenance_starts(self, state_list=None):
        return self.get_maintenance_periods(state_list=state_list).filter([0, 1])
        # return [(r, s) for (r, s, e) in maintenances]

    def get_status(self, candidate):
        """
        This function is great for debugging
        :param candidate: a resource
        :return: dataframe with everything that's going on with the resource
        """
        data = self.solution.data
        if 'aux' not in data:
            data['aux'] = sd.SuperDict()
        for t in ['rut', 'ret']:
            if t not in data['aux']:
                self.set_remaining_usage_time_all(time=t)
        if 'start' not in data['aux']:
            self.set_start_periods()
        data_maints = self.instance.get_maintenances()
        _rut = {m: data['aux']['rut'][m].get(candidate, {}) for m in self.instance.get_rt_maints('rut')}
        rut = pd.DataFrame.from_dict(_rut)
        _ret = {m: data['aux']['ret'][m].get(candidate, {}) for m in self.instance.get_rt_maints('ret')}
        ret = pd.DataFrame.from_dict(_ret)
        # ret = pd.DataFrame.from_dict(data['aux']['ret']['M'].get(candidate, {}), orient='index')
        start = pd.DataFrame.from_dict(data['aux']['start'].get(candidate, {}), orient='index')
        # state = pd.DataFrame.from_dict(data['state'].get(candidate, {}), orient='index')
        state_m = pd.DataFrame.from_dict(data['state_m'].get(candidate, {}), orient='index')
        task = pd.DataFrame.from_dict(data['task'].get(candidate, {}), orient='index')
        args = {'left_index': True, 'right_index': True, 'how': 'left'}
        table = rut.merge(ret, **args).merge(task, **args).\
            merge(start, **args).sort_index().merge(state_m, **args)
        names = np.all(table.isna(), axis=0)
        list_names = names[~names].index
        table = table.filter(list_names)
        return table.reset_index().rename(columns={'index': 'period'})

    def get_solution(self):
        """
        Makes a deep copy of the current solution.

        :return: dictionary with data
        :rtype: sd.SuperDict
        """
        data = self.solution.data
        data_copy = ujson.loads(ujson.dumps(data))
        return sd.SuperDict.from_dict(data_copy)

    def set_solution(self, data):
        """
        Updates current solution with a new solution taken as argument.

        :param sd.SuperDict data: dictionary with a solution's data
        :return: True
        :rtype: bool
        """
        data = ujson.loads(ujson.dumps(data))
        self.solution.data = sd.SuperDict.from_dict(data)
        return True

    def get_inconsistency(self):
        """
        Checks that the status of aircraft are updated correctly when the solution
        is modified.

        :return: inconsistencies
        :rtype: set
        """
        b = self.solution.data['aux'].to_dictup().to_tuplist().to_set()
        self.check_solution()
        a = self.solution.data['aux'].to_dictup().to_tuplist().to_set()
        return a ^ b

    def solve(self, **kwargs):
        raise NotImplementedError('An experiment needs to be subclassed and be given a solve method.')


def exp_get_info(path, get_log_info=True, get_exp_info=True):
    # print(path)
    parameters = log_info = inst_info = {}
    if get_exp_info:
        exp = Experiment.from_dir(path, format="json")
        if exp is None:
            return None
        parameters = exp.instance.get_param()
        inst_info = exp.instance.get_info()
    options_path = os.path.join(path, "options.json")
    options = di.load_data(options_path)
    if not options:
        return None
    log_path = os.path.join(path, "results.log")
    if os.path.exists(log_path) and get_log_info:
        log_info = ol.get_info_solver(log_path, options['solver'])
    return {**parameters, **options, **log_info, **inst_info}


if __name__ == "__main__":
    import package.params as pm

    pass

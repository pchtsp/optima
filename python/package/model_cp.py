import docplex.cp.model as doc
import package.auxiliar as aux
import package.config as conf
import package.solution as sol

######################################################
# TODO: add minimum mission duration assignment
# TODO: contraintes les posibilités des candidates: maximum X candidates par task ou par resource


def solve_model(instance, options=None):
    l = instance.get_domains_sets()
    ub = instance.get_bounds()
    last_period = instance.get_param('end')
    first_period = instance.get_param('start')
    # last_period_lessM = aux.shift_month(last_period, -instance.get_param('maint_duration'))
    consumption = {k: round(v) for k, v in instance.get_tasks('consumption').items()}
    requirement = instance.get_tasks('num_resource')
    ret_init = instance.get_initial_state("elapsed")
    rut_init = instance.get_initial_state("used")
    # ret_obj = sum(ret_init[a] for a in l['resources'])
    # rut_obj = sum(rut_init[a] for a in l['resources'])
    # num_resource_working = instance.get_total_period_needs()
    num_resource_maint = aux.fill_dict_with_default(instance.get_total_fixed_maintenances(), l['periods'])
    num_resource_maint_cluster = aux.dict_to_lendict(instance.get_fixed_maintenances_cluster())
    # instance.get_total_fixed_maintenances()
    # maint_weight = instance.get_param("maint_weight")
    # unavail_weight = instance.get_param("unavail_weight")
    maint_capacity = instance.get_param('maint_capacity')
    max_elapsed = instance.get_param('max_elapsed_time')
    max_usage = instance.get_param('max_used_time')
    min_percent = 0.10
    min_value = 1
    st = {k: v for k, v in enumerate(instance.get_tasks().keys())}
    st[len(st)] = 'M'
    st[len(st)] = 'D'  # doing nothing
    st_i = {v: k for k, v in st.items()}

    pe = {k+1: v for k, v in enumerate(instance.get_periods())}
    period_0_pe = 0
    pe[period_0_pe] = l['period_0']
    pe_i = {v: k for k, v in pe.items()}
    last_period_pe = pe_i[last_period]

    c_needs = instance.get_cluster_needs()
    c_candidates = instance.get_cluster_candidates()
    c_num_candidates = aux.dict_to_lendict(c_candidates)
    c_slack = {tup: c_num_candidates[tup[0]] - c_needs[tup] - num_resource_maint_cluster.get(tup, 0)
               for tup in c_needs}
    c_min = {(k, t): min(
        c_slack[(k, t)],
        max(
            int(c_num_candidates[k] * min_percent),
            min_value)
        ) for (k, t) in c_slack
    }
    fixed_state = {tup: st_i['M'] for tup in l['at_maint']}

    # MODEL
    model = doc.CpoModel()

    # VARIABLES:
    # integer:
    state_at = [(a, pe_i[t]) for a, t in l['at_avail']]
    start_at = [(a, pe_i[t]) for a, t in l['at_start']]
    rt_at0 = [(a, pe_i[t]) for a, t in l['at0']]

    state = {}
    for (a, t) in state_at:
        domain = [st_i[st] for st in l['v_at'][a, pe[t]] + ['M', 'D']]
        state[a, t] = model.integer_var(name="state", domain=domain)
    start = model.binary_var_dict(name="start", keys=start_at)
    # consum = model.integer_var_dict(name="consum", keys=st.keys())

    # numeric:
    ret = model.integer_var_dict(name="ret", min=0, max=ub['ret'], keys=rt_at0)
    rut = model.integer_var_dict(name="rut", min=0, max=ub['rut'], keys=rt_at0)

    # objective function:
    num_maint = model.integer_var(name="num_maint", min=0, max=ub['num_maint'])
    ret_obj_var = model.integer_var(name="ret_obj_var", min=0, max=ub['ret_end'])
    rut_obj_var = model.integer_var(name="rut_obj_var", min=0, max=ub['rut_end'])

    # OBJECTIVE:
    model.minimize(num_maint * max_elapsed * 2 -
                   ret_obj_var -
                   rut_obj_var * max_elapsed / max_usage)

    # TODO: Model with interval variables; we would use interval.start_of
    # TODO: Model with step_at for hours and months
    # CONSTRAINTS:

    # # consum initialize:
    # for st1 in st.keys():
    #     model.add(consum[st1] == round(consumption.get(st[st1], 0)))

    # num resources:
    for (v, t), a_s in l['a_vt'].items():
        model.add(sum(state[a, pe_i[t]] == st_i[v] for a in a_s) == requirement[v])
        # model += pl.lpSum(task[(a, v, t)] for a in l['a_vt'][(v, t)]) == requirement[v]

    for a, t in l['at_start']:
        # start
        # we start a maintenance if we have a maintenance assignment and we didn't before.
        at_prev = a, l["previous"][t]
        at_prev_pe = a, pe_i[l["previous"][t]]
        at_pe = a, pe_i[t]

        if t == first_period or at_prev in fixed_state:
            model.add(start[at_pe] == (state[at_pe] == st_i['M']))
        else:
            model.add(start[at_pe] ==
                      ((state[at_pe] == st_i['M']) & (state[at_prev_pe] != st_i['M']))
                      )

        # if we start a maintenance=> we have the following states as maintenances too
        model.add(
            model.if_then(start[at_pe] == 1,
                          model.all([state[a, pe_i[t2]] == st_i['M'] for t2 in l['t2_at1'][a, t]])
                          )
        )

    # remaining used time calculations:
    # remaining elapsed time calculations:
    for (a, t) in l['at']:
        # at_prev = a, l["previous"][t]
        at_prev_pe = a, pe_i[l["previous"][t]]
        at_pe = a, pe_i[t]
        state_ = state.get(at_pe, st_i['D'])
        ret_ = ret[at_pe]
        rut_ = rut[at_pe]

        # ret
        model.add(
            model.if_then(state_ != st_i['M'], ret_ == (ret[at_prev_pe] - 1))
        )
        model.add(
            model.if_then(state_ == st_i['M'], ret_ == ub['ret'])
        )
        # rut
        model.add(
            model.if_then(state_ != st_i['M'],
                          rut_ == (rut[at_prev_pe] - consumption.get(state_, 0))
                          )
        )
        model.add(
            model.if_then(state_ == st_i['M'], rut_ == ub['rut'])
        )

    # the start period is given by parameters:
    for a in l['resources']:
        model.add(rut[(a, period_0_pe)] == round(rut_init[a]))
        model.add(ret[(a, period_0_pe)] == round(ret_init[a]))

    # minimum availability per cluster and period
    for k, t in c_needs:
        model.add(
            sum(state[a, pe_i[t]] == st_i['M'] for a in c_candidates[k] if (a, t) in l['at_avail']) +
            c_needs[(k, t)] +
            c_min[(k, t)] +
            num_resource_maint_cluster.get((k, t), 0)
            <= c_num_candidates[k]
        )
        # maintenances decided by the model to candidates +
        # assigned resources to tasks in cluster +
        # minimum resources for cluster +
        # <= resources already in maintenance

    # count the number of maintenances:
    model.add(num_maint == sum(state[a, pe_i[t]] == st_i['M'] for (a, t) in l['at_avail']))

    # max number of maintenances:
    for t in l['periods']:
        model.add(
            sum(state[a, pe_i[t]] == st_i['M'] for a in l['resources']
                if (a, t) in l['at_avail']) + num_resource_maint[t] <= maint_capacity
        )
        # model += pl.lpSum(start[(a, _t)] for (a, _t) in l['at1_t2'][t] if (a, _t) in l['at_start']) + \
        #          num_resource_maint[t] <= maint_capacity

    # calculate the rem and ret:
    model.add(sum(ret[a, last_period_pe] for a in l['resources']) == ret_obj_var)
    model.add(sum(rut[a, last_period_pe] for a in l['resources']) == rut_obj_var)

    # SOLVING
    result = model.solve(TimeLimit=options.get('timeLimit', 300))

    if not result:
        print("Model resulted in non-feasible status")
        return None

    _start = {(a, pe[t]): 1 for (a, t), v in start.items() if result[v] > 0.5}
    _task = {(a, pe[t]): st[result[v]] for (a, t), v in state.items()
             if st[result[v]] in instance.get_tasks()}
    _rut = {(a, pe[t]): result[v] for (a, t), v in rut.items()}
    _ret = {(a, pe[t]): result[v] for (a, t), v in ret.items()}

    _state = {tup: 'M' for tup in l['planned_maint']}
    _state.update({(a, t2): 'M' for (a, t) in _start for t2 in l['t2_at1'][a, t]})

    solution_data_pre = {
        'state': _state,
        'task': _task,
        'aux': {
            'start': _start,
            'rut': _rut,
            'ret': _ret
        }
    }

    solution_data = {k: aux.dicttup_to_dictdict(v)
                     for k, v in solution_data_pre.items() if k != "aux"}
    solution_data['aux'] = {k: aux.dicttup_to_dictdict(v)
                            for k, v in solution_data_pre['aux'].items()}
    solution = sol.Solution(solution_data)

    return solution


if __name__ == "__main__":
    pass
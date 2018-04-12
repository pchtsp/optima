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
    consumption = {k: round(v) for k, v in instance.get_tasks('consumption').items()}
    requirement = instance.get_tasks('num_resource')
    ret_init = instance.get_initial_state("elapsed")
    rut_init = instance.get_initial_state("used")
    num_resource_maint = aux.fill_dict_with_default(instance.get_total_fixed_maintenances(), l['periods'])
    num_resource_maint_cluster = aux.dict_to_lendict(instance.get_fixed_maintenances_cluster())
    maint_capacity = instance.get_param('maint_capacity')
    max_elapsed = instance.get_param('max_elapsed_time')
    max_usage = instance.get_param('max_used_time')
    min_percent = 0.10
    min_value = 1
    st = {k: v for k, v in enumerate(l['tasks'])}
    st_tasks = list(st.keys())
    st[len(st)] = 'M'
    st[len(st)] = 'D'  # doing nothing
    st_i = {v: k for k, v in st.items()}
    states_eq_sorted = sorted(st.items(), key=lambda x: x[0])
    consumption_si = [consumption.get(v, 0) for k, v in states_eq_sorted]
    requirement_si = [requirement.get(v, 0) for k, v in states_eq_sorted]

        # {st_i[k]: v for k, v in requirement.items()}

    pe = {k+1: v for k, v in enumerate(l['periods'])}
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
    # cash = step_at(0, 0)
    # for p in Houses:
    #     cash += mdl4.step_at(60 * p, 30000)
    # for h in Houses:
    #     for i, t in enumerate(TaskNames):
    #         cash -= mdl4.step_at_start(itvs[h, t], 200 * Duration[i])
    # CONSTRAINTS:

    # num resources:

    for t in l['periods']:
        active_tasks = [st_i[v] for v in l['tasks'] if (v, t) in l['vt']]
        model.distribute(counts=requirement_si,
                         exprs=list(state.values()),
                         values=active_tasks)

    # for (v, t), a_s in l['a_vt'].items():
    #     model.add(sum(state[a, pe_i[t]] == st_i[v] for a in a_s) == requirement[v])

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
                          rut_ == (rut[at_prev_pe] - model.element(consumption_si, state_))
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
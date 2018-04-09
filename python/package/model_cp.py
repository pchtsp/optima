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
    consumption = instance.get_tasks('consumption')
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
    si = {k: v for k, v in enumerate(instance.get_tasks().keys())}
    si[len(si)] = 'M'
    si[len(si)] = 'D'  # doing nothing
    si_inv = {v: k for k, v in si.items()}

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

    # MODEL
    model = doc.CpoModel()

    # VARIABLES:
    # integer:
    state = model.integer_var_dict(keys=l['at_free'], name="state", domain=list(si.keys()))
    # task = pl.LpVariable.dicts(name="task", indexs=l['avt'], lowBound=0, upBound=1, cat=pl.LpInteger)
    # start = pl.LpVariable.dicts(name="start", indexs=l['at_start'], lowBound=0, upBound=1, cat=pl.LpInteger)

    # numeric:
    ret = model.integer_var_dict(name="ret", min=0, max=round(ub['ret']), keys=l['at0'])
    rut = model.integer_var_dict(name="rut", min=0, max=round(ub['rut']), keys=l['at0'])
    # ret = pl.LpVariable.dicts(name="ret", indexs=l['at0'], lowBound=0, upBound=ub['ret'], cat=var_type)
    # rut = pl.LpVariable.dicts(name="rut", indexs=l['at0'], lowBound=0, upBound=ub['rut'], cat=var_type)

    # objective function:
    num_maint = model.integer_var(name="num_maint", min=0, max=ub['num_maint'])
    ret_obj_var = model.integer_var(name="ret_obj_var", min=0, max=round(ub['ret_end']))
    rut_obj_var = model.integer_var(name="rut_obj_var", min=0, max=round(ub['rut_end']))
    # ret_obj_var = pl.LpVariable(name="ret_obj_var", lowBound=0, upBound=ub['ret_end'], cat=var_type)
    # rut_obj_var = pl.LpVariable(name="rut_obj_var", lowBound=0, upBound=ub['rut_end'], cat=var_type)

    # OBJECTIVE:
    model.minimize(num_maint * max_elapsed * 2 -
                   ret_obj_var -
                   rut_obj_var * max_elapsed / max_usage)

    # TODO: planned maintenances.
    # TODO: maintenance lasts 6 periods
    # TODO: model start of maintenance in a separate variable
    # CONSTRAINTS:
    # num resources:
    for (v, t), a_s in l['a_vt'].items():
        model.add(sum(state[a, t] == si_inv[v] for a in a_s) == requirement[v])
        # model += pl.lpSum(task[(a, v, t)] for a in l['a_vt'][(v, t)]) == requirement[v]

    # remaining used time calculations:
    # remaining elapsed time calculations:
    for (a, t) in l['at_free']:
        if t == first_period:
            continue
        # ret
        model.add(
            model.if_then((state[a, t] != si_inv['M']) | (state[a, l["previous"][t]] == si_inv['M']),
                          ret[a, t] == ret[a, l["previous"][t]] - 1)
        )
        model.add(
            model.if_then((state[a, t] == si_inv['M']) & (state[a, l["previous"][t]] != si_inv['M']),
                          ret[a, t] == ub['ret'])
        )
        # rut
        model.add(
            model.if_then(state[a, t] != si_inv['M'],
                          rut[a, t] == rut[(a, l["previous"][t])] - consumption.get(state[a, t], 0))
        )
        model.add(
            model.if_then(state[a, t] == si_inv['M'], ret[a, t] == ub['rut'])
        )


    # the start period is given by parameters:
    for a in l['resources']:
        model.add(rut[(a, l['period_0'])] == round(rut_init[a]))
        model.add(ret[(a, l['period_0'])] == round(ret_init[a]))

    # minimum availability per cluster and period
    for k, t in c_needs:
        model.add(
            sum(state[(a, t)] == si_inv['M'] for a in c_candidates[k] if (a, t) in l['at_free']) +
            c_needs[(k, t)] +
            c_min[(k, t)] +
            num_resource_maint_cluster.get((k, t), 0) \
            <= c_num_candidates[k]
        )
        # maintenances decided by the model to candidates +
        # assigned resources to tasks in cluster +
        # minimum resources for cluster +
        # <= resources already in maintenance

    # count the number of maintenances:
    model.add(num_maint == sum(state[(a, t)] == si_inv['M'] for (a, t) in l['at_free']))

    # max number of maintenances:
    for t in l['periods']:
        model.add(sum(state[(a, t)] == si_inv['M'] for a in l['resources']
                      if (a, t) in l['at_free']) <= maint_capacity)
        # model += pl.lpSum(start[(a, _t)] for (a, _t) in l['at1_t2'][t] if (a, _t) in l['at_start']) + \
        #          num_resource_maint[t] <= maint_capacity

    # calculate the rem and ret:
    model.add(sum(ret[(a, last_period)] for a in l['resources']) == ret_obj_var)
    model.add(sum(rut[(a, last_period)] for a in l['resources']) == rut_obj_var)

    # SOLVING
    result = model.solve(TimeLimit=options.get('timeLimit', 300))

    if not result:
        print("Model resulted in non-feasible status")
        return None

    _start = {(a, t): 1 for (a, t), v in state.items()
              if result[v] == si_inv['M'] and
              (t == first_period or
              result[state[a, l['previous'][t]]] != si_inv['M'])
              }
    _task = {k: si[result[v]] for k, v in state.items() if si[result[v]] in instance.get_tasks()}
    _rut = {t: result[rut[t]] for t in rut}
    _ret = {t: result[ret[t]] for t in ret}

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
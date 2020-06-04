import pulp as pl
import pytups.superdict as sd

def big_mip_fix_variables(change, variable, t1, t2, index, name):
    # variable= self.big_mip.start_T or self.big_mip.start_M
    # t1 = 2 or 1
    # t2 = 3 or 2
    # index = [0, 1] or [0]

    # four cases:
    # 1. resources not in change: all variables fixed.
    # 2. both dates (t1 and t2) are inside the window: variable is free (do nothing)
    # 3. the start date is inside the window:
    #   get assignments that end in the fixed date and select one
    # 4. the end date is inside the window:
    #   get assignments that start in the fixed date and end inside the window and select one

    resources = set(change['resources'])
    var_with_value = variable.vfilter(lambda v: v.value())
    all_variables = variable.keys_tl()
    var_value_filt = var_with_value.keys_tl().vfilter(lambda k: k[0] in resources)
    all_var_id = all_variables.to_dict(indices=index, result_col=[t1, t2])
    start, end = change['start'], change['end']

    force_outside = \
        variable. \
            kfilter(lambda k: (k[t2] < start or k[t1] > end)
                              or (k[t1] < start and k[t2] > end)
                    ).values_tl() + var_with_value.kfilter(lambda k: k[0] not in resources).values_tl()

    active_start_outside = \
        var_value_filt. \
            vfilter(lambda k: k[t1] < start and (start <= k[t2] <= end)). \
            to_dict(result_col=t1, indices=index).vapply(set)

    force_start_outside = \
        {k: all_var_id[k].\
            vfilter(lambda tt: tt[0] in periods and start <= tt[1] <= end)
         for k, periods in active_start_outside.items()}

    active_end_outside = \
        var_value_filt. \
            vfilter(lambda k: k[t2] > end and (start <= k[t1] <= end)). \
            to_dict(result_col=t2, indices=index).vapply(set)

    force_end_outside = \
        {k: all_var_id[k].\
            vfilter(lambda tt: tt[1] in periods and start <= tt[0] <= end)
         for k, periods in active_end_outside.items()}

    # this ones we do not fix:
    # tasks_inside = variable_with_value.vfilter(lambda k: (start <= k[t1]) and (k[t2] <= end))

    # # fixing!
    # for k in force_outside:
    #     variable[k].fixValue()

    # new constraints:
    def _to_constraint(k, tt, i):
        if not isinstance(k, tuple):
            k = k,
        indices = [tuple(list(k) + list(t)) for t in tt]
        return pl.lpSum(variable[i] for i in indices) >= 1, '{}_delete_{}'.format(name, i)

    all_new_constraints = list(force_end_outside.items()) + list(force_start_outside.items())
    constraints = [_to_constraint(k, tt, pos) for pos, (k, tt) in enumerate(all_new_constraints)]

    return force_outside, constraints


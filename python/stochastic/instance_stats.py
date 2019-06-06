import pytups.superdict as sd
import pandas as pd
import stochastic.params as params

def get_resources_of_type(instance, _type=0):
    return sd.SuperDict(instance.get_resources('type')).\
        clean(func=lambda v: v == _type).to_tuplist().filter(0).to_set()


def is_type(task, _type, property_name='type_resource'):
    return task[property_name] == _type


def min_assign_consumption(instance, _type=0):
    tasks = instance.get_tasks()
    tasks_tt = \
        sd.SuperDict(tasks).\
        clean(func=is_type, _type=_type).\
        apply(lambda k, v: v['consumption']*v['num_resource']*v['min_assign'])
    return pd.Series(tasks_tt.values_l())


def get_rel_consumptions(instance, _type=0):
    ranged = instance.get_periods_range
    tasks = instance.get_tasks()
    tasks_tt = \
        sd.SuperDict(tasks). \
            clean(func=is_type, _type=_type). \
            apply(lambda k, v:
                  sd.SuperDict({p: v['consumption']*v['min_assign']
                                for p in ranged(v['start'], v['end'])})). \
            to_dictup(). \
            to_tuplist(). \
            to_dict(result_col=2, indices=[1]). \
            apply(lambda _, x: sum(x)).to_tuplist()
    tasks_tt.sort()
    dates, values = zip(*tasks_tt)
    return pd.Series(values)


def get_consumptions(instance, hours=True, _type=0):

    ranged = instance.get_periods_range
    tasks = instance.get_tasks()
    tasks = sd.SuperDict.from_dict(tasks)
    if not hours:
        tasks = tasks.apply(lambda k, v: {**v, **{'consumption': 1}})
    tasks_tt = \
        tasks. \
        clean(func=is_type, _type=_type). \
        vapply(lambda v:
                       sd.SuperDict({p: v['consumption']*v['num_resource']
                                     for p in ranged(v['start'], v['end'])})).\
        to_dictup().\
        to_tuplist().\
        to_dict(result_col=2, indices=[1]).\
        apply(lambda _, x: sum(x)).to_tuplist()
    tasks_tt.sort()
    dates, values = zip(*tasks_tt)
    return pd.Series(values)


def get_init_hours(instance, _type=0):
    resources = sd.SuperDict.from_dict(instance.get_resources())
    data =\
        resources.\
            clean(func=is_type, _type=_type, property_name='type').\
            get_property('initial_used').values_l()
    return pd.Series(data)


def get_argmedian(consumption, prop=0.5):
    limit = sum(consumption) * prop
    so_far = 0
    for pos, item in enumerate(consumption):
        so_far += item
        if so_far > limit:
            return pos
    return len(consumption)


def get_num_special(instance, _type=0):
    _dist = instance.get_dist_periods
    tasks = sd.SuperDict.from_dict(instance.get_tasks())
    tasks_spec = \
        tasks.\
            clean(func=is_type, _type=_type).\
            get_property('capacities').\
            to_lendict().clean(1).keys_l()
    spec_hours = \
        tasks.filter(tasks_spec).\
            vapply(lambda v: v['consumption']*v['num_resource']*
                             (_dist(v['start'], v['end'])+1)
                   ).\
            values()
    return sum(spec_hours)


def get_geomean(consumption):
    total = sum(consumption)
    result = sum(pos * item for pos, item in enumerate(consumption)) / total
    return result


def calculate_stat(instance, coefs, _type):
    intercept = coefs.get('intercept', 0)
    consumption = get_consumptions(instance, hours=True, _type=_type)
    mean_consum = consumption.mean()
    init = get_init_hours(instance, _type=_type).mean()
    cons_min = min_assign_consumption(instance, _type=_type)
    rel_consumption = get_rel_consumptions(instance, _type=_type)
    quantsw = rel_consumption.rolling(12).mean().shift(-11).quantile(q=[0.5, 0.75, 0.9]).tolist()

    data = \
        sd.SuperDict(
            mean_consum=mean_consum
            , mean_consum2=mean_consum**2
            , mean_consum3=mean_consum**3
            , init = init
            , spec_tasks = get_num_special(instance, _type)
            , var_consum= consumption.agg('var')
            , max_consum = consumption.agg('max')
            , cons_min_max=cons_min.agg('max')
            , quant9w = quantsw[2]

        )
    missing_info = set(coefs) - set(data)
    if len(missing_info) > 1:
        # intercept is the only one that should be there
        raise KeyError('missing keys in data: {}'.format(missing_info))
    return sum(data.apply(lambda k, v: v * coefs.get(k, 0)).values()) + intercept


def get_bound_var(instance, variable, _type):
    data = params.get_bound_var_data
    if variable not in data:
        raise ValueError('data not found for variable: {}'.format(variable))
    return calculate_stat(instance, data[variable], _type=_type)


def get_min_dist_2M(instance, _type):
    min, max = params.get_min_dist_2M_min_max
    mean_consum = get_consumptions(instance, hours=True, _type=_type).mean()
    if mean_consum < min:
        return instance.get_param('max_elapsed_time')
    if mean_consum > max:
        return params.get_min_dist_2M_min_elapsed
    data = params.get_bound_var_data['min_cycle_2M_min']
    return round(calculate_stat(instance, coefs=data, _type=_type))

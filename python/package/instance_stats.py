import pytups.superdict as sd
import pandas as pd


def min_assign_consumption(instance):
    tasks = instance.get_tasks()
    tasks_tt = \
        sd.SuperDict(tasks).\
        apply(lambda k, v: v['consumption']*v['num_resource']*v['min_assign'])
    return pd.Series(tasks_tt.values_l())


def get_rel_consumptions(instance):
    ranged = instance.get_periods_range
    tasks = instance.get_tasks()
    tasks_tt = \
        sd.SuperDict(tasks). \
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


def get_consumptions(instance, hours=True):

    ranged = instance.get_periods_range
    tasks = instance.get_tasks()
    tasks = sd.SuperDict.from_dict(tasks)
    if not hours:
        tasks = tasks.apply(lambda k, v: {**v, **{'consumption': 1}})
    tasks_tt = \
        tasks.\
        apply(lambda k, v:
                  sd.SuperDict({p: v['consumption']*v['num_resource']
                                for p in ranged(v['start'], v['end'])})).\
        to_dictup().\
        to_tuplist().\
        to_dict(result_col=2, indices=[1]).\
        apply(lambda _, x: sum(x)).to_tuplist()
    tasks_tt.sort()
    dates, values = zip(*tasks_tt)
    return pd.Series(values)


def get_init_hours(instance):
    return pd.Series([*instance.get_resources('initial_used').values()])


def get_argmedian(consumption, prop=0.5):
    half = sum(consumption) * prop
    so_far = 0
    for pos, item in enumerate(consumption):
        so_far += item
        if so_far > half:
            return pos
    return len(consumption)


def get_geomean(consumption):
    total = sum(consumption)
    result = sum(pos * item for pos, item in enumerate(consumption)) / total
    return result

def calculate_stat(instance, coefs):
    intercept = coefs.get('intercept', 0)
    mean_consum = get_consumptions(instance, hours=True).mean()
    init = get_init_hours(instance).mean()
    data = \
        sd.SuperDict(
            mean_consum=mean_consum
            , mean_consum2=mean_consum**2
            , mean_consum3=mean_consum**3
            , init = init
        )
    return sum(data.apply(lambda k, v: v * coefs.get(k, 0)).values_l()) + intercept


def get_bound_var(instance, variable):
    data = \
        {'maints': {'mean_consum': 0.0,
                    'init': -0.011996316,
                    'mean_consum2': -0.0001735429,
                    'mean_consum3': 8.8102027e-07,
                    'intercept': 30.652218142375244},
         'mean_2maint': {'mean_consum': 0.0,
                         'init': -0.016398298,
                         'mean_consum2': -0.00050938537,
                         'mean_consum3': 2.357292e-06,
                         'intercept': 19.424510800979323},
         'mean_dist': {'mean_consum': -0.11756852,
                       'init': -0.010143913,
                       'mean_consum2': 0.0010128354,
                       'mean_consum3': -2.3906576e-06,
                       'intercept': 60.588303439359386}}
    if variable not in data:
        raise ValueError('data not found for variable: {}'.format(variable))
    return calculate_stat(instance, data[variable])

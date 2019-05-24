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
    data = \
        sd.SuperDict(
            consumption=get_consumptions(instance, hours=True).mean()
            ,consumption2=get_consumptions(instance, hours=True).mean()**2
            ,init = get_init_hours(instance).mean()
        )
    return sum(data.apply(lambda k, v: v * coefs.get(k, 0)).values_l()) + intercept


def get_bound_var(instance, variable):
    data = {
        'maints': {'mean_consum': -0.083981156,
                   'init': -0.014008507,
                   'mean_consum2': 0.00033154176,
                   'intercept': 36.63064544413732},
        'mean_2maint': {'mean_consum': -0.33839343,
                        'init': -0.02039799,
                        'mean_consum2': 0.0011614594,
                        'intercept': 44.222750378908614},
        'mean_dist': {'mean_consum': 0.10140094,
                      'init': -0.012073993,
                      'mean_consum2': -0.00026209599,
                      'intercept': 50.271137909384024}
    }
    if variable not in data:
        raise ValueError('data not found for variable: {}'.format(variable))
    return calculate_stat(instance, data[variable])

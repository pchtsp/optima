import pytups.superdict as sd
import pandas as pd


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


def calculate_stat(instance, coefs, type=0):
    # TODO: think on this and below
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
        {'max_maints': {'mean_consum': 0.0,
                        'init': -0.014491014,
                        'mean_consum2': -0.00021415783,
                        'mean_consum3': 1.0077952e-06,
                        'intercept': 32.11475261509869},
         'min_maints': {'mean_consum': 0.44668751,
                        'init': -0.018518972,
                        'mean_consum2': -0.0021752748,
                        'mean_consum3': 3.6075012e-06,
                        'intercept': 0.16115577492676791},
         'max_mean_2maint': {'mean_consum': 0.079769466,
                             'init': -0.017577526,
                             'mean_consum2': -0.00093425999,
                             'mean_consum3': 3.0289197e-06,
                             'intercept': 14.851572921387236},
         'min_mean_2maint': {'mean_consum': 0.24287724,
                             'init': -0.018909612,
                             'mean_consum2': -0.0014578421,
                             'mean_consum3': 3.0106766e-06,
                             'intercept': 0.18510246178846076},
         'max_mean_dist': {'mean_consum': -0.02652565,
                           'init': -0.010174868,
                           'mean_consum2': 0.00047905823,
                           'mean_consum3': -1.4342716e-06,
                           'intercept': 56.15615658837415},
         'min_mean_dist': {'mean_consum': 0.72213221,
                           'init': -0.0071054628,
                           'mean_consum2': -0.0032535334,
                           'mean_consum3': 4.3151898e-06,
                           'intercept': 0.7021066395663593}}
    if variable not in data:
        raise ValueError('data not found for variable: {}'.format(variable))
    return calculate_stat(instance, data[variable])

def get_min_dist_2M(instance):
    min, max = [150, 300]
    data = \
        {'mean_consum': 4.527784863082707,
         'mean_consum2': -0.021417007220022065,
         'mean_consum3': 3.135859788535103e-05,
         'init': 0.01026743076834458,
         "intercept": -251.89244412049118}
    mean_consum = get_consumptions(instance, hours=True).mean()
    if mean_consum < min:
        return instance.get_param('max_elapsed_time')
    if mean_consum > max:
        return instance.get_param('max_elapsed_time') - instance.get_param('elapsed_time_size')
    return round(calculate_stat(instance, coefs=data))

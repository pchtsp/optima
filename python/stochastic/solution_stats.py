import pandas as pd
import stochastic.instance_stats as istats


def get_num_maints(case, _type=0):
    """
    :param case:
    :param _type:
    :return: total number of maintenances in the planning period
    """
    res = istats.get_resources_of_type(case.instance, _type=_type)
    maints = \
        case.get_maintenance_starts().\
            filter_list_f(lambda v: v[0] in res)
    return len(maints)


def get_prev_1M_dist(case, _type=0):
    """
    :param case:
    :param _type:
    :return: the distance between the first maintenance and the one before the
    beginning of the planning period
    """
    res = istats.get_resources_of_type(case.instance, _type=_type)
    m_starts = case.get_maintenance_starts().filter_list_f(lambda v: v[0] in res)
    dist = case.instance.get_dist_periods
    first, last = (case.instance.get_param(p) for p in ['start', 'end'])
    init_ret = case.instance.get_resources('initial_elapsed')
    max_ret = case.instance.get_param('max_elapsed_time')

    dist_to_1M = \
        m_starts.\
        to_dict(1). \
        vapply(lambda v: [vv for vv in v if vv >= first]). \
        vapply(sorted).\
        vapply(lambda v: dist(first, v[0])).\
        apply(lambda k, v: max_ret - init_ret[k] + v)

    return dist_to_1M


def get_1M_2M_dist(case, _type=0):
    """
    :param case:
    :param _type:
    :return: the distance between the first and second maintenance for each aircraft
    """
    res = istats.get_resources_of_type(case.instance, _type=_type)
    cycles = case.get_all_maintenance_cycles().filter(list(res))
    dist = case.instance.get_dist_periods
    max_value = case.instance.get_param('max_elapsed_time')

    # now we only want to see the distance when
    # there is a second maintenance
    cycles_between = \
        cycles.clean(func=lambda v: len(v)==3).\
        apply(lambda k, v: dist(*v[1]) + 1)

    if not len(cycles_between):
        return pd.Series(max_value)
    return pd.Series(cycles_between.values_l())


def get_post_2M_dist(case, _type=0):
    """
    :param case:
    :param _type:
    :return: the distance between the last maintenance and the end date
    """
    res = istats.get_resources_of_type(case.instance, _type=_type)
    dist = case.instance.get_dist_periods
    next = case.instance.get_next_period
    first, last = (case.instance.get_param(p) for p in ['start', 'end'])
    last =  next(last)
    m_starts = case.get_maintenance_starts().filter_list_f(lambda v: v[0] in res)

    last_dist = \
        m_starts.\
        to_dict(1). \
        vapply(lambda v: [vv for vv in v if vv >= first]). \
        vapply(sorted).\
        vapply(lambda v: v+[last]).\
        vapply(lambda v: dist(v[1], last))

    return last_dist

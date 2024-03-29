import pandas as pd
import stochastic.instance_stats as istats
import numpy as np


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
    first, last = case.instance.get_start_end()
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


def get_1M_2M_dist(case, _type=0, count_1M=False):
    """
    :param case:
    :param _type:
    :param count_1M: count or not aircraft with just one maintenance
    :return: the distance between the first and second maintenance for each aircraft
    """
    res = istats.get_resources_of_type(case.instance, _type=_type)
    cycles = case.get_all_maintenance_cycles().filter(list(res))

    # if an aircraft starts a maintenance in the first period,
    # we need to count the empty cycle until that maintenance.
    first, last = case.instance.get_start_end()
    duration = case.instance.get_param('maint_duration')
    shift = case.instance.shift_period
    limit = shift(first, duration)
    # if an aircraft starts a maintenance in the first period;
    # we artificially add a shadow cycle before it
    cycles_edit = cycles.clean(func=lambda v: v[0][0] == limit).vapply(lambda v: [()]+v)
    cycles.update(cycles_edit)

    dist = case.instance.get_dist_periods
    max_value = case.instance.get_param('max_elapsed_time')
    size_horizon = dist(first, last) + 1

    if count_1M:
        # we count the second cycle.
        # sometimes an aircraft has no maintenances done.
        # (I thought this was impossible...)
        cycles_between = \
            cycles.vapply(lambda v: dist(*v[1]) + 1 if len(v)>1 else size_horizon)
    else:
        # we only want to see the distance when
        # there is a second maintenance
        cycles_between = \
            cycles.clean(func=lambda v: len(v)==3).\
            vapply(lambda v: dist(*v[1]) + 1 if len(v)>1 else size_horizon)

    if not len(cycles_between):
        return pd.Series(max_value)
    return pd.Series(cycles_between.values_l())


def get_all_maints_decisions(case, _type=0, num_max_maints=2):
    """
    :param case:
    :param _type:
    :return: the distance between the last maintenance and the end date
    """
    res = istats.get_resources_of_type(case.instance, _type=_type)
    next = case.instance.get_next_period
    first, last = case.instance.get_start_end()
    last =  next(last)
    m_starts = case.get_maintenance_starts().vfilter(lambda v: v[0] in res)

    return \
        m_starts. \
        vfilter(lambda v: v[1] >= first).\
        to_dict(1). \
        vapply(sorted).\
        vapply(lambda v: v+[last]).\
        vapply(lambda v: v[:num_max_maints])


def get_post_2M_dist(case, _type=0):
    first, last = case.instance.get_start_end()
    dist = case.instance.get_dist_periods
    maints = get_all_maints_decisions(case, _type)
    return maints.vapply(lambda v: dist(v[1], last))

def get_variance_dist(case, _type=0):
    distances = get_1M_2M_dist(case, _type=_type, count_1M=True)
    return np.var(distances)


def to_JR_format(case, _type=0):
    maint_starts = get_all_maints_decisions(case, _type)
    # we translate the period into a position:
    positions = case.instance.get_period_positions()
    return maint_starts.to_tuplist().vapply(lambda v: (v[0], positions[v[1]]))

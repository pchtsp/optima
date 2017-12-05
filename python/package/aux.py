# /usr/bin/python3
import arrow
import pandas as pd
import os
import datetime
import pickle


def get_months(start_month, end_month):
    """
    :param start_month: month in YYY-MM
    :param end_month: month in YYY-MM
    :return: list of months in text format
    """
    # we convert to arrows:
    start = arrow.get(start_month + "-01")
    end = arrow.get(end_month + "-01")
    periods = []
    current = start
    while current <= end:
        periods.append(current.format("YYYY-MM"))
        current = current.shift(months=1)
    return periods


def get_fixed_maintenances(previous_states, first_period, duration):
    previous_states_n = {}
    last_maint = {}
    planned_maint = []
    for (res, period) in previous_states:
        if res not in previous_states_n:
            previous_states_n[res] = []
        previous_states_n[res].append(period)
    # after initialization, we search for the scheduled maintenances that:
    # 1. do not continue the maintenance of the previous month
    # 2. happen in the last X months before the start of the planning period.
    for res in previous_states_n:
        _list = list(previous_states_n[res])
        _list_n = [period for period in _list if get_prev_month(period) not in _list
                   if shift_month(first_period, -duration) < period < first_period]
        if len(_list_n):
            last_maint[res] = max(_list_n)
            _temp = first_period
            finish_maint = shift_month(last_maint[res], duration-1)
            while _temp <= finish_maint:
                planned_maint.append((res, _temp))
                _temp = shift_month(_temp, 1)
    return planned_maint


def shift_month(month, shift=1):
    return arrow.get(month + "-01").shift(months=shift).format("YYYY-MM")


def get_prev_month(month):
    return shift_month(month, shift=-1)


def clean_dict(dictionary, default_value=0):
    return {key: value for key, value in dictionary.items() if value != default_value}


def tup_to_dict(at, result_col=0, is_list=True):
    cols = [col for col in range(len(at[0])) if col != result_col]
    table = pd.DataFrame(at).groupby(cols)[result_col]
    if is_list:
        return table.apply(lambda x: x.tolist()).to_dict()
    else:
        return table.apply(lambda x: x.tolist()[0]).to_dict()


def vars_to_tups(var):
    return [tup for tup in var if var[tup].value()]


def load_solution(path):
    if not os.path.exists(path):
        return False
    with open(path, 'rb') as f:
        return pickle.load(f)


def export_solution(path, obj, name=None):
    if not os.path.exists(path):
        return False
    if name is None:
        name = get_timestamp()
    path = os.path.join(path, name + ".pickle")
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    return True


def get_property_from_dic(dic, property):
    return {key: value[property] for key, value in dic.items()}


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M")
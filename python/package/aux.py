# /usr/bin/python
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


def get_prev_month(month):
    return arrow.get(month + "-01").shift(months=-1).format("YYYY-MM")


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


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M")
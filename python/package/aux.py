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
    return [date.format("YYYY-MM") for date in arrow.Arrow.range('month', start, end)]


def shift_month(month, shift=1):
    return arrow.get(month + "-01").shift(months=shift).format("YYYY-MM")


def get_prev_month(month):
    return shift_month(month, shift=-1)


def get_next_month(month):
    return shift_month(month, shift=1)


def clean_dict(dictionary, default_value=0):
    return {key: value for key, value in dictionary.items() if value != default_value}


def tup_to_dict(tup, result_col=0, is_list=True):
    cols = [col for col in range(len(tup[0])) if col != result_col]
    table = pd.DataFrame(tup).groupby(cols)[result_col]
    if is_list:
        return table.apply(lambda x: x.tolist()).to_dict()
    else:
        return table.apply(lambda x: x.tolist()[0]).to_dict()


def dicttup_to_dictdict(tupdict):
    """
    :param tupdict: a dictionary with tuples as lists
    :return: a (recursive) dictionary of dictionaries
    """
    # tupdict = solution['used']

    dictdict = {key[0]: {} for key in tupdict}
    for key, value in tupdict.items():
        dictdict[key[0]][key[1]] = value
    return dictdict


def dict_to_tup(dict_to_transform):
    tup_list = []
    for key, value in dict_to_transform.items():
        tup_list.append(tuple(list(key) + [value]))
    return tup_list


def tup_tp_start_finish(tup):
    """
    Takes a calendar tuple list of the form: (id, month) and
    returns a tuple list of the form (id, start_month, end_month)
    :param tup:
    :param first_period
    :return:
    """
    res_start_finish = []
    for (resource, period) in tup:
        prev_period = get_prev_month(period)
        if (resource, prev_period) in tup:
            continue
        stop = False
        next_period = period
        while not stop:
            next_period = get_next_month(next_period)
            stop = (resource, next_period) not in tup
        res_start_finish.append((resource, period, get_prev_month(next_period)))

    return res_start_finish


def vars_to_tups(var):
    return [tup for tup in var if var[tup].value()]


def get_property_from_dic(dic, property):
    return {key: value[property] for key, value in dic.items() if property in value}


def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M")


if __name__ == "__main__":
    pass
    # pd.DataFrame(aux.dict_to_tup(tupdict)).groupby(level=0).apply(lambda df: df.xs(df.name).to_dict()).to_dict()
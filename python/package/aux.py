# /usr/bin/python3

import arrow
import pandas as pd
import os
import datetime


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


def tup_to_dicts(dict_out, tup, value):
    elem = tup[0]
    if elem not in dict_out:
        dict_out[elem] = {}
    if len(tup) == 1:
        dict_out[elem] = value
        return dict_out
    else:
        tup_to_dicts(dict_out[elem], tup[1:], value)
    return dict_out


def dicts_to_tup(result, keys, content):
    if type(content) is not dict:
        result[tuple(keys)] = content
        return result
    for key, value in content.items():
        dicts_to_tup(result, keys + [key], value)
    return result


def dicttup_to_dictdict(tupdict):
    """
    Useful to get json-compatible objects from the solution
    :param tupdict: a dictionary with tuples as keys
    :return: a (recursive) dictionary of dictionaries
    """
    dictdict = {}
    for tup, value in tupdict.items():
        tup_to_dicts(dictdict, tup, value)
    return dictdict


def dictdict_to_dictup(dictdict):
    """
    Useful when reading a json and wanting to convert it to tuples.
    Opposite to dicttup_to_dictdict
    :param dictdict: a dictionary of dictionaries
    :return: a dictionary with tuples as keys
    """
    return dicts_to_tup({}, [], dictdict)


def dict_to_tup(dict_in):
    """
    The last element of the returned tuple was the dict's value.
    :param dict_in: dictionary indexed by tuples
    :return: a list of tuples.
    """
    tup_list = []
    for key, value in dict_in.items():
        tup_list.append(tuple(list(key) + [value]))
    return tup_list


def tup_to_start_finish(tup):
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
    content = tup_to_dicts({1: {2: {4: 5}}}, (1, 2, 3), 1)
    result = {}
    result = dicts_to_tup(result, [], {1: {2: {4: 5}}})
    result = dicts_to_tup({}, [], content)

    pass
    # pd.DataFrame(aux.dict_to_tup(tupdict)).groupby(level=0).apply(lambda df: df.xs(df.name).to_dict()).to_dict()
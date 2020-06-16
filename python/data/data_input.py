import os
import data.dates as aux
import numpy as np
import copy
import json
import pickle
import ujson
import math


def load_data(path, file_type=None):
    if file_type is None:
        splitext = os.path.splitext(path)
        if len(splitext) == 0:
            raise ImportError("file type not given")
        else:
            file_type = splitext[1][1:]
    if file_type not in ['json', 'pickle']:
        raise ImportError("file type not known: {}".format(file_type))
    if not os.path.exists(path):
        return False
    if file_type == 'pickle':
        with open(path, 'rb') as f:
            return pickle.load(f)
    if file_type == 'json':
        with open(path, 'r') as f:
            return ujson.load(f)


def load_data_zip(zipobj, path, file_type='json'):
    if file_type not in ['json']:
        raise ImportError("file type not known: {}".format(file_type))
    if file_type == 'json':
        try:
            data = zipobj.read(path)
        except KeyError:
            return False
        return ujson.loads(data)


def export_data(path, dictionary, name=None, file_type="json", exclude_aux=False):
    # I'm gonna add the option to exclude the 'aux' section of the object, if exists
    # we're assuming obj is a dictionary
    dictionary = copy.deepcopy(dictionary)
    if exclude_aux and 'aux' in dictionary:
        dictionary.pop('aux', None)
    if not os.path.exists(path):
        os.mkdir(path)
    if name is None:
        name = aux.get_timestamp()
    path = os.path.join(path, name + "." + file_type)
    if file_type == "pickle":
        with open(path, 'wb') as f:
            pickle.dump(dictionary, f)
    if file_type == 'json':
        with open(path, 'w') as f:
            json.dump(dictionary, f, cls=MyEncoder, indent=4, sort_keys=True)
    return True


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def copy_dict(_dict):
    return ujson.loads(ujson.dumps(_dict))


def round_down(x, num=10):
    try:
        return int(math.floor(x / num)) * num
    except TypeError:
        return None

def round_up(x, num=10):
    try:
        return int(math.ceil(x / num)) * num
    except TypeError:
        return None

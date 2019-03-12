import numpy as np
import pandas as pd


class SuperDict(dict):

    def keys_l(self):
        return list(self.keys())

    def values_l(self):
        return list(self.values())

    def clean(self, default_value=0, func=None):
        if func is None:
            func = lambda x: x != default_value
        return SuperDict({key: value for key, value in self.items() if func(value)})

    def filter(self, indices, check=True):
        if type(indices) is not list:
            indices = [indices]
        if not check:
            return SuperDict({k: self[k] for k in indices if k in self})
        bad_elem = np.setdiff1d(indices, list(self.keys()))
        if len(bad_elem) > 0:
            raise KeyError("following elements not in keys: {}".format(bad_elem))
        return SuperDict({k: self[k] for k in indices})

    def to_dictdict(self):
        """
        Useful to get json-compatible objects from the solution
        :param self: a dictionary with tuples as keys
        :return: a (recursive) dictionary of dictionaries
        """
        dictdict = SuperDict()
        for tup, value in self.items():
            dictdict.tup_to_dicts(tup, value)
        return dictdict

    def tup_to_dicts(self, tup, value):
        """
        recursive function to build a nested dictionary from a tuple dictionary
        It is also very useful to initialize dictionaries with keys
        :param tup: remaining tuple with keys
        :param value: value to assign at the end of nested dictionary
        :return:
        """
        elem = tup[0]
        if elem not in self:
            # initialize a new level of dictionary
            self[elem] = SuperDict()
        if len(tup) == 1:
            # if on last level: assign value and return
            self[elem] = value
            return self
        else:
            # if not on last level, continue with remaining keys in tuple
            self[elem].tup_to_dicts(tup[1:], value)
        return self

    def dicts_to_tup(self, keys, content):
        """
        recursive function to accumulate tuples from nested dictionaries
        :param keys: already extracted keys (list dependant on current depth of nested dictionary)
        :param content: posible dictionary to continue flatting
        :return:
        """
        if type(content) is not SuperDict:
            self[tuple(keys)] = content
            return self
        for key, value in content.items():
            self.dicts_to_tup(keys + [key], value)
        return self

    def to_dictup(self):
        """
        Useful when reading a json and wanting to convert it to tuples.
        Opposite to to_dictdict
        :param self: a dictionary of dictionaries
        :return: a dictionary with tuples as keys
        """
        return SuperDict().dicts_to_tup([], self)

    def reverse(self):
        return SuperDict({v: k for k, v in self.items()})

    def list_reverse(self):
        """
        :param self: a dictionary with a list as a result
        :return: a dictionary with the list elements as keys and
        old keys as values.
        """
        new_keys = list(set(val for l in self.values() for val in l))
        dict_out = SuperDict({k: [] for k in new_keys})
        for k, v in self.items():
            for el in v:
                dict_out[el].append(k)
        return dict_out

    def to_tuplist(self):
        """
        The last element of the returned tuple was the dict's value.
        :param self: dictionary indexed by tuples
        :return: a list of tuples.
        """
        import package.tuplist as tl

        tup_list = tl.TupList()
        for key, value in self.items():
            if type(key) is tuple:
                tup_list.append(tuple(list(key) + [value]))
            else:
                tup_list.append(tuple([key, value]))
        return tup_list

    def fill_with_default(self, keys, default=0):
        return SuperDict({**{k: default for k in keys}, **self})

    def get_property(self, property):
        return SuperDict({key: value[property] for key, value in self.items() if property in value})

    def to_lendict(self):
        return SuperDict({k: len(v) for k, v in self.items()})

    def index_by_property(self, property, get_list=False):
        el = self.keys_l()[0]
        if property not in self[el]:
            raise IndexError('property {} is not present in el {} of dict {}'.
                             format(property, el, self))

        result = {v[property]: {} for v in self.values()}
        for k, v in self.items():
            result[v[property]][k] = v

        result = SuperDict.from_dict(result)
        if get_list:
            return result.values_l()
        return result

    def index_by_part_of_tuple(self, position, get_list=False):
        el = self.keys_l()[0]
        if len(el) <= position:
            raise IndexError('length of dict {} keys is smaller than position {}'.
                             format(self, position))

        result = {k[position]: {} for k in self.keys()}
        for k, v in self.items():
            result[k[position]][k] = v

        result = SuperDict.from_dict(result)
        if get_list:
            return result.values_l()
        return result

    def to_df(self, **kwargs):
        return pd.DataFrame.from_dict(self, **kwargs)

    @classmethod
    def from_dict(cls, dictionary):
        if type(dictionary) is not dict:
            return dictionary
        dictionary = cls(dictionary)
        for key, value in dictionary.items():
            dictionary[key] = cls.from_dict(value)
        return dictionary

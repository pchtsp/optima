import numpy as np
import package.auxiliar as aux


class TupList(list):

    def filter(self, indices):
        """
        filters the tuple of each element of the list according
        to a list of positions
        :param indices: a list of positions
        :return: a new TuplList with the modifications
        """
        if type(indices) is not list:
            # indices = [indices]
            return TupList([np.take(tup, indices) for tup in self])
        return TupList([tuple(np.take(tup, indices)) for tup in self])

    def filter_list_f(self, function):
        """
        :param function: function to apply to tuple
        :return: filtered list of tuple
        """
        return TupList([i for i in self if function(i)])

    def to_dict(self, result_col=0, is_list=True, indices=None):
        """
        This magic function converts a tuple list into a dictionary
        by taking one or several of the columns as the result.
        :param result_col: a list of positions of the tuple for the result
        :param is_list: the value of the dictionary will be a list?
        :param indices: optional way of determining the indeces instead of
            being the complement of result_col
        :return: a dictionary
        """
        import package.superdict as sd

        if type(result_col) is not list:
            result_col = [result_col]
        if len(self) == 0:
            return sd.SuperDict()
        if indices is None:
            indices = [col for col in range(len(self[0])) if col not in result_col]
        result = sd.SuperDict()
        for tup in self:
            index = tuple(np.take(tup, indices))
            if len(index) == 1:
                index = index[0]
            content = tuple(np.take(tup, result_col))
            if len(content) == 1:
                content = content[0]
            if not is_list:
                result[index] = content
                continue
            if index not in result:
                result[index] = []
            result[index].append(content)
        return result

    def add(self, *args):
        """
        this is just a shortcut for doing
            list.append((arg1, arg2, arg3))
        by doing:
            list.add(arg1, arg2, arg3)
        which is a little more friendly and short
        :param args: any number of elements to append
        :return: nothing.
        """
        self.append(tuple(args))

    def unique(self):
        return TupList(np.unique(self))

    def unique2(self):
        return TupList(set(self))

    def tup_to_start_finish(self, pp=1):
        """
        Takes a calendar tuple list of the form: (id, month) and
        returns a tuple list of the form (id, start_month, end_month)
        it works with a bigger tuple too.
        :param self: [(id, month), (id, month)]
        :param pp: the position in the tuple where the period is
        :return:
        """
        self.sort(key=lambda x: (x[0], x[pp]))
        res_start_finish = []
        last_tup = ()
        all_periods = []
        current_period = []
        for tup in self:
            if tup == self[0] or tup[0] != last_tup[0] or\
                    tup[pp] != aux.get_next_month(last_tup[pp]):
                if len(current_period):
                    all_periods.append(current_period)
                current_period = [tup]
            else:
                current_period.append(tup)
            last_tup = tup

        for list_tup in all_periods:
            res_start_finish.append(tuple(list(list_tup[0]) + [list_tup[-1][pp]]))
        return TupList(res_start_finish)
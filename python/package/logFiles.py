# /usr/bin/python3
import re
import pandas as pd
import io
import numpy as np
import os


class LogFile(object):
    """
    This represents the log files that solvers return.
    We implement functions to get different information
    """

    def __init__(self, path):
        with open(path, 'r') as f:
            content = f.read()

        self.content = content
        self.number = r'[\de\.\+]+'
        self.numberSearch = r'({})'.format(self.number)
        self.wordSearch = r'([\w, ]+)'

    def apply_regex(self, regex, content_type=None, first=True, pos=None, **kwargs):
        solution = re.findall(regex, self.content, **kwargs)
        if solution is None:
            return None
        if not first:
            return solution
        if len(solution) == 0:
            return None
        possible_tuple = solution[0]
        if pos is None:
            if content_type is None:
                return possible_tuple
            if content_type == "float":
                return [float(val) for val in possible_tuple]
            if content_type == "int":
                return [int(val) for val in possible_tuple]
        value = possible_tuple[pos]
        if content_type == 'int':
            return int(value)
        if content_type == 'float':
            return float(value)
        return possible_tuple[pos]


    # CPLEX PART:

    def get_objective(self):
        """
        :return: tuple of length 2
        """
        regex = r'MIP\s+-\s+{1}:\s+Objective\s+=\s+{0}\n'.format(self.numberSearch, self.wordSearch)
        result = self.apply_regex(regex, flags=re.MULTILINE)
        if result is None:
            return None, None
        return result[0], float(result[1])

    def get_gap(self):
        """
        :return: tuple of length 3: bound, absolute gap, relative gap
        """
        regex = r'Current MIP best bound =  {0} \(gap = {0}, {0}%\)'.format(self.numberSearch)
        result = self.apply_regex(regex, content_type='float')
        if result is None:
            return None, None, None
        return result

    def get_matrix(self):
        """
        :return: tuple of length 3
        """
        regex = r'Reduced MIP has {0} rows, {0} columns, and {0} nonzeros'.format(self.numberSearch)
        result = self.apply_regex(regex, content_type="int")
        if result is None:
            return None, None, None
        return result

    def get_cuts(self):
        """
        :return: dictionary of cuts
        """
        regex = r'{1} cuts applied:  {0}'.format(self.numberSearch, self.wordSearch)
        result = self.apply_regex(regex, first=False)
        if result is None:
            return None
        return {k[0]: int(k[1]) for k in result}

    def get_lp_presolve(self):
        """
        :return: tuple  of length 3
        """
        regex = r'Presolve time = {0} sec. \({0} ticks\)'.format(self.numberSearch)
        time = self.apply_regex(regex, pos=0, content_type="float")

        regex = r'LP Presolve eliminated {0} rows and {0} columns'.format(self.numberSearch)
        result = self.apply_regex(regex, content_type="int")
        if result is None:
            result = None, None
        return {'time': time, 'rows': result[0], 'cols': result[1]}

    def get_time(self):
        regex = r'Solution time =\s+{0} sec\.\s+Iterations = {0}\s+Nodes = {0}'.format(self.numberSearch)
        return self.apply_regex(regex, content_type="float", pos=0)

    def get_root_time(self):
        regex = r'Root relaxation solution time = {0} sec\. \({0} ticks\)'.format(self.numberSearch)
        return self.apply_regex(regex, pos=0, content_type="float")

    def get_cuts_time(self):
        regex = r'Elapsed time = {0} sec\. \({0} ticks, tree = {0} MB, solutions = {0}\)'.format(self.numberSearch)
        return self.apply_regex(regex, content_type="float", pos=0)

    def get_results_after_cuts(self):
        progress = self.get_progress()
        df_filter = np.all((progress.Node.str.match(r"^\*?\s*0"),
                            progress.NodesLeft.str.match(r"^\+?\s*2")),
                           axis=0)
        sol_value = progress.BestInteger.iloc[-1]
        relax_value = progress.CutsBestBound.iloc[-1]
        sol_after_cuts = None
        relax_after_cuts = None

        if np.any(df_filter):
            sol_value = progress.BestInteger[df_filter].iloc[0]
            relax_value = progress.CutsBestBound[df_filter].iloc[0]
        if re.search(r'\s*\d', sol_value):
            sol_after_cuts = float(sol_value)
        if re.search(r'\s*\d', relax_value):
            relax_after_cuts = float(relax_value)
        return relax_after_cuts, sol_after_cuts

    def get_first_results(self):
        progress = self.get_progress()
        df_filter = progress.CutsBestBound.apply(lambda x: re.search(r"^\s*\d", x) is not None)
        first_relax = float(progress.CutsBestBound[df_filter][0])

        df_filter = progress.BestInteger.str.match(r"^\s*\d")
        first_solution = "-"
        if len(df_filter) > 0:
            first_solution = float(progress.BestInteger[df_filter].iloc[0])
        return first_relax, first_solution

    def get_progress(self):
        """
        :return: pandas dataframe with 8 columns
        """
        # before we clean the rest of lines:
        regex = r'(^\*?\s+\d.*$)'
        lines = self.apply_regex(regex, first=False, flags=re.MULTILINE)
        lines_content = io.StringIO('\n'.join(lines)[2:])
        names = ['Node', 'NodesLeft', 'Objective', 'IInf', 'BestInteger', 'CutsBestBound', 'ItCnt', 'Gap']
        widths = [7, 6, 14, 6, 14, 14, 9, 9]
        df = pd.read_fwf(lines_content, names=names, header=None, dtype=str, widths=widths)
        return df

    def get_log_info_cplex(self):
        cons, vars, nonzeroes = self.get_matrix()
        status, objective = self.get_objective()
        bound, gap_abs, gap_rel = self.get_gap()
        if bound is None:
            bound = objective
            gap_abs = 0
            gap_rel = 0
        cuts = self.get_cuts()
        presolve = self.get_lp_presolve()
        time_out = self.get_time()
        rootTime = self.get_root_time()
        cutsTime = self.get_cuts_time()
        progress = self.get_progress()
        after_cuts, sol_after_cuts = self.get_results_after_cuts()
        first_relax, first_solution = self.get_first_results()

        return {
            'bound_out': bound,
            'objective_out': objective,
            'gap_out': gap_rel,
            'time_out': time_out,
            'cons': cons,
            'vars': vars,
            'nonzeros': nonzeroes,
            'cuts': cuts,
            'rootTime': rootTime,
            'cutsTime': cutsTime,
            'presolve': presolve,
            'first_relaxed': first_relax,
            'after_cuts': after_cuts,
            'progress': progress,
            'first_solution': first_solution,
            'sol_after_cuts': sol_after_cuts
        }


    # GUROBI PART:
    def get_log_info_gurobi(self):

        regex = r'Best objective {0}, best bound {0}, gap {0}%'.format(self.numberSearch)
        # "Best objective 6.500000000000e+01, best bound 5.800000000000e+01, gap 10.7692%"
        solution = re.search(regex, self.content)

        regex = r'Optimize a model with {0} rows, {0} columns and {0} nonzeros'.format(self.numberSearch)
        size = re.search(regex, self.content)

        regex = r"Explored {0} nodes \({0} simplex iterations\) in {0} seconds".format(self.numberSearch)
        stats = re.findall(regex, self.content)
        time_out = -1
        if stats:
            time_out = float(stats[0][2])

        return {
            'bound_out': float(solution.group(2)),
            'objective_out': float(solution.group(1)),
            'gap_out': float(solution.group(3)),
            'cons': int(size.group(1)),
            'vars': int(size.group(2)),
            'nonzeros': int(size.group(3)),
            'time_out': time_out
        }


if __name__ == "__main__":
    import pprint as pp
    route = '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/'
    # path = '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201801102259/results.log'
    # path = '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201801141334/results.log'
    # path = '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201801131817/results.log'
    # path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201801141331/results.log"
    exps = ['201801102259', '201801141334', '201801131817', '201801141331']

    exps = ['201801141705']
    # exps = ['201801102259']
    for e in exps:
        path = os.path.join(route, e, 'results.log')
        log = LogFile(path)
        result = log.get_log_info_cplex()
        pp.pprint(result)
    #
    # re.search(r"\s*\d", result.CutsBestBound[1])

    # status, objective = log.get_objective()
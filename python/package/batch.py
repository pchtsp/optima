# /usr/bin/python3

import package.experiment as exp
import data.data_input as di

import pytups.tuplist as tl
import pytups.superdict as sd

import orloge as ol
import os
import zipfile
import pandas as pd
import shutil
import re


class Batch(object):
    """
    This is a group of experiments.
    It reads a path of the form /PATH/TO/BATCH/ where:
    /PATH/TO/BATCH/scenario1/instance1/
    /PATH/TO/BATCH/scenario1/instance2/
    /PATH/TO/BATCH/scenario2/instance2/
    ...
    /PATH/TO/BATCH/scenarioX/instanceY/

    /PATH/TO/BATCH/scenario1/instance1/ is assumed to be the path for an experiment.
    i.e., exp.Experiment.from_dir('/PATH/TO/BATCH/scenario1/instance1/') should work.
    names are obtained by inspection so it can be any name for scenario or instance.

    if no_scenario is True:
    /PATH/TO/BATCH/instance1/
    /PATH/TO/BATCH/instance2/
    ...
    /PATH/TO/BATCH/instanceY/
    """

    def __init__(self, path, no_scenario=False, scenarios=None):
        """

        :param path: path to results
        :param no_scenario: if True, there is no scenarios, instances directly
        :param scenarios: in order to filter the scenarios to load
        """
        self.path = path
        self.cases = None
        self.logs = None
        self.errors = None
        self.options = None
        self.seeds = None
        self.no_scenario = no_scenario
        self.scenarios = scenarios

    def get_instances_paths(self):
        scenarios = self.scenarios
        if not scenarios:
            scenarios = os.listdir(self.path)
        scenario_paths = {s: os.path.join(self.path, s) for s in scenarios}
        if self.no_scenario:
            return sd.SuperDict.from_dict(scenario_paths)
        scenario_instances = {s: os.listdir(v) for s, v in scenario_paths.items()}
        scenario_paths_in, instances_paths_in = self.re_make_paths(scenario_instances)
        return sd.SuperDict.from_dict(instances_paths_in).to_dictup()

    def re_make_paths(self, scenario_instances):
        scenario_paths = {s: os.path.join(self.path, s) for s in scenario_instances}
        instances_paths = {s: {i: os.path.join(scenario_paths[s], i) for i in instances}
                           for s, instances in scenario_instances.items()}
        return scenario_paths, instances_paths

    def get_cases(self):
        if self.cases is not None:
            return self.cases

        load_data = exp.Experiment.from_dir
        self.cases = self.get_instances_paths().vapply(load_data)
        return self.cases

    def get_solver(self):
        opt_info = self.get_options()
        available_solvers = ['CPLEX', 'GUROBI', 'CBC']
        default = 'CPLEX'
        try:
            el = list(opt_info.keys())[0]
            engine = opt_info[el]['solver']
        except:
            return default
        if '.' in engine:
            engine, solver = engine.split('.')
        else:
            solver = engine
        if solver in available_solvers:
            return solver
        # one last try
        for s in available_solvers:
            if re.search(s, solver):
                return s
        return default

    def get_logs(self, get_progress=False, solver=None):
        if self.logs is not None:
            return self.logs

        solver = self.get_solver()

        self.logs = \
            self.get_instances_paths(). \
            vapply(lambda v: os.path.join(v, 'results.log')).\
            clean(func=os.path.exists). \
            vapply(lambda v: ol.get_info_solver(v, solver, get_progress=get_progress))
        return self.logs

    def get_json(self, name):
        load_data = di.load_data

        return \
            self.get_instances_paths().\
            vapply(lambda v: os.path.join(v, name)). \
            vapply(load_data). \
            clean(). \
            vapply(sd.SuperDict.from_dict)

    def get_errors(self):
        if self.errors is not None:
            return self.errors

        self.errors = \
            self.get_json('errors.json').\
            vapply(lambda v: v.to_dictup()). \
            to_lendict()

        return self.errors

    def get_options(self):
        if self.options is not None:
            return self.options
        self.options = self.get_json('options.json')
        return self.options

    def get_errors_df(self):
        errors = self.get_errors()
        return self.format_df(errors).rename(columns={0: 'errors'})

    def get_seeds(self):
        if self.seeds is not None:
            return self.seeds
        self.seeds = self.get_options().get_property('simulation').get_property('seed')
        return self.seeds

    def format_df(self, table):
        seeds = self.get_seeds()
        table = table.to_df(orient='index')
        axis_name = 'name'
        if not self.no_scenario:
            table.index = pd.MultiIndex.from_tuples(table.index)
            axis_name = ('scenario', 'name')
        table['instance'] = table.index.map(seeds)
        return table.rename_axis(axis_name).reset_index()

    def get_log_df(self, **kwargs):
        log_info = self.get_logs(**kwargs)
        table = self.format_df(log_info)

        for name in ['matrix', 'presolve', 'matrix_post']:
            try:
                aux_table = table[name].apply(pd.Series)
                aux_table.columns = [name + '_' + c for c in aux_table.columns]
                table = pd.concat([table, aux_table], axis=1)
            except:
                pass

        return table

    def get_status_df(self, vars_extract=None):
        table = self.get_log_df()
        if vars_extract is None:
            vars_extract = ['scenario', 'name', 'sol_code', 'status_code',
                            'time', 'gap', 'best_bound', 'best_solution']

        master = \
            pd.DataFrame({'sol_code': [ol.LpSolutionIntegerFeasible, ol.LpSolutionOptimal,
                                       ol.LpSolutionInfeasible, ol.LpSolutionNoSolutionFound],
                          'status': ['IntegerFeasible', 'Optimal', 'Infeasible', 'NoIntegerFound']})

        status_df = table.filter(vars_extract).merge(master, on='sol_code')

        status_df['gap_abs'] = status_df.best_solution - status_df.best_bound
        return status_df

    def list_experiments(self, exp_list=None, get_log_info=True, get_exp_info=True):
        """

        :param exp_list: a list to filter cases to return
        :param get_log_info: whether to get or not information on the log files
        :param get_exp_info: whether to get or not information on the instances
        :return: dictionary of data per each instance
        """
        notNone = lambda x: x is not None
        exp_info = {}
        log_info = {}
        opt_info = self.get_options()
        if get_exp_info:
            cases = self.get_cases().clean(func=notNone)
            exp_info = cases.vapply(lambda v: v.instance.get_param())
            inst_info = cases.vapply(lambda v: v.instance.get_info())
            exp_info.update(inst_info)
        if get_log_info:
            log_info = self.get_logs()
        exp_info.update(log_info)
        exp_info.update(opt_info)
        if exp_list:
            return exp_info.filter(exp_list)
        return exp_info

    def clean_experiments(self, clean=True, func=None):
        """
        loads and cleans all experiments that are incomplete
        :param clean: if set to false it only shows the files instead of deleting them
        :param func: optional function to filter cases
        :return: deleted experiments
        """

        cases = self.get_cases()
        if func is not None:
            func = lambda x: x is None
        cases = cases.clean(func=func)
        paths = self.get_instances_paths().filter(cases.keys_l())

        if clean:
            for path in paths.values():
                shutil.rmtree(path)
        return paths.values_l()

    def generate_JR_dataset(self):
        """
        This is a format that Johannes Roysset requested.
        :return:
        """
        import stochastic.solution_stats as sol_stats

        cases = self.get_cases()
        optimal_cases = self.get_logs().vfilter(lambda v: v['status_code'] == 1)
        maintenances = \
            cases. \
                filter(optimal_cases). \
                vapply(sol_stats.to_JR_format).to_tuplist()
        data = \
            maintenances. \
                take([1, 2, 3]). \
                vapply(lambda v: (v[0], '{:02d}'.format(int(v[1])), v[2])). \
                sorted(). \
                to_list()
        cols = ['instance', 'aircraft', 'date']
        tab = \
            pd.DataFrame. \
                from_records(data, columns=cols)
        tab['count'] = 1
        tab['num_maint'] = tab.groupby(['instance', 'aircraft'])['count'].cumsum()
        tab2 = \
            tab. \
                set_index(['instance', 'aircraft', 'num_maint']). \
                drop('count', axis=1). \
                unstack(['aircraft', 'num_maint'], 'num_maint')
        return tab2


class ZipBatch(Batch):
    """
    Only difference is it's contained inside a zip file.
    """
    def __init__(self, path, *args, **kwargs):
        name, ext = os.path.splitext(path)
        zip_ext = '.zip'
        if not ext:
            path = name + zip_ext
        elif ext != zip_ext:
            raise ValueError('Only zip is supported')
        super().__init__(path, *args, **kwargs)

    def get_instances_paths(self):
        num_slashes = 2
        keys_positions = [1, 2]
        if self.no_scenario:
            num_slashes = 1
            keys_positions = 1
        zipobj = zipfile.ZipFile(self.path)
        all_files = tl.TupList(self.dirs_in_zip(zipobj))
        scenario_instances = \
            all_files.\
                vfilter(lambda f: f.count("/") == num_slashes)
        keys = \
            scenario_instances.\
            vapply(str.split, '/').\
            vapply(tuple).\
            take(keys_positions)
        result_dict = sd.SuperDict(zip(keys, scenario_instances))
        if self.scenarios:
            scenarios = set(self.scenarios)
            return result_dict.kfilter(lambda k: k[0] in scenarios)
        else:
            return result_dict


    def re_make_paths(self, scenario_instances):
        pass

    def get_cases(self):
        if self.cases is not None:
            return self.cases

        zipobj = zipfile.ZipFile(self.path)
        load_data = lambda e: exp.Experiment.from_zipfile(zipobj, e)
        self.cases = self.get_instances_paths().vapply(load_data)
        return self.cases

    def get_logs(self, get_progress=False, solver=None):
        if self.logs is not None:
            return self.logs

        zipobj = zipfile.ZipFile(self.path)
        if not solver:
            solver = self.get_solver()

        def _read_zip(x):
            try:
                return zipobj.read(x)
            except:
                return 0

        if solver == 'HEUR':
            func_to_get_log = get_info_heur
            filename = '/output.log'
        else:
            func_to_get_log = ol.get_info_solver
            filename = '/results.log'

        self.logs = \
            self.get_instances_paths(). \
            vapply(lambda v: v + filename).\
            vapply(_read_zip). \
            clean(). \
            vapply(lambda x: str(x, 'utf-8')). \
            vapply(lambda v: func_to_get_log(v, solver, get_progress=get_progress, content=True))
        return self.logs

    def get_json(self, name):
        zipobj = zipfile.ZipFile(self.path)
        load_data = lambda v: di.load_data_zip(zipobj=zipobj, path=v)

        return \
            self.get_instances_paths().\
            vapply(lambda v: v + '/' + name). \
            vapply(load_data). \
            clean(). \
            vapply(sd.SuperDict.from_dict)

    def parent_dirs(self, pathname, subdirs=None):
        """Return a set of all individual directories contained in a pathname

        For example, if 'a/b/c.ext' is the path to the file 'c.ext':
        a/b/c.ext -> set(['a','a/b'])
        """
        if subdirs is None:
            subdirs = set()
        parent = os.path.dirname(pathname)
        if parent:
            subdirs.add(parent)
            self.parent_dirs(parent, subdirs)
        return subdirs

    def dirs_in_zip(self, zf):
        """Return a list of directories that would be created by the ZipFile zf"""
        alldirs = set()
        for fn in zf.namelist():
            alldirs.update(self.parent_dirs(fn))
        return alldirs


class LogHeuristic(ol.LogFile):

    def __init__(self, path, **options):
        super().__init__(path, **options)
        self.progress_filter = '.*INFO:time=.*'
        self.progress_names = ['Date', 'Time', 'ItpNode', 'Temperature', 'Objective', 'BestInteger', 'Errors']
        self.name = 'HEUR'

    def process_line(self, line):
        keys = ['t', 'it', 'temp', 'curr', 'best', 'err']
        args = {k: self.numberSearch for k in keys}
        info_re = 'time={t}, iteration={it}, temperaure={temp}, current={curr}, best={best}, errors={err}'.format(**args)
        str_look = r'(\d\d\d\d-\d\d-\d\d\ \d\d:\d\d:\d\d,\d\d\d) INFO:{}'.format(info_re)
        find = re.search(str_look, line)
        if not find:
            return None
        return find.groups()

    def get_log_info(self):
        if self.options.get('get_progress', True):
            progress = self.get_progress()
        else:
            progress = pd.DataFrame()
        first_solution = time = best_solution = None
        if len(progress):
            last_line = progress.iloc[-1]
            first_line = progress.iloc[0]
            best_solution = float(last_line.BestInteger)
            time = float(last_line.Time)
            first_solution = float(first_line.Objective)

        return {
            'version': 1,
            'solver': self.name,
            'status': 1,
            'best_bound': -1,
            'best_solution': best_solution,
            'gap': -1,
            'time': time,
            'matrix_post': dict(),
            'matrix': dict(),
            'cut_info': dict(),
            'rootTime': -1,
            'presolve': -1,
            'first_relaxed': -1,
            'progress': progress,
            'first_solution': first_solution,
            'status_code': 1,
            'sol_code': 1,
            'nodes': -1
        }

def get_info_heur(path, solver, **options):
    log_obj = LogHeuristic(path, **options)
    return log_obj.get_log_info()

if __name__ == '__main__':
    import package.params as params
    nn = "IT000125_20190725"
    nn = 'dell_20190422_new_new_cplex_newobjs_rut'
    self = ZipBatch(path=params.PATHS['results'] + nn)
    cases = self.get_cases()
    logs = self.get_logs()
    errors = self.get_log_df()
    rrr = self.list_experiments()

    other_path = '/home/pchtsp/Documents/projects/optima_results_old/'
    nn = "clust1_20190408_remake"
    self = Batch(path=other_path + nn)
    cases = self.get_cases()
    logs = self.get_logs()
    errors = self.get_errors()
    err_df = self.get_errors_df()

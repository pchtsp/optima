import os
import datetime as dt

# TODO: TEMP not sure why python doesn't find the ENV VAR from bash.bashrc
if 'GUROBI_HOME' in os.environ:
    if 'LD_LIBRARY_PATH' not in os.environ:
        os.environ['LD_LIBRARY_PATH'] = ""
    os.environ['LD_LIBRARY_PATH'] += ':' + os.environ['GUROBI_HOME'] + "/lib"

filename = os.path.realpath(__file__)
directory = os.path.dirname(filename)
path_project = os.path.join(directory, '..', '..') + '/'
path_root = os.path.join(path_project, '..') + '/'
path_results = os.path.join(path_root, 'optima_results/')


PATHS = {
    'root': path_root
    ,'results': path_results
    ,'experiments': path_results + "experiments/"
    ,'img': path_project + "img/"
    ,'latex': path_project + "latex/"
    ,'data': path_project + "data/"
}

PATHS['input'] = PATHS['data'] + 'raw/parametres_DGA_final.xlsm'
PATHS['hist'] = PATHS['data'] + 'raw/Planifs M2000.xlsm'

params_cplex = \
[
'set mip cuts flowcovers 1'
, 'set mip cuts mircut 1'
, 'set mip strategy backtrack 0.1'
, 'set mip strategy heuristicfreq 100'
, 'set mip strategy presolvenode 2'
, 'set mip strategy probe 3'
, 'set mip limits gomorycand 10000'
, 'set mip limits gomorypass 10'
  ]

params_cbc = ["presolve on",
             "gomory on",
             "probing on"]

OPTIONS = {
    'timeLimit': 600  # seconds
    , 'solver': "CPLEX"  # HEUR, CPO, CHOCO, CPLEX, GUROBI, CBC, HEUR_mf HEUR_mf_CPLEX
    , 'black_list': ['O8', 'O10', 'O6']
    , 'white_list': []
    , 'start': '2018-01'
    , 'num_period': 90
    , 'path': os.path.join(
        PATHS['experiments'],
        dt.datetime.now().strftime("%Y%m%d%H%M")
    ) + '/'
    , 'simulate': True
    , 'exclude_aux': False
    # heuristic params:
    , 'seed': 42
    , 'num_change': [0.8, 0.1, 0.1]
    , 'temperature': 2
    , 'prob_free_aircraft': 0.1
    , 'prob_free_periods': 0.1
    , 'cooling': 0.9995
    , 'debug': False
    , 'max_iters': 99999999
    , 'prob_delete_maint': 0.8
    # MIP params:
    , 'noise_assignment': True
    , 'gap': 0
    , 'gap_abs': 40
    , 'memory': None
    , 'slack_vars': "No"  # ['No', 'Yes', 3, 6]
    , 'integer': False
    , 'writeLP': False
    , 'writeMPS': False
    , 'price_rut_end': 0
    , 'solver_add_opts': {'CPLEX': params_cplex, 'CBC': params_cbc}
    , 'mip_start': False
    , 'fix_start': False
    , 'threads': None
    , 'solver_path': None
    , 'keepfiles': 0
    , 'do_not_solve': False
    # stats-cut-data
    , 'StochCuts' : {
        'active': False,
        'bounds': ['min', 'max'],  # ['min', 'max']
        'cuts': ['maints', 'mean_2maint', 'mean_dist']
    }, 'reduce_2M_window': {
        'active': False,
        'window_size': 10
    }
    # simulation params:
    , 'simulation': {
        'num_resources': 15  # this depends on the number of tasks actually
        , 'num_parallel_tasks': 1
        , 'maint_duration': 6
        , 'max_used_time': 1000
        , 'max_elapsed_time': 60  # max time without maintenance
        , 'max_elapsed_time_2M': None
        , 'elapsed_time_size': 30  # size of window to do next maintenance
        , 'elapsed_time_size_2M': None
        , 'min_usage_period': 0 # minimum consumption per period
        , 'perc_capacity': 0.15
        , 'min_avail_percent': 0.1  # min percentage of available aircraft per type
        , 'min_avail_value': 1  # min num of available aircraft per type
        , 'min_hours_perc': 0.5  # min percentage of maximum possible hours of fleet type
        , 'seed': 47
        # The following are fixed options, not arguments for the scenario:
        , 't_min_assign': [2, 3, 6]  # minimum assignment time for tasks
        , 'initial_unbalance': (-3, 3)
        , 't_required_hours': (30, 50, 80) # triangular distribution params
        , 't_num_resource': (2, 5)
        , 't_duration': (6, 12)
        , 'perc_in_maint': 0.07
        , 'perc_add_capacity': 0.1  # probability of having an added capacity to mission
    }
}


import os
import datetime as dt

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


temp_path = \
    os.path.join(
        PATHS['experiments'],
        dt.datetime.now().strftime("%Y%m%d%H%M")
    ) + '/'

OPTIONS = {
    'timeLimit': 600  # seconds
    , 'solver': "HEUR_mf"  # HEUR, CPO, CHOCO, CPLEX, GUROBI, CBC, HEUR_mf HEUR_mf_CPLEX
    , 'black_list': ['O8', 'O10', 'O6']  # only used to read from DGA Excel.
    , 'white_list': []  # only used to read from DGA Excel.
    , 'start': '2018-01'
    , 'num_period': 90
    , 'simulate': True
    , 'template': True
    , 'solve': True
    , 'graph': 2
    , 'warm_start': False
    , 'R_HOME': r'C:\Users\pchtsp\Downloads\R-Portable\App\R-Portable'
    # data
    , 'path': temp_path
    , 'input_template_path': temp_path + 'template_in.xlsx'
    , 'output_template_path': temp_path + 'template_out.xlsx'
    , 'exclude_aux': False
    # heuristic params:
    , 'seed': 42
    , 'num_change': [0.8, 0.1, 0.1]
    , 'temperature': 2
    , 'prob_free_aircraft': 0.1
    , 'prob_free_periods': 0.5
    , 'cooling': 0.999
    , 'debug': False
    , 'max_iters': 99999999
    , 'prob_delete_maint': 0.5
    , 'log_output': ['file', 'console']
    , 'log_handler': None
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
    , 'solver_add_opts': params_cplex
    , 'fix_start': False
    # TODO: this should go and be changed by capacity in each maintenance
    , 'default_type2_capacity': 66
    # simulation params:
    , 'simulation': {
        'num_resources': 15  # this depends on the number of tasks actually
        , 'num_parallel_tasks': 0
        , 'maint_duration': 6
        , 'max_used_time': 1000
        , 'max_elapsed_time': 60  # max time without maintenance
        , 'elapsed_time_size': 30  # size of window to do next maintenance
        , 'min_usage_period': 15 # minimum consumption per period
        , 'perc_capacity': 0.3
        , 'min_avail_percent': 0  # min percentage of available aircraft per type
        , 'min_avail_value': 0  # min num of available aircraft per type
        , 'min_hours_perc': 0.05  # min percentage of maximum possible hours of fleet type
        , 'seed': 47
        # The following are fixed options, not arguments for the scenario:
        , 't_min_assign': [2, 3, 6]  # minimum assignment time for tasks
        , 'initial_unbalance': (-3, 3)
        , 't_required_hours': (30, 50, 80) # triangular distribution params
        , 't_num_resource': (2, 5)
        , 't_duration': (6, 12)
        , 'perc_in_maint': 0.07
        , 'maintenances': {
            # type=1 is unite based capacity.
            # type=2 is time based capacity.
            'M': {
                'duration_periods': 4
                , 'capacity_usage': 1
                , 'max_used_time': 1000
                , 'max_elapsed_time': 60
                , 'elapsed_time_size': 3
                , 'used_time_size': 1000
                , 'type': '1'
                , 'depends_on': []
                , 'affects': []  # this gets filled during initialization
                , 'priority': 0  # the least, the sooner we assign
            }
            ,'VG': {
                'duration_periods': 1
                ,'capacity_usage': 3
                , 'max_used_time': None
                , 'max_elapsed_time': 8
                , 'elapsed_time_size': 3
                , 'used_time_size': None
                , 'type': '2'
                , 'depends_on': ['M']
                , 'affects': []
                , 'priority': 10
            }
            ,'VI': {
                'duration_periods': 1
                ,'capacity_usage': 6
                , 'max_used_time': None
                , 'max_elapsed_time': 17
                , 'elapsed_time_size': 3
                , 'used_time_size': None
                , 'type': '2'
                , 'depends_on': ['M']
                , 'affects': []
                , 'priority': 5
            }
            ,'VS': {
                'duration_periods': 1
                , 'capacity_usage': 4
                , 'max_used_time': 600
                , 'max_elapsed_time': None
                , 'elapsed_time_size': None
                , 'used_time_size': 200
                , 'type': '2'
                , 'depends_on': ['M']
                , 'affects': []
                , 'priority': 2
            }
        }
    }
}

# TODO: grouping improves capacity

# white_list = ['O1', 'O5']
# black_list = []
# black_list = ['O10', 'O8', 'O6']
# black_list = ['O8']  # this task has less candidates than what it asks.


import os
import datetime as dt

path_root = '/home/pchtsp/Documents/projects/'
path_results = path_root + "OPTIMA_documents/results/"
path_project = path_root + "OPTIMA/"

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

OPTIONS = {
    'timeLimit': 3600*3  # seconds
    , 'gap': 0
    , 'solver': "CPO"  # HEUR, CPO, CHOCO, CPLEX, GUROBI, CBC
    , 'print': True
    , 'integer': True
    , 'black_list': ['O8', 'O10']
    , 'white_list': []
    , 'start_pos': 0
    , 'end_pos': 50
    , 'path': os.path.join(
        PATHS['experiments'],
        dt.datetime.now().strftime("%Y%m%d%H%M")
    ) + '/'
}

# white_list = ['O1', 'O5']
# black_list = []
# black_list = ['O10', 'O8', 'O6']
# black_list = ['O8']  # this task has less candidates than what it asks.


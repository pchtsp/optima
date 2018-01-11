from ggplot import *
import pprint as pp
import package.aux as aux
import pandas as pd
import package.instance as inst
import package.tests as exp
import os
import numpy as np
import random


path_abs = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/"
path_img = "/home/pchtsp/Documents/projects/OPTIMA/img/"
if __name__ == "__main__":

    instance = inst.Instance()
    paths = [path_abs + 'weights/' + str(id) for id in range(11)]
    experiment = exp.Experiment.from_dir(paths[0])
    # here we get some data inside

    ################################################
    # Task table
    ################################################

    cols = ['consumption', 'num_resource', 'candidates']

    cols_rename = {'consumption': 'usage (h)', 'num_resource': '# resource', 'candidates': '# candidates'}

    data = pd.DataFrame.from_dict(instance.get_tasks(), orient="index")[cols]
    data.candidates = data.candidates.apply(len)
    data.rename(index=str, columns = cols_rename, inplace=True)

    print(data.to_latex())

    ################################################
    # Histogram
    ################################################

    pp.pprint(instance.get_param())

    data = pd.DataFrame.from_dict(instance.get_resources('initial_used'), orient='index').\
        rename(columns={0: 'initial_used'})

    plot = ggplot(aes(x='initial_used'), data=data) + geom_histogram() + \
           xlab("Initial Remaining Usage Time (hours)")
    plot.save(path_img + 'initial_used.png')

    ################################################
    # Instances table
    ################################################

    paths = os.listdir(path_abs)
    exps = exp.list_experiments(path_abs)
    pp.pprint(exps)
    # input: num missions, num periods, variables, constraints, assignments, timeLimit
    input_cols = ['periods', 'assignments', 'tasks', 'vars', 'cons', 'nonzeros']
    cols_rename = {'periods': '$|\mathcal{T}|$', 'tasks': '$|\mathcal{J}|$', 'assignments': 'assign',
                   'timeLimit': 'time (s)', 'index': 'id'}
    table = pd.DataFrame.from_dict(exps, orient="index")
    table = table[np.all((table.model == 'no_states',
                  table.gap == 0,
                  table.timeLimit >= 500),
                 axis=0)] \
        [input_cols].reset_index().rename(columns=cols_rename)
    table.id = table.id.str.slice(8)
    print(table.to_latex(escape=False, bold_rows=True, index=False, float_format='%.0f'))

    ################################################
    # Results table
    ################################################

    paths = os.listdir(path_abs)
    exps = exp.list_experiments(path_abs)
    pp.pprint(exps)
    # output: number of cuts, initial relaxation, relaxation after cuts, end relaxation, cuts' time.
    # variable number before and after cuts
    # input_cols = ['timeLimit']
    cols_rename = {'timeLimit': 'time (s)', 'index': 'id', 'objective_out': 'OF',
                   'gap_out': 'gap (%)', 'bound_out': 'bound'}
    table = pd.DataFrame.from_dict(exps, orient="index")
    table = table[np.all((table.model == 'no_states',
                  table.gap == 0,
                  table.timeLimit >= 500),
                 axis=0)].reset_index()\
        [list(cols_rename.keys())].rename(columns=cols_rename)
    table.id = table.id.str.slice(8)
    table = table.iloc[:, [1, 0, 2, 3, 4]]
    print(table.to_latex(bold_rows=True, index=False, float_format='%.1f'))

    ################################################
    # Multiobjective
    ################################################
    # paths = os.listdir()
    path_comp = path_abs + "weights/"
    exps = exp.list_experiments(path_comp)
    pp.pprint(exps)

    experiments = {path: exp.Experiment.from_dir(path_comp + path)
                   for path in os.listdir(path_comp)}

    maint_weight = {k: v.instance.get_param('maint_weight') for k, v in experiments.items()}
    unavailable = {k: max(v.solution.get_unavailable().values()) for k, v in experiments.items()}
    maint = {k: max(v.solution.get_in_maintenance().values()) for k, v in experiments.items()}
    gaps = aux.get_property_from_dic(exps, "gap_out")
    obj = aux.get_property_from_dic(exps, "objective_out")
    obj2 = {k: v.get_objective_function() for k, v in experiments.items()}
    checks = {k: v.check_solution() for k, v in experiments.items()}

    data_dic = \
        {k:
             {'maint_weight': maint_weight[k],
              'unavailable': unavailable[k],
              'maint': maint[k],
              'gap': gaps[k],
              'unavail_weight': 1 - maint_weight[k]}
         for k in maint_weight
         }

    cols_rename = {'maint_weight': 'w1', 'unavail_weight': 'w2', 'index': 'id', 'gap': 'gap (%)'}
    data = pd.DataFrame.from_dict(data_dic, orient='index').reset_index().rename(columns=cols_rename)
    data.id = data.id.apply(lambda x: 'W' + x.zfill(2))
    print(data.to_latex(bold_rows=True, index=False))

    data.maint = data.maint.apply(lambda x: x + (random.random() - 0.5)/3)
    data.unavailable = data.unavailable.apply(lambda x: x + (random.random() - 0.5)/3)

    plot = \
        ggplot(data, aes(x='maint', y='unavailable', label='id')) + \
        geom_point() +\
        geom_text(hjust=0.15, vjust=0.1) + \
        xlab('max resources in maintenance') + \
        ylab('max resources unavailable')

    plot.save(path_img + 'multiobjective.png')


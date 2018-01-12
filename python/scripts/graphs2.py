from ggplot import *
import pprint as pp
import package.aux as aux
import pandas as pd
import package.instance as inst
import package.tests as exp
import os
import numpy as np
import random
import package.data_input as di

path_abs = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/"
path_img = "/home/pchtsp/Documents/projects/OPTIMA/img/"
path_latex = "/home/pchtsp/Documents/projects/OPTIMA/latex/"
if __name__ == "__main__":

    # here we get some data inside

    ################################################
    # Task table
    ################################################
    model_data = di.get_model_data()
    historic_data = di.generate_solution_from_source()
    model_data = di.combine_data_states(model_data, historic_data)
    instance = inst.Instance(model_data)

    cols = ['consumption', 'num_resource', 'candidates']

    cols_rename = {'consumption': 'usage (h)', 'num_resource': '# resource',
                   'candidates': '# candidates', 'index': 'task'}

    data = pd.DataFrame.from_dict(instance.get_tasks(), orient="index")[cols]
    data.candidates = data.candidates.apply(len)
    data = data.reset_index().rename(index=str, columns = cols_rename)
    data['order'] = data.task.str.slice(1).apply(int)
    data = data.sort_values("order").iloc[:, :4]

    latex = data.to_latex(bold_rows=True, index=False, float_format='%.0f')
    with open(path_latex + 'MOSIM2018/tables/tasks.tex', 'w') as f:
        f.write(latex)


    ################################################
    # Histogram
    ################################################
    model_data = di.get_model_data()
    historic_data = di.generate_solution_from_source()
    model_data = di.combine_data_states(model_data, historic_data)
    instance = inst.Instance(model_data)

    # pp.pprint(instance.get_param())

    data = pd.DataFrame.from_dict(instance.get_resources('initial_used'), orient='index').\
        rename(columns={0: 'initial_used'})

    plot = ggplot(aes(x='initial_used'), data=data) + geom_histogram() + \
           theme(axis_text=element_text(size=15)) + \
           xlab("Initial Remaining Usage Time (hours)")
    plot.save(path_img + 'initial_used.png')

    ################################################
    # Instances table
    ################################################

    paths = os.listdir(path_abs)
    exps = exp.list_experiments(path_abs)
    # pp.pprint(exps)
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
    latex = table.to_latex(escape=False, bold_rows=True, index=False, float_format='%.0f')
    with open(path_latex + 'MOSIM2018/tables/instance.tex', 'w') as f:
        f.write(latex)

    ################################################
    # Results table
    ################################################

    paths = os.listdir(path_abs)
    exps = exp.list_experiments(path_abs)
    # pp.pprint(exps)
    # output: number of cuts, initial relaxation, relaxation after cuts, end relaxation, cuts' time.
    # variable number before and after cuts
    # input_cols = ['timeLimit']
    cols_rename = {
        'timeLimit': 'time (s)', 'index': 'id', 'objective_out': 'objective',
                   'gap_out': 'gap (%)', 'bound_out': 'bound'
                   }
    cols_rename2 = {
        'index': 'id', 'after_cuts': 'bound cuts',
        'first_relaxed': 'root', '# cuts': 'cuts (#)',
        'cutsTime': 'cuts (s)', 'bound_out': 'bound'}

    table = pd.DataFrame.from_dict(exps, orient="index")
    table = table[np.all((table.model == 'no_states',
                  table.gap == 0,
                  table.timeLimit >= 500),
                 axis=0)].reset_index()
    table['# cuts'] = table.cuts.apply(lambda x: sum(x.values()))
    table['most cuts'] = table.cuts.apply(
        lambda x: ["{}: {}".format(k, v)
                   for k, v in x.items()
                   if v == max(x.values())
                   ][0])
    table['index'] = table['index'].str.slice(8)

    table1 = table.rename(columns=cols_rename2)[
        ['id', 'root', 'bound cuts',
         'bound', 'cuts (#)', 'cuts (s)']]
    latex = table1.to_latex(bold_rows=True, index=False, float_format='%.1f')
    with open(path_latex + 'MOSIM2018/tables/results.tex', 'w') as f:
        f.write(latex)

    table2 = table.rename(columns=cols_rename)[
        ['id', 'objective', 'gap (%)', 'time (s)', 'bound']]
    latex = table2.to_latex(bold_rows=True, index=False, float_format='%.1f')
    with open(path_latex + 'MOSIM2018/tables/summary.tex', 'w') as f:
        f.write(latex)


    ################################################
    # Multiobjective
    ################################################
    # paths = os.listdir()
    path_comp = path_abs + "weights/"
    exps = exp.list_experiments(path_comp)
    # pp.pprint(exps)

    experiments = {path: exp.Experiment.from_dir(path_comp + path)
                   for path in os.listdir(path_comp)}

    maint_weight = {k: v.instance.get_param('maint_weight') for k, v in experiments.items()}
    unavailable = {k: max(v.solution.get_unavailable().values()) for k, v in experiments.items()}
    maint = {k: max(v.solution.get_in_maintenance().values()) for k, v in experiments.items()}
    gaps = aux.get_property_from_dic(exps, "gap_out")
    obj = aux.get_property_from_dic(exps, "objective_out")
    obj2 = {k: v.get_objective_function() for k, v in experiments.items()}
    checks = {k: v.check_solution() for k, v in experiments.items()}
    aux.get_property_from_dic(exps, "timeLimit")

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
    data = data.sort_values("id").iloc[:, [0, 3, 2, 1, 4, 5]]
    latex = data.to_latex(bold_rows=True, index=False)
    with open(path_latex + 'MOSIM2018/tables/multiobj.tex', 'w') as f:
        f.write(latex)

    # data.maint = data.maint.apply(lambda x: x + (random.random() - 0.5)/3)
    # data.unavailable = data.unavailable.apply(lambda x: x + (random.random() - 0.5)/3)

    data_graph = data.groupby(['maint', 'unavailable'])['id'].agg(lambda x: ', '.join(x))
    data_graph = data_graph.reset_index()
    condition = np.logical_or(data_graph.id == 'W01', data_graph.id == 'W05, W06, W09')
    data_graph['pareto optimal'] = np.where(condition, 'yes', 'no')

    plot = \
        ggplot(data_graph, aes(x='maint', y='unavailable', label='id')) + \
        geom_point(size=50) +\
        geom_text(hjust=0.15, vjust=0) + \
        theme(axis_text=element_text(size=15), legend_position="none") + \
        xlab('max resources in maintenance') + \
        ylab('max resources unavailable')

    # plot.add_legend()
    # theme(axis_text=element_text(size=15),
    #       axis_title_x=element_text(size=20)) + \


    plot.save(path_img + 'multiobjective.png')


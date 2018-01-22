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


# here we get some data inside
def task_table():
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
    data = data[data.task != "O8"]
    data['order'] = data.task.str.slice(1).apply(int)
    data = data.sort_values("order").drop(['order'], axis=1).reset_index(drop=True)
    data['order'] = data.index + 1
    data['temporal'] = (data.order <= 4)*1
    data['task'] = data['order'].apply(lambda x: "O" + str(x))
    data = data.drop(['order'], axis=1)

    latex = data.to_latex(bold_rows=True, index=False, float_format='%.0f')
    with open(path_latex + 'MOSIM2018/tables/tasks.tex', 'w') as f:
        f.write(latex)


def remaining_graph():
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
           xlab(element_text("Initial Remaining Usage Time (hours)", size=15))+ \
           ylab(element_text("Number of resources", size=15))
    plot.save(path_img + 'initial_used.png')


def get_results_table(path_abs):
    # paths = os.listdir(path_abs)
    exps = exp.list_experiments(path_abs)
    table = pd.DataFrame.from_dict(exps, orient="index")
    table = table[np.all((table.model == 'no_states',
                          table.gap == 0,
                          table.timeLimit >= 500),
                         axis=0)].sort_values(['tasks', 'periods']).reset_index()
    table['date'] = table['index']
    table['ref'] = table['index'].str.slice(8)
    table['index'] = ['I\_' + str(num) for num in table.index]
    return table


def results_table():
    ################################################
    # Instances and Results table
    ################################################
    table = get_results_table(path_abs)

    # pp.pprint(exps)
    # input: num missions, num periods, variables, constraints, assignments, timeLimit
    input_cols = ['index', 'tasks', 'periods', 'assignments', 'vars', 'cons', 'nonzeros']
    cols_rename = {'periods': '$|\mathcal{T}|$', 'tasks': '$|\mathcal{J}|$', 'assignments': 'assign',
                   'timeLimit': 'time (s)', 'index': 'id'}

    table_in = table[input_cols].rename(columns=cols_rename)
    latex = table_in.to_latex(escape=False, bold_rows=True, index=False, float_format='%.0f')
    with open(path_latex + 'MOSIM2018/tables/instance.tex', 'w') as f:
        f.write(latex)

    # output: number of cuts, initial relaxation, relaxation after cuts, end relaxation, cuts' time.
    # variable number before and after cuts
    # input_cols = ['timeLimit']
    cols_rename = {
        'time_out': 'time (s)', 'index': 'id', 'objective_out': 'objective',
                   'gap_out': 'gap (\%)', 'bound_out': 'bound'
                   }
    cols_rename2 = {
        'index': 'id', 'after_cuts': 'b. cuts',
        'first_relaxed': 'root', '# cuts': 'cuts (\#)',
        'cutsTime': 'cuts (s)', 'bound_out': 'bound'}

    cols_rename3 = {
        'index': 'id', 'sol_after_cuts': 'sol. cuts',
        'first_solution': 'first', 'objective_out': 'last'
    }

    # table['most cuts'] = table.cuts.apply(
    #     lambda x: ["{}: {}".format(k, v)
    #                for k, v in x.items()
    #                if v == max(x.values())
    #                ][0])

    table1 = table.rename(columns=cols_rename)[
        ['id', 'objective', 'gap (\%)', 'time (s)', 'bound']]
    latex = table1.to_latex(escape=False, bold_rows=True, index=False, float_format='%.1f')
    with open(path_latex + 'MOSIM2018/tables/summary.tex', 'w') as f:
        f.write(latex)

    table2 = table
    table2['# cuts'] = table2.cuts.apply(lambda x: sum(x.values()))
    table2 = table.rename(columns=cols_rename2)[
        ['id', 'root', 'b. cuts',
         'bound', 'cuts (\#)', 'cuts (s)']]
    latex = table2.to_latex(escape=False, bold_rows=True, index=False, float_format='%.1f')
    with open(path_latex + 'MOSIM2018/tables/results.tex', 'w') as f:
        f.write(latex)

    table3 = table.rename(columns=cols_rename3)[
        ['id', 'first', 'sol. cuts',
         'last']]
    latex = table3.to_latex(escape=False, bold_rows=True, index=False, float_format='%.1f')
    with open(path_latex + 'MOSIM2018/tables/results2.tex', 'w') as f:
        f.write(latex)


def multi_get_info(path_comp):
    exps = exp.list_experiments(path_comp)
    # pp.pprint(exps)

    experiments = {path: exp.Experiment.from_dir(path_comp + path)
                   for path in os.listdir(path_comp)
                   if exp.Experiment.from_dir(path_comp + path) is not None}

    maint_weight = {k: v.instance.get_param('maint_weight') for k, v in experiments.items()}
    unavailable = {k: max(v.solution.get_unavailable().values()) for k, v in experiments.items()}
    maint = {k: max(v.solution.get_in_maintenance().values()) for k, v in experiments.items()}
    gaps = aux.get_property_from_dic(exps, "gap_out")
    # obj = aux.get_property_from_dic(exps, "objective_out")
    # obj2 = {k: v.get_objective_function() for k, v in experiments.items()}
    # checks = {k: v.check_solution() for k, v in experiments.items()}
    # aux.get_property_from_dic(exps, "timeLimit")

    data_dic = \
        {k:
             {'maint_weight': maint_weight[k],
              'unavailable': unavailable[k],
              'maint': maint[k],
              'gap': gaps[k],
              'unavail_weight': 1 - maint_weight[k]}
         for k in maint_weight
         }

    return data_dic


def get_pareto_points(x, y):
    # x and y are dictionaries
    pareto_points = []
    x_sorted = sorted(x, key=x.get)
    min_y = 999
    for x_key in x_sorted:
        y_value = y[x_key]
        if y_value < min_y:
            # print(x[x_key], y_value)
            min_y = y_value
            pareto_points.append(x_key)

    return pareto_points


def multiobjective_table():
    ################################################
    # Multiobjective
    ################################################
    path_comp = path_abs + "weights2/"
    data_dic = multi_get_info(path_comp)

    cols_rename = {'maint_weight': '$W_1$', 'unavail_weight': '$W_2$', 'index': 'exp', 'gap': 'gap (\%)'}
    data = pd.DataFrame.from_dict(data_dic, orient='index').reset_index().rename(columns=cols_rename)
    data.exp = data.exp.apply(lambda x: 'W' + x.zfill(2))
    data = data.sort_values("exp")[['exp', 'maint', 'unavailable', '$W_1$', '$W_2$']]
    latex = data.to_latex(bold_rows=True, index=False, escape=False)
    with open(path_latex + 'MOSIM2018/tables/multiobj.tex', 'w') as f:
        f.write(latex)

    data_graph = data.groupby(['maint', 'unavailable'])['exp'].agg(lambda x: ', '.join(x))
    data_graph = data_graph.reset_index()
    condition = np.any((data_graph.exp == 'W00, W01, W02',
                        data_graph.exp == 'W03, W04',
                        data_graph.exp == 'W05, W06, W08'
                       ), axis=0)
    data_graph['Pareto optimal'] = np.where(condition, 'yes', 'no')

    plot = \
        ggplot(data_graph, aes(x='maint', y='unavailable', label='exp', color='Pareto optimal')) + \
        geom_point(size=50) +\
        geom_text(hjust=0.15, vjust=0) + \
        theme(axis_text=element_text(size=15)) + \
        xlab(element_text('max resources in maintenance', size=15)) + \
        ylab(element_text('max resources unavailable', size=15))

    plot.save(path_img + 'multiobjective.png')


def multi_multiobjective_table():
    path_exp = path_abs + 'weights_all/'
    path_comps = {path: path_exp + path + '/' for path in os.listdir(path_exp)}
    data_dic = {}
    for k, v in path_comps.items():
        data_dic[k] = multi_get_info(v)

    pareto_dict = {key: {} for key in data_dic}
    # pareto_p2 = {}
    for key, value in data_dic.items():
        # key = '201801141646'
        x = aux.get_property_from_dic(value, 'maint')
        y = aux.get_property_from_dic(value, 'unavailable')
        points1 = get_pareto_points(x, y)
        points2 = get_pareto_points(y, x)
        points = np.intersect1d(points1, points2).tolist()
        pareto_dict[key]['points'] = points
        point = points[0]
        pareto_dict[key]['First'] = (x[point], y[point])
        point = points[-1]
        pareto_dict[key]['Last'] = (x[point], y[point])

    table_ref = get_results_table(path_abs)
    table_ref = table_ref[['date', 'index']].set_index('date')
    table = pd.DataFrame.from_dict(pareto_dict, orient='index')
    table['\# points'] = table.points.apply(len)
    table = pd.merge(table_ref, table, left_index=True, right_index=True).\
        reset_index(drop=True).rename(columns={'index': 'id'})
    data = table[['id', '\# points', 'First', 'Last']]
    latex = data.to_latex(bold_rows=True, index=False, escape=False)
    with open(path_latex + 'MOSIM2018/tables/multi-multiobj.tex', 'w') as f:
        f.write(latex)


def progress_graph():
    ################################################
    # Progress
    ################################################
    path = path_abs + '201801131817/results.log'
    table = di.get_log_info_cplex_progress(path)

    plot = \
        ggplot(table, aes(x='it', y='obj')) + \
        geom_line(color='blue') +\
        geom_line(aes(y='relax'), color='red') +\
        theme(axis_title_x=element_text('Iterations', size=15),
              axis_title_y=element_text('Objective and relaxation', size=15))+\
        theme(axis_text=element_text(size=15))

    plot.save(path_img + 'progress.png')


def maintenances_graph(maint=True):
    ################################################
    # Maintenances graph
    ################################################
    path = path_abs + '201801131817'
    experiment = exp.Experiment.from_dir(path)
    data = experiment.solution.get_in_maintenance()
    name = 'maintenances'
    if not maint:
        data = experiment.solution.get_unavailable()
        name = 'unavailables'
    table = pd.DataFrame.from_dict(data, orient="index").\
        reset_index().rename(columns={'index': 'month', 0: 'maint'})
    # table.month = pd.to_datetime(table.month.apply(lambda x:
    #                                 aux.month_to_arrow(x).naive))
    # table.month = table.month.apply(lambda x: x + "-01")
    # table.month = table.month.apply(lambda x: aux.month_to_arrow(x).datetime)
    # pd.DataFrame.from_dict(unavailable, orient="index")

    # print(ggplot(meat, aes('date', 'beef')) + geom_line() + scale_x_date(breaks=date_breaks('10 years'),
    #                                                                      labels=date_format('%B %-d, %Y'),
    #                                                                      limits=[aux.month_to_arrow('1940-01').naive,
    #                                                                              aux.month_to_arrow('2011-01').naive]))

    # p = ggplot(table, aes(x='month', y='maint')) + \
    # scale_x_date(labels="%Y-%m", breaks='1 months', limits=(aux.month_to_arrow('2017-01').naive,
    #                                                            aux.month_to_arrow('2021-01').naive)) + \
    first = table.month.values.min()
    last = table.month.values.max()
    multiple = 6
    total = len(aux.get_months(first, last))
    numticks = total // 4

    ticks = [aux.shift_month(first, n*multiple) for n in range(numticks-2)]
    labels = ticks
    limits = [first, ticks[-1]]
    p = \
        ggplot(table, aes(x='month', y='maint')) + geom_line() + \
        scale_x_continuous(breaks=ticks, labels=labels, limits=limits) +\
        theme(axis_title_y=element_text('Number of {}'.format(name), size=15),
              axis_title_x=element_text('Periods (months)', size=15))
    # geom_point() + \
    p.save(path_img + 'num-{}.png'.format(name))
        # scale_x_date(labels='%Y')

    # ggplot(meat, aes('date', 'beef')) + \
    # geom_line() + \
    # scale_x_date(labels="%Y-%m-%d")


def tests():
    ################################################
    # Tests
    ################################################
    path_weights = path_abs + "weights3/"
    paths = os.listdir(path_abs)
    small_tests = \
    ['201801141705', '201801141331', '201801141646', '201801131813',
     '201801091304', '201801091338', '201801091342', '201801091346',
     '201801091412', '201801141607', '201712142345', '201801102259']

    weights_wrong = [0, 2, 3, 4, 7, 8, 10]

    comp = {}
    for experiment in range(11):
        # path = path_abs + experiment
        path = path_weights + str(experiment)
        comp[experiment] = exp.Experiment.from_dir(path).check_solution()

    pp.pprint([k for k, v in comp.items() if len(v) > 0])
    weights2_wrong = [4, 7, 9]

    no_feasible = \
    ['201712142345', '201801091338', '201801131813', '201801091304', '201801091412', '201801091346']

    experiment = "201801091346"
    experiment = "201801182313"
    experiment = "201801091412"
    experiment = "weights/7"
    experiment = "weights2/7"
    experiment = "201801190010"

    path = path_abs + experiment
    test = exp.Experiment.from_dir(path)
    checks = test.check_solution()
    pp.pprint(checks)
    # comp[experiment]
    schedule = test.solution.get_schedule()

    tasks = test.solution.get_tasks()
    # tasks
    # schedule[]
    exps = exp.list_experiments(path_abs)
    exps = exp.list_experiments(path_weights)
    pp.pprint(exps["7"])


if __name__ == "__main__":

    # maintenances_graph()
    # maintenances_graph(maint=False)
    multi_multiobjective_table()
    # pp.pprint(d)
    # for x_key in x:
    #     print(x[x_key], y[x_key])

    # multiobjective_table()
    # pp.pprint(pareto_p2)
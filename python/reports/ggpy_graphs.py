from ggplot import *
import data.data_input as di
from package.params import PATHS
import package.instance as inst
import pandas as pd
import orloge as log
import numpy as np
import package.auxiliar as aux
import package.experiment as exp
import reports.reports as rep


path_root = PATHS['root']
path_abs = PATHS['experiments']
path_img = PATHS['img']
path_latex = PATHS['latex']
path_results = PATHS['results']


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
           theme(axis_text=element_text(size=20)) + \
           xlab(element_text("Initial Remaining Usage Time (hours)",
                             size=20,
                             vjust=-0.05))+ \
           ylab(element_text("Number of resources",
                             size=20,
                             vjust=0.15))
    # print(plot)
    plot.save(path_img + 'initial_used.png')

def multi_objective_graph():

    path_comp = path_abs + "weights3/"
    data_dic = rep.multi_get_info(path_comp)

    cols_rename = {'maint_weight': '$W_1$', 'unavail_weight': '$W_2$', 'index': 'exp', 'gap': 'gap (\%)'}
    data = pd.DataFrame.from_dict(data_dic, orient='index').reset_index().rename(columns=cols_rename)
    data.exp = data.exp.apply(lambda x: 'W' + x.zfill(2))
    data = data.sort_values("exp")[['exp', 'maint', 'unavailable', '$W_1$', '$W_2$']]

    data_graph = data.groupby(['maint', 'unavailable'])['exp'].agg(lambda x: ', '.join(x))
    data_graph = data_graph.reset_index()
    condition = np.any((data_graph.exp == 'W00, W01, W02',
                        data_graph.exp == 'W03, W04',
                        data_graph.exp == 'W05, W06, W08'
                       ), axis=0)
    data_graph['Pareto optimal'] = np.where(condition, 'yes', 'no')

    plot = \
        ggplot(data_graph, aes(x='maint', y='unavailable', label='exp')) + \
        geom_point(size=70) +\
        geom_text(hjust=-0, vjust=0.05, size=20) + \
        theme(axis_text=element_text(size=20)) + \
        xlab(element_text('Max resources in maintenance', size=20, vjust=-0.05)) + \
        ylab(element_text('Max resources unavailable', size=20, vjust=0.15)) + \
        xlim(low=14.5, high=21)

    plot.save(path_img + 'multiobjective.png')


def progress_graph():
    ################################################
    # Progress
    ################################################
    # TODO: correct this to new logging library
    path = path_abs + '201801131817/results.log'
    result_log = log.LogFile(path)
    table = result_log.get_progress_cplex()
    table.rename(columns={'ItCnt': 'it', 'Objective': 'relax', 'BestInteger': 'obj'},
                 inplace=True)
    table = table[table.it.str.match(r'\s*\d')][['it', 'relax', 'obj']]
    table.it = table.it.astype(int)
    table.obj = table.obj.str.replace(r'^\s+$', '0')
    table.relax = table.relax.str.replace(r'^\s+$', '0')
    table.obj = table.obj.astype(float)
    table.relax = table.relax.astype(float)
    table.it = round(table.it / 1000)

    plot = \
        ggplot(table, aes(x='it', y='obj')) + \
        geom_line(color='blue') +\
        geom_line(aes(y='relax'), color='red') +\
        theme(axis_title_x=element_text('Iterations (thousands)', size=20, vjust=-0.05),
              axis_title_y=element_text(
                  'Objective and relaxation',
                  size=20,
                  vjust=0.15),
              axis_text=element_text(size=20))
    # print(plot)
    plot.save(path_img + 'progress.png')


def maintenances_graph(maint=True):
    ################################################
    # Maintenances graph
    ################################################
    path = path_abs + '201802061201'
    experiment = exp.Experiment.from_dir(path)
    data = experiment.solution.get_in_some_maintenance()
    name = 'maintenances'
    if not maint:
        data = experiment.solution.get_unavailable()
        name = 'unavailables'
    table = pd.DataFrame.from_dict(data, orient="index").\
        reset_index().rename(columns={'index': 'month', 0: 'maint'}).sort_values('month')

    first = table.month.values.min()
    last = table.month.values.max()
    multiple = 6
    total = len(aux.get_months(first, last))
    numticks = total // 4

    ticks = [aux.shift_month(first, n*multiple) for n in range(numticks-2)]
    labels = ticks
    limits = [first, ticks[-1]]
    p = \
        ggplot(table, aes(x='month', y='maint')) + geom_step() + \
        theme(axis_text=element_text(size=20)) +\
        scale_x_continuous(breaks=ticks, labels=labels, limits=limits) +\
        theme(axis_title_y=element_text('Number of {}'.format(name), size=20, vjust=0.15),
              axis_title_x=element_text('Periods (months)', size=20, vjust=-0.02)) + theme_bw()

    # print(p)
    # geom_point() + \
    p.save(path_img + 'num-{}.png'.format(name))
        # scale_x_date(labels='%Y')

    # ggplot(meat, aes('date', 'beef')) + \
    # geom_line() + \
    # scale_x_date(labels="%Y-%m-%d")


def boxplot_instances(table, column='time_out'):
    # table = table >> dp.filter_by(~X.inf)
    return ggplot(aes(x='code', y=column), data=table) + geom_boxplot() + \
    xlab(element_text("Scenario", size=20, vjust=-0.05)) + \
    ylab(element_text("Solving time (in seconds)", size=20, vjust=0.15))

from ggplot import *
import pprint as pp
import package.auxiliar as aux
import pandas as pd
import package.instance as inst
import package.experiment as exp
import os
import numpy as np
import random
import package.data_input as di
import tabulate
import orloge as log
import re
import package.heuristics as heur
from package.params import PATHS
import package.superdict as sd
import scripts.names as na

import dfply as dp
from dfply import X
# from dfply import left_join, rename, X, select, mutate, \
#     group_by, filter_by, summarize, distinct, make_symbolic, first, n


path_root = PATHS['root']
path_abs = PATHS['experiments']
path_img = PATHS['img']
path_latex = PATHS['latex']
path_results = PATHS['results']

# here we get some data inside
def task_table():
    ################################################
    # Task table
    ################################################
    model_data = di.get_model_data()
    historic_data = di.generate_solution_from_source()
    model_data = di.combine_data_states(model_data, historic_data)
    instance = inst.Instance(model_data)

    for k, v in instance.get_task_candidates().items():
        instance.data['tasks'][k]['candidates'] = v

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
           theme(axis_text=element_text(size=20)) + \
           xlab(element_text("Initial Remaining Usage Time (hours)",
                             size=20,
                             vjust=-0.05))+ \
           ylab(element_text("Number of resources",
                             size=20,
                             vjust=0.15))
    # print(plot)
    plot.save(path_img + 'initial_used.png')


def get_results_table(path_abs, exp_list=None, **kwargs):
    exps = exp.list_experiments(path_abs, exp_list=exp_list, **kwargs)
    table = pd.DataFrame.from_dict(exps, orient="index")

    if exp_list is not None:
        table = table.loc[exp_list]
    if not len(table):
        return table
    # table = table.sort_values(['tasks', 'periods']).reset_index()
    table.reset_index(inplace=True)
    table['date'] = table['index']
    table['ref'] = table['index'].str.slice(8)
    table['index'] = ['I\_' + str(num) for num in table.index]
    return table


def results_table(dir_origin=path_abs, dir_dest=path_latex + 'MOSIM2018/tables/', exp_list=None):
    ################################################
    # Instances and Results table
    ################################################
    # Changed the logic to a fix set of results.
    # This makes it possible to recover the original results
    # if exp_list is None:
    if exp_list is None:
        exp_list = [
            '201801141705', '201801141331', '201801141334', '201801141706',
            '201801141646', '201801131813', '201801102127', '201801131817',
            '201801141607', '201801102259'
        ]
    table = get_results_table(dir_origin, exp_list=exp_list)

    # pp.pprint(exps)
    # input: num missions, num periods, variables, constraints, assignments, timeLimit
    input_cols = ['index', 'tasks', 'periods', 'assignments', 'vars', 'cons', 'nonzeros']
    cols_rename = {'periods': '$|\mathcal{T}|$', 'tasks': '$|\mathcal{J}|$', 'assignments': 'assign',
                   'timeLimit': 'time (s)', 'index': 'id'}

    table_in = table[input_cols].rename(columns=cols_rename)
    latex = table_in.to_latex(escape=False, bold_rows=True, index=False, float_format='%.0f')
    destination = dir_dest + 'instance.tex'
    with open(destination, 'w') as f:
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
    latex = table1.to_latex(escape=False, bold_rows=True, index=False, float_format='%.2f')
    destination = dir_dest + 'summary.tex'
    with open(destination, 'w') as f:
        f.write(latex)

    table2 = table
    table2['# cuts'] = table2.cuts.apply(lambda x: sum(x.values()))
    table2 = table.rename(columns=cols_rename2)[
        ['id', 'root', 'b. cuts',
         'bound', 'cuts (\#)', 'cuts (s)']]
    latex = table2.to_latex(escape=False, bold_rows=True, index=False, float_format='%.1f')
    destination = dir_dest + 'results.tex'
    with open(destination, 'w') as f:
        f.write(latex)

    table3 = table.rename(columns=cols_rename3)[
        ['id', 'first', 'sol. cuts',
         'last']]
    latex = table3.to_latex(escape=False, bold_rows=True, index=False, float_format='%.1f')
    # we replaces the possible NaN values for a - symbol
    latex = re.sub(r'NaN', '-', latex)
    destination = dir_dest + 'results2.tex'
    with open(destination, 'w') as f:
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
    path_comp = path_abs + "weights3/"
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
        ggplot(data_graph, aes(x='maint', y='unavailable', label='exp')) + \
        geom_point(size=70) +\
        geom_text(hjust=-0, vjust=0.05, size=20) + \
        theme(axis_text=element_text(size=20)) + \
        xlab(element_text('Max resources in maintenance', size=20, vjust=-0.05)) + \
        ylab(element_text('Max resources unavailable', size=20, vjust=0.15)) + \
        xlim(low=14.5, high=21)

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
    # TODO: correct this to new logging interface
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
    data = experiment.solution.get_in_maintenance()
    name = 'maintenances'
    if not maint:
        data = experiment.solution.get_unavailable()
        name = 'unavailables'
    table = pd.DataFrame.from_dict(data, orient="index").\
        reset_index().rename(columns={'index': 'month', 0: 'maint'}).sort_values('month')
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


def solvers_comp():


    # exps = sd.SuperDict(exps)
    # solver = exps.get_property('solver')
    # solver.clean('CPO')
    # exps_list = [k for k, v in solver.items()]
    options_e = exp.list_options(path_abs)
    for e, v in options_e.items():
        if 'end_pos' not in v:
            continue
        v['tasks'] = 12 - len(v.get('black_list', []))
        v['periods'] = v['end_pos'] - v['start_pos'] + 1

    exps_list = [e for e in options_e if e > '201804']
    exps_solved = exp.list_experiments(path_abs, get_log_info=False, exp_list=exps_list)

    experiments = {e: exp.Experiment.from_dir(path_abs + e) for e in exps_solved}
    kpis_info = {e: {'maintenances': len(v.solution.get_maintenance_periods())}
                 for e, v in experiments.items()}
    checks = {e: {'infeasible': sum(len(vv) for vv in v.check_solution().values())}
              for e, v in experiments.items()}
    default_dict = {'maintenances': 9999, 'infeasible': 9999}
    dict_join = {e: {**exps_solved.get(e, default_dict),
                     **kpis_info.get(e, default_dict),
                     **checks.get(e, default_dict),
                     **options_e[e],
                     } for e in exps_list}

    table = pd.DataFrame.from_dict(dict_join, orient="index")
    table_filt = table[['tasks', 'periods', 'solver', 'infeasible', 'maintenances']]. \
        sort_values(['solver', 'tasks', 'periods', 'maintenances']).\
        drop_duplicates(subset=['tasks', 'periods', 'solver'])
    # sort_values(['maintenances'], ascending=True).\
    unstacked = table_filt.set_index(['tasks', 'periods', 'solver'])['maintenances'].unstack('solver')
    print(unstacked.pipe(tabulate.tabulate, headers='keys', tablefmt='pipe'))

    # results = get_results_table(path_abs, get_log_info=False)

    # gurobi_results = get_results_table(path_abs + 'GUROBI/')



def gurobi_vs_cplex():
    cols_rename = {
        'time_out': 'time (s)', 'index': 'id', 'objective_out': 'objective',
                   'gap_out': 'gap (\%)', 'bound_out': 'bound'
                   }
    cplex_results = get_results_table(path_abs)
    gurobi_results = get_results_table(path_abs + 'GUROBI/')
    df_cplex = cplex_results.rename(columns=cols_rename)[list(cols_rename.values())]
    df_gurobi = gurobi_results.rename(columns=cols_rename)[list(cols_rename.values())]
    print(df_cplex.pipe(tabulate.tabulate, headers='keys', tablefmt='pipe'))
    print(df_gurobi.pipe(tabulate.tabulate, headers='keys', tablefmt='pipe'))


def add_task_types():
    # path_types = path_abs + 'pieces10/'
    # path_to_compare = path_abs + '201801141607'

    path_types = path_abs + 'pieces20/'
    path_to_compare = path_abs + '201801102259'

    exp_complete = exp.Experiment.from_dir(path_to_compare)
    unavailables_complete = exp_complete.solution.get_unavailable()
    maintenances_complete = exp_complete.solution.get_in_maintenance()

    table_ref =\
        pd.merge(
            pd.DataFrame.from_dict(unavailables_complete, orient='index').rename(columns={0: 'ref_unavail'})
            ,pd.DataFrame.from_dict(maintenances_complete, orient='index').rename(columns={0: 'ref_maint'})
            ,left_index = True, right_index = True
        )

    exps = {f: exp.Experiment.from_dir(os.path.join(path_types, f)) for f in os.listdir(path_types)}
    unavailables = {k: v.solution.get_unavailable() for k, v in exps.items()}
    maintenances = {k: v.solution.get_in_maintenance() for k, v in exps.items()}
    # exp.exp_get_info(path_to_compare)['tasks']
    # exp.exp_get_info(path_to_compare)['periods']
    # sum(exp.exp_get_info(path_types + f)['tasks']
    #     for f in os.listdir(path_types))
    # [exp.exp_get_info(path_types + f)['periods']
    #     for f in os.listdir(path_types)]

    # pp.pprint(di.load_data(path_to_compare + '/options.json'))
    # pp.pprint(di.load_data(path_types + '2000_D/options.json'))
    # pp.pprint(exp.)
    # di.load_data(path_to_compare + '/options.json')

    table1 = pd.DataFrame.from_dict(unavailables, orient='index')
    table1 = table1.reset_index().melt(value_vars=table1.columns, id_vars='index').\
        rename(columns={'index': 'type', 'variable': 'period', 'value': 'unavail'}).\
        groupby('period').unavail.apply(sum).to_frame()

    table2 = pd.DataFrame.from_dict(maintenances, orient='index')
    table2 = table2.reset_index().melt(value_vars=table2.columns, id_vars='index').\
        rename(columns={'index': 'type', 'variable': 'period', 'value': 'maint'}).\
        groupby('period').maint.apply(sum).to_frame()

    table =\
        pd.merge(
            table1
            ,table2
            ,left_index = True, right_index = True
        ).merge(table_ref,left_index = True, right_index = True)

    print(table.pipe(tabulate.tabulate, headers='keys', tablefmt='pipe'))


def compare_heur_model():
    experiments = \
        ['201801141705', '201801141331', '201801141334', '201801141706',
         '201801141646', '201801131813', '201801102127', '201801131817',
         '201801141607', '201801102259']

    paths = {k: path_abs + k for k in experiments}
    exps = {k: exp.Experiment.from_dir(v) for k, v in paths.items()}
    heurs = {}
    for k, v in exps.items():
        heur_obj = heur.GreedyByMission(v.instance, options={'print': False})
        heur_obj.solve()
        heurs[k] = heur_obj

    results = {}
    results2 = {}
    for k in experiments:
        results[k] = exps[k].get_kpis()
        results2[k] = heurs[k].get_kpis()
        results2[k]["_resources"] = sum(heurs[k].check_solution().get('resources', {}).values())

    args = {'left_index': True, 'right_index': True, 'how': 'left', 'suffixes': ('_mod', '_heur')}
    table = pd.merge(
        pd.DataFrame.from_dict(results, orient="index")
        ,pd.DataFrame.from_dict(results2, orient="index")
        ,**args
    )
    table = table[sorted(table.columns)]
    table.reset_index(drop=True)

    reftable = get_results_table(path_abs)
    reftable = reftable[['index', 'date']].set_index('date')
    comparison = pd.merge(table, reftable, left_index=True, right_index=True).set_index('index').sort_index()
    print(comparison.pipe(tabulate.tabulate, headers='keys', tablefmt='pipe'))
    return comparison


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


def get_simulation_results(experiment, cols_rename=None):

    if cols_rename is None:
        cols_rename = {
            'time_out': 'time_out', 'index': 'id', 'objective_out': 'objective',
            'gap_out': 'gap_out', 'bound_out': 'bound', 'status': 'status',
            'cons': 'cons', 'vars': 'vars', 'nonzeros': 'nonzeros'
                       }
    path_exps = path_results + experiment
    exps = {p: os.path.join(path_exps, p) + '/' for p in os.listdir(path_exps)}

    results_list = {k: get_results_table(v, get_exp_info=False) for k, v in exps.items()}
    df_list = {k: df.rename(columns=cols_rename)[list(cols_rename.values())] for k, df in results_list.items()
               if len(df) > 0}
    table = pd.concat(df_list).reset_index() >> dp.rename(scenario=X.level_0, instance=X.level_1)
    names_df = na.config_to_latex(table.scenario)
    scenarios = table >> dp.distinct(X.scenario) >> dp.select(X.scenario)
    scenarios = scenarios.reset_index(drop=True) >> dp.mutate(code = X.index)

    @dp.make_symbolic
    def find_regex(series, regex_text):
        return [re.search(regex_text, t) is not None
                if t is not None and not pd.isna(t) else False
                for t in series]

    return table >> \
           dp.left_join(names_df, on="scenario") >> \
           dp.left_join(scenarios, on='scenario') >> \
           dp.mutate(no_int=find_regex(X.status, 'no integer solution'),
                  inf=find_regex(X.status, 'infeasible'))



def sim_list_to_md(df_list):
    for k, df in df_list.items():
        print(k)
        print(df.pipe(tabulate.tabulate, headers='keys', tablefmt='pipe'))


def summary_table(table_in):

    t3 = \
    table_in >> \
        dp.rename(t= 'time_out', g = 'gap_out') >> \
        dp.group_by(X.scenario) >> \
        dp.mutate(sizet = dp.n(X.t)) >> \
        dp.filter_by(~X.inf) >> \
        dp.summarize(g_med = X.g.median(),
                     t_max = X.t.max(),
                     t_min = X.t.min(),
                     t_med = X.t.median(),
                     cons = X.cons.mean(),
                     vars = X.vars.mean(),
                     non_0 = X.nonzeros.mean(),
                     no_int = X.no_int.sum(),
                     inf = dp.first(X.sizet) - dp.n(X.t),
                     code = dp.first(X.code),
                     case=dp.first(X.case)
                )

    # t3_b = {k: treat_table(v) for k, v in df_list.items()}
    return t3


def summary_to_latex(experiment, table, path):
    names_df = na.config_to_latex(table.scenario)

    eqs = {'${}^{{{}}}$'.format(a, m): '{}_{}'.format(a, m) for m in ['max', 'min', 'med'] for a in ['t', 'g']}
    eqs.update({'no-int': 'no_int', 'non-zero': 'non_0'})
    t4 = table.sort_values(by='t_med') >> \
         dp.left_join(names_df, on='scenario') >> \
        dp.select(X.code, X.case, X.t_min, X.t_med, X.t_max, X.non_0,
                  X.vars, X.cons,
                  # X.no_int, X.inf,
                  X.g_med) >> \
         dp.rename(**eqs)

    # t4 = t4.reindex(columns=['scenario']+t4.columns[:-1].tolist())

    latex = t4.to_latex(float_format='%.1f', escape=False, index=False)
    file_path = os.path.join(path, '{}.tex'.format(experiment))
    with open(file_path, 'w') as f:
        f.write(latex)


def boxplot_instances(table, column='time_out'):
    # table = table >> dp.filter_by(~X.inf)
    return ggplot(aes(x='code', y=column), data=table) + geom_boxplot() + \
    xlab(element_text("Scenario", size=20, vjust=-0.05)) + \
    ylab(element_text("Solving time (in seconds)", size=20, vjust=0.15))

if __name__ == "__main__":

    # path, _ = os.path.split(path_abs)
    # path, _ = os.path.split(path)
    # path = os.path.join(path, 'MOSIM2018_new_model')
    # exp_list = [d for d in os.listdir(path) if d.startswith('2018')]
    # dir_dest = "/home/pchtsp/Documents/projects/OPTIMA/R/presentations/CLAIO_presentation/"
    # results_table(dir_dest=dir_dest, dir_origin=path, exp_list=exp_list)

    # results_table()
    # remaining_graph()
    # progress_graph()
    # maintenances_graph()
    # maintenances_graph(maint=False)
    # multiobjective_table()
    # multi_multiobjective_table()
    # pp.pprint(d)
    # for x_key in x:
    #     print(x[x_key], y[x_key])

    # multiobjective_table()
    # pp.pprint(pareto_p2)
    # multiobjective_table()
    # table = get_results_table(path_abs)
    # solvers_comp()
    # experiment = 'clust1_20181031'



    pass
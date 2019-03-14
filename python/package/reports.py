import package.auxiliar as aux
import pandas as pd
import package.instance as inst
import package.experiment as exp
import os
import numpy as np
import package.data_input as di
import tabulate
import re
import package.heuristics as heur
from package.params import PATHS
import scripts.names as na

import dfply as dp
from dfply import X


path_root = PATHS['root']
path_abs = PATHS['experiments']
path_img = PATHS['img']
path_latex = PATHS['latex']
path_results = PATHS['results']


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
    maint = {k: max(v.solution.get_in_some_maintenance().values()) for k, v in experiments.items()}
    gaps = aux.get_property_from_dic(exps, "gap_out")


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
    print_table_md(unstacked)


def gurobi_vs_cplex():
    cols_rename = {
        'time_out': 'time (s)', 'index': 'id', 'objective_out': 'objective',
                   'gap_out': 'gap (\%)', 'bound_out': 'bound'
                   }
    cplex_results = get_results_table(path_abs)
    gurobi_results = get_results_table(path_abs + 'GUROBI/')
    df_cplex = cplex_results.rename(columns=cols_rename)[list(cols_rename.values())]
    df_gurobi = gurobi_results.rename(columns=cols_rename)[list(cols_rename.values())]
    print_table_md(df_cplex)
    print_table_md(df_gurobi)


def add_task_types():
    # path_types = path_abs + 'pieces10/'
    # path_to_compare = path_abs + '201801141607'

    path_types = path_abs + 'pieces20/'
    path_to_compare = path_abs + '201801102259'

    exp_complete = exp.Experiment.from_dir(path_to_compare)
    unavailables_complete = exp_complete.solution.get_unavailable()
    maintenances_complete = exp_complete.solution.get_in_some_maintenance()

    table_ref =\
        pd.merge(
            pd.DataFrame.from_dict(unavailables_complete, orient='index').rename(columns={0: 'ref_unavail'})
            ,pd.DataFrame.from_dict(maintenances_complete, orient='index').rename(columns={0: 'ref_maint'})
            ,left_index = True, right_index = True
        )

    exps = {f: exp.Experiment.from_dir(os.path.join(path_types, f)) for f in os.listdir(path_types)}
    unavailables = {k: v.solution.get_unavailable() for k, v in exps.items()}
    maintenances = {k: v.solution.get_in_some_maintenance() for k, v in exps.items()}

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
    print_table_md(table)


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
    print_table_md(comparison)
    return comparison


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


def get_simulation_results(experiment, cols_rename=None):

    if cols_rename is None:
        cols_rename = {
            'time': 'time_out', 'index': 'id', 'best_solution': 'objective',
            'gap': 'gap_out', 'best_bound': 'bound', 'status': 'status',
            'sol_code': 'sol_code', 'status_code': 'status_code',
            'matrix_post': 'matrix'
        }
    path_exps = path_results + experiment
    exps = {p: os.path.join(path_exps, p) + '/' for p in os.listdir(path_exps)}

    results_list = {k: get_results_table(v, get_exp_info=False) for k, v in exps.items()}
    # table.columns
    table = \
        pd.concat(results_list) >> \
        dp.select(list(cols_rename.keys()))

    table = \
        table.rename(columns=cols_rename).reset_index() >> \
        dp.rename(scenario=X.level_0, instance=X.level_1)

    # here we join the sub columns from the matrix
    if 'matrix' in table.columns:
        table = table >> \
                dp.bind_cols(table.matrix.apply(pd.Series)) >> \
                dp.rename(vars = X.variables, cons = X.constraints, nonzeros = X.nonzeros)

    names_df = na.config_to_latex(table.scenario)
    scenarios = table >> dp.distinct(X.scenario) >> dp.select(X.scenario)
    scenarios = scenarios.reset_index(drop=True) >> dp.mutate(code = X.index)

    @dp.make_symbolic
    def replace_3600(series):
        return np.where(series>=3600, 3600, series)

    table_n = \
        table >> \
           dp.left_join(names_df, on="scenario") >> \
           dp.left_join(scenarios, on='scenario')\

    if 'sol_code' in table_n.columns:
        table_n = table_n >> dp.mutate(no_int=X.sol_code == 0,
                                       inf=X.sol_code == -1)

    if 'time_out' in table_n.columns:
        table_n = table_n >> dp.mutate(time_out=replace_3600(X.time_out))

    return table_n


def sim_list_to_md(df_list):
    for k, df in df_list.items():
        print(k)
        print_table_md(df)


def summary_table(table_in):

    t3 = \
    table_in >> \
        dp.rename(t= 'time_out', g = 'gap_out') >> \
        dp.group_by(X.scenario) >> \
        dp.mutate(sizet = dp.n(X.t)) >> \
        dp.filter_by(~X.inf) >> \
        dp.summarize(g_med = X.g.median(),
                     g_avg=X.g.mean(),
                     t_max = X.t.max(),
                     t_min = X.t.min(),
                     t_med = X.t.median(),
                     t_avg=X.t.mean(),
                     cons = X.cons.mean(),
                     vars = X.vars.mean(),
                     non_0 = X.nonzeros.mean(),
                     no_int = X.no_int.sum(),
                     inf = dp.first(X.sizet) - dp.n(X.t),
                     # code = dp.first(X.code),
                     case=dp.first(X.case)
                )
    return t3


def summary_to_latex(experiment, table, path):
    eqs = {'${}^{{{}}}$'.format(a, m): '{}_{}'.format(a, m) for m in ['max', 'min', 'avg'] for a in ['t', 'g']}
    eqs.update({'no-int': 'no_int', 'non-zero': 'non_0', 'case': 'scenario'})
    t4 = table  >> \
         dp.ungroup() >>\
         dp.arrange(X.t_avg, X.g_avg) >>\
         dp.select(X.case, X.t_min, X.t_avg,
                   # X.t_max,
                   X.non_0,
                   X.vars, X.cons,
                   X.no_int,
                   X.inf,
                   X.g_avg) >> \
         dp.rename(**eqs)

    latex = t4.to_latex(float_format='%.1f', escape=False, index=False)
    file_path = os.path.join(path, '{}.tex'.format(experiment))
    with open(file_path, 'w') as f:
        f.write(latex)


def print_table_md(table):
    print(table.pipe(tabulate.tabulate, headers='keys', tablefmt='pipe'))


def col_names_collapsed(table):
    return ['.'.join(reversed(col)).strip() for col in table.columns.values]

if __name__ == "__main__":

    pass
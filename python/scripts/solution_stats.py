import package.experiment as exp
import pandas as pd
import package.params as params
import pytups.superdict as sd
import pytups.tuplist as tl
import os
import numpy as np
import package.rpy_graphs as rpyg
import package.reports as rep
import package.data_input as di
import package.model_upper_bound as mub

import orloge as ol

from rpy2.robjects import pandas2ri
import rpy2.robjects.lib.ggplot2 as ggplot2
import rpy2.robjects as ro
# from importlib import reload

pandas2ri.activate()

# name = 'dell_20190502_num_maint_2'
# name = 'dell_20190501'
name = 'dell_20190505'
name = 'dell_20190507'
name = 'dell_20190507_all'
path = params.PATHS['results'] + name +'/base/'
path_graphs = r'\\luq\franco.peschiera.fr$\MyDocs\graphs/'
os.environ['path'] += r';C:\Program Files (x86)\Graphviz2.38\bin'

experiments = [os.path.join(path, i) for i in os.listdir(path)]


def min_assign_consumption(case):
    tasks = case.instance.get_tasks()
    tasks_tt = \
        sd.SuperDict(tasks).\
        apply(lambda k, v: v['consumption']*v['num_resource']*v['min_assign'])
    return pd.Series(tasks_tt.values_l())


def get_num_maints(case):
    maints = case.get_maintenance_starts()
    return len(maints)


def get_cycle_sizes(case):

    cycles = case.get_all_maintenance_cycles()
    dist = case.instance.get_dist_periods

    cycles_between = \
        cycles.apply(lambda k, v: [dist(*vv) for vv in v[1:2]])

    return pd.Series([vv for v in cycles_between.values() for vv in v])


def get_last_maint_date(case):
    dist = case.instance.get_dist_periods
    next = case.instance.get_next_period
    first, last = (case.instance.get_param(p) for p in ['start', 'end'])
    last =  next(last)
    m_starts = case.get_maintenance_starts()

    last_dist = \
        m_starts.\
        to_dict(1). \
        vapply(lambda v: [vv for vv in v if vv >= first]). \
        vapply(sorted).\
        vapply(lambda v: v+[last]).\
        vapply(lambda v: dist(v[1], last))

    return last_dist


def get_rel_consumptions(case):
    ranged = case.instance.get_periods_range
    tasks = case.instance.get_tasks()
    tasks_tt = \
        sd.SuperDict(tasks). \
            apply(lambda k, v:
                  sd.SuperDict({p: v['consumption']*v['min_assign']
                                for p in ranged(v['start'], v['end'])})). \
            to_dictup(). \
            to_tuplist(). \
            to_dict(result_col=2, indices=[1]). \
            apply(lambda _, x: sum(x)).to_tuplist()
    tasks_tt.sort()
    dates, values = zip(*tasks_tt)
    return pd.Series(values)


def get_consumptions(case, hours=True):

    ranged = case.instance.get_periods_range
    tasks = case.instance.get_tasks()
    tasks = sd.SuperDict.from_dict(tasks)
    if not hours:
        tasks = tasks.apply(lambda k, v: {**v, **{'consumption': 1}})
    tasks_tt = \
        tasks.\
        apply(lambda k, v:
                  sd.SuperDict({p: v['consumption']*v['num_resource']
                                for p in ranged(v['start'], v['end'])})).\
        to_dictup().\
        to_tuplist().\
        to_dict(result_col=2, indices=[1]).\
        apply(lambda _, x: sum(x)).to_tuplist()
    tasks_tt.sort()
    dates, values = zip(*tasks_tt)
    return pd.Series(values)


def get_init_hours(case):
    return pd.Series([*case.instance.get_resources('initial_used').values()])


def get_argmedian(consumption, prop=0.5):
    half = sum(consumption) * prop
    so_far = 0
    for pos, item in enumerate(consumption):
        so_far += item
        if so_far > half:
            return pos
    return len(consumption)


def get_geomean(consumption):
    total = sum(consumption)
    result = sum(pos * item for pos, item in enumerate(consumption)) / total
    return result


def cycle_sizes():
    cycles =[get_cycle_sizes(case) for case in cases if case is not None]
    all_cycles = np.asarray(cycles).flatten()

    plot = ggplot2.ggplot(pd.DataFrame(all_cycles, columns=['cycle'])) + \
           ggplot2.aes_string(x='cycle') + \
           ggplot2.geom_bar(position="identity") + \
           ggplot2.theme_minimal()

    path_out = path_graphs + r'hist_{}_{}.png'.format('all_cycles', name)
    plot.save(path_out)


def instance_status(experiments, vars_extract):
    log_paths = sd.SuperDict({os.path.basename(e): os.path.join(e, 'results.log')
                              for e in experiments})
    ll = \
        log_paths.\
        clean(func=os.path.exists).\
        vapply(lambda v: ol.get_info_solver(v, 'CPLEX', get_progress=False)). \
        vapply(lambda x: {var: x[var] for var in vars_extract})

    master = \
        pd.DataFrame({'sol_code': [ol.LpSolutionIntegerFeasible, ol.LpSolutionOptimal, ol.LpSolutionInfeasible],
                      'status': ['IntegerFeasible', 'Optimal', 'Infeasible']})
    return \
        pd.DataFrame.\
        from_dict(ll, orient='index').\
        rename_axis('name').\
        reset_index().merge(master, on='sol_code')

result = []
cases = [exp.Experiment.from_dir(e) for e in experiments]
basenames = [os.path.basename(e) for e in experiments]

for p, e in enumerate(experiments):
    # print(e)
    case = cases[p]
    # we clean errors.
    if case is None:
        continue
    consumption = get_consumptions(case)
    aircraft_use = get_consumptions(case, hours=False)
    rel_consumption = get_rel_consumptions(case)
    cycle_size = get_cycle_sizes(case)
    l_maint_date = get_last_maint_date(case).values_l()
    init_hours = get_init_hours(case)
    cy_sum = cycle_size.agg(['mean', 'max', 'min']).tolist()
    airc_sum = aircraft_use.agg(['mean', 'max', 'var']).tolist()
    cons_sum = consumption.agg(['mean', 'max', 'var']).tolist()
    l_maint_date_stat = pd.Series(l_maint_date).agg(['mean', 'max', 'min']).tolist()
    pos_consum = [get_argmedian(consumption, prop) for prop in [0.5, 0.75, 0.9]]
    pos_aircraft = [get_argmedian(aircraft_use, prop) for prop in [0.5, 0.75, 0.9]]
    geomean_airc = get_geomean(aircraft_use)
    geomean_cons = get_geomean(consumption)
    quantsw = rel_consumption.rolling(12).mean().shift(-11).quantile(q=[0.5, 0.75, 0.9]).tolist()
    init_sum = init_hours.agg(['mean']).tolist()
    num_maints = [get_num_maints(case)]
    cons_min_assign = min_assign_consumption(case).agg(['mean', 'max']).tolist()
    num_errors = 0
    errors = di.load_data(e + '/errors.json')
    _case_name = basenames[p]
    if errors:
        num_errors = sd.SuperDict(errors).to_dictup().len()
    result.append([_case_name] + init_sum +
                  cons_sum +
                  airc_sum +
                  num_maints +
                  cons_min_assign + quantsw +
                  pos_consum +
                  pos_aircraft +
                  [geomean_cons, geomean_airc] + [num_errors] +
                  cy_sum +
                  l_maint_date_stat)

names = ['name', 'init',
         'mean_consum', 'max_consum', 'var_consum',
         'mean_airc', 'max_airc', 'var_airc',
         'maints',
         'cons_min_mean', 'cons_min_max', 'quant5w', 'quant75w', 'quant9w',
         'pos_consum5', 'pos_consum75', 'pos_consum9',
         'pos_aircraft5', 'pos_aircraft75', 'pos_aircraft9',
         'geomean_cons', 'geomean_airc', 'num_errors',
         'mean_dist', 'max_dist', 'min_dist',
         'mean_2maint',  'max_2maint', 'min_2maint']

renames = {p: n for p, n in enumerate(names)}
result_tab = pd.DataFrame(result).rename(columns=renames)
# result_tab = result_tab[result_tab.num_errors==0]

result_tab['mean_consum_cut'] = pd.qcut(result_tab.mean_consum, q=10).astype(str)
indeces = pd.DataFrame({'mean_consum_cut': sorted(result_tab['mean_consum_cut'].unique())}).\
    reset_index().rename(columns={'index': 'mean_consum_cut_2'})
result_tab = result_tab.merge(indeces, on='mean_consum_cut')

for col in ['init', 'quant75w', 'quant9w', 'quant5w',
            'var_consum', 'cons_min_mean', 'pos_consum5', 'pos_consum9']:
    result_tab[col+'_cut'] = pd.qcut(result_tab[col], q=3).astype(str)

result_tab_summ = \
    result_tab.groupby(['mean_consum_cut_2', 'init_cut']).\
    agg({'mean_dist': ['mean', 'var'],
         'maints': ['mean', 'var']})

status_df = instance_status(experiments, ['sol_code', 'status_code', 'time', 'gap', 'best_bound', 'best_solution'])
status_df['gap_abs'] = status_df.best_solution - status_df.best_bound
# status_df[status_df.sol_code==ol.LpSolutionInfeasible]
status_df[status_df.gap_abs > 100]
# status_df.columns

# TODO: we filter instances that are too far... from optimality
#

def write_index():
    _filter = np.all([status_df.sol_code==ol.LpSolutionIntegerFeasible,
                     status_df.gap_abs >= 50], axis=0)
    remakes_df = status_df[_filter]

    remakes = remakes_df.name.tolist()
    remakes_path = r'C:\Users\franco.peschiera.fr\Documents\optima_results\dell_20190507_remakes2/base/index.txt'

    with open(remakes_path, 'w') as f:
        f.write('\n'.join(remakes))

#####################
# EXPORT THINGS
#####################

# rep.print_table_md(result_tab_summ)
# result_tab.to_csv(path_graphs + 'table.csv', index=False)
# rpyg.gantt_experiment(e+'/')

#####################
# histograms
#####################
def draw_hist(var='maints'):
    plot = ggplot2.ggplot(result_tab) + \
           ggplot2.aes_string(x=var) + \
           ggplot2.geom_bar(position="identity") + \
           ggplot2.theme_minimal()

    path_out = path_graphs + r'hist_{}_{}.png'.format(var, name)
    plot.save(path_out)

for var in ['maints', 'mean_dist', 'max_dist', 'min_dist']:
    draw_hist(var)
    
draw_hist('mean_2maint')

def hist_no_agg(cases):
    # basenames= 1
    var = 'sec_maint'
    cases_dict = sd.SuperDict(zip(basenames, cases))
    tt = cases_dict.clean(func=lambda v: v is not None).vapply(get_last_maint_date)
    ttt = pd.DataFrame.from_dict(tt).stack().rename(var).reset_index()
    # ttt = ttt[ttt.sec_maint>0]
    # ttt[ttt.sec_maint >= 40]
    plot = ggplot2.ggplot(ttt) + \
           ggplot2.aes_string(x=var) + \
           ggplot2.geom_histogram(position="identity") + \
           ggplot2.scale_y_log10()+\
           ggplot2.theme_minimal()

    path_out = path_graphs + r'hist_all_{}_{}.png'.format(var, name)
    plot.save(path_out)

# case = cases_dict['201905081456_765']
# tt['201905081456_765']
# case.get_maintenance_starts()

#####################
# consumption + init against vars
#####################
def cons_init(var='mean_dist'):
    table = result_tab.merge(status_df, on=['name'], how='left')
    plot = ggplot2.ggplot(table) + \
           ggplot2.aes_string(x='mean_consum', y=var, color='status') + \
           ggplot2.geom_jitter(alpha=0.8, height=0.2) + \
           ggplot2.geom_smooth(method = 'loess') + \
           ggplot2.facet_grid(ro.Formula('init_cut ~ .')) + \
           ggplot2.theme_minimal()

    path_out = path_graphs + r'mean_consum_init_vs_{}_{}.png'.format(var, name)
    plot.save(path_out)

for var in ['maints', 'mean_dist', 'mean_2maint']:
    cons_init(var)

#####################
# consumption_min vs main_dist
#####################
var = 'cons_min_max_vs_maints'
def test3():
    plot = ggplot2.ggplot(result_tab) + \
           ggplot2.aes_string(x='cons_min_max', y='maints') + \
           ggplot2.geom_jitter(alpha=0.8, height=0.1) + \
           ggplot2.geom_smooth(method = 'loess') + \
           ggplot2.theme_minimal()

    path_out = path_graphs + r'{}_{}.png'.format(var, name)
    plot.save(path_out)

#####################
var = 'mean_dist_vs_maints'
#####################
def test4():
    plot = ggplot2.ggplot(result_tab) + \
           ggplot2.aes_string(x='mean_dist', y='maints') + \
           ggplot2.geom_jitter(alpha=0.8, height=0.1) + \
           ggplot2.theme_minimal()

    path_out = path_graphs + r'{}_{}.png'.format(var, name)
    plot.save(path_out)


#####################
var = 'cons_min_max_vs_max_dist_by_pos_consum'
#####################
def test6():
    plot = ggplot2.ggplot(result_tab) + \
           ggplot2.aes_string(x='cons_min_max', y='max_dist') + \
           ggplot2.geom_jitter(alpha=0.8, height=0.1) + \
           ggplot2.facet_grid(ro.Formula('pos_consum9_cut ~ .'))+\
           ggplot2.theme_minimal()

    path_out = path_graphs + r'{}_{}.png'.format(var, name)
    plot.save(path_out)

#####################
# check instances that do not fit
#####################
def test5():
    few_maints = result_tab.query('maints <= 27 & mean_consum >= 170').copy()
    few_maints['freq_maints'] = 'few'
    many_maints = result_tab.query('maints == 31 & mean_consum <= 150').copy()
    many_maints['freq_maints'] = 'many'
    weird_maints = pd.concat([few_maints, many_maints])

    experiments_few_mains = [os.path.basename(experiments[p]) for p in few_maints.index]
    experiments_many_maints = [os.path.basename(experiments[p]) for p in many_maints.index]

    result_tab_t = result_tab.join(weird_maints[['freq_maints']])
    result_tab_t.freq_maints.fillna('other', inplace=True)
    result_tab_t.init_cut = result_tab_t.init_cut.astype(str).str[1]
    result_tab_t.groupby(['freq_maints', 'init_cut']).agg('mean').stack().unstack(0)


    e = experiments_few_mains[0]
    case = exp.Experiment.from_dir(e)
    len(case.get_maintenance_starts())

# x='mean_consum', y='mean_dist'

####################
# Lineal regression
####################

####################
# Forecasting
####################

from sklearn import tree
from sklearn import linear_model
import graphviz
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# from sklearn.tree import export_graphviz
# from sklearn import preprocessing

#
# le = preprocessing.LabelEncoder()
# le.fit(["paris", "paris", "tokyo", "amsterdam"])
# le.transform(["tokyo", "tokyo", "paris"])
# list(le.inverse_transform([2, 2, 1]))

# for col in ['init', 'quant75w', 'quant9w', 'quant5w', 'var_consum', 'cons_min_mean']:
#     result_tab[col+'_cut'] = pd.qcut(result_tab[col], q=2).astype(str)
var = 'maints'
# var = 'mean_dist'
# var = 'max_dist'
# var = 'is_less_60'
# var = 'maint_less_28'
# result_tab[var] = result_tab.max_dist < 58
# result_tab['maint_less_28'] = result_tab.maints < 28
path_out = path_graphs + r'tree_{}_{}'.format(var, name)
aux_take_out = [col for col in result_tab.columns
                if col.endswith('_cut')
                or col.endswith('_dist')]
other_out = ['maints', 'name']
# aux_take_out = [col for col in result_tab.columns if col.endswith('_cut') or col.startswith('pos_consum')]
X = result_tab.drop(other_out, axis=1).drop(aux_take_out, axis=1).drop(var, errors='ignore', axis=1)
Y = result_tab['maints']

def create_X_Y(x_vars, y_var):
    X = result_tab[x_vars].copy()
    X['mean_consum_2'] = X.mean_consum ** 2
    # X['mean_consum_3'] = X.mean_consum**3
    Y = result_tab[y_var]
    # 70% training and 30% test
    return \
        train_test_split(X, Y, test_size=0.3, random_state=1)
    # return

result_tab.columns
# X.columns[18]
# X.columns[14]
# X.columns[0]
# X.columns[1]
# X.columns[18]
# X.columns[20]

# X.init

def decision_tree(x_vars, y_var):
    X_train, X_test, y_train, y_test = create_X_Y(x_vars, y_var)
    clf = tree.DecisionTreeRegressor(max_depth=10)
    clf.fit(X=X_train, y=y_train)
    y_pred = clf.predict(X_test)
    metrics.mean_absolute_error(y_test, y_pred)
    metrics.mean_squared_error(y_test, y_pred)

def regression(x_vars, y_var):
    # clf = linear_model.LassoCV()
    # clf.fit(X, Y)
    X_train, X_test, y_train, y_test = create_X_Y(x_vars, y_var)
    clf = linear_model.LinearRegression()
    clf.fit(X=X_train, y=y_train)
    print(clf.coef_)
    y_pred = clf.predict(X_test)
    args = (y_test, y_pred)
    metrics.mean_absolute_error(*args)
    metrics.mean_squared_error(*args)
    return metrics.r2_score(*args)

def classify(x_vars, y_var):
    X_train, X_test, y_train, y_test = create_X_Y(x_vars, y_var)
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf.fit(X=X_train, y=y_train, sample_weight=y_train*10 +1)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    return metrics.confusion_matrix(y_test, y_pred)

regression(['mean_consum', 'init', 'pos_consum5'], 'mean_dist')
regression(['mean_consum', 'init', 'geomean_cons'], 'maints')

# ['name', 'init', 'mean_consum', 'max_consum', 'var_consum', 'mean_airc',
#        'max_airc', 'var_airc', 'maints', 'cons_min_mean', 'cons_min_max',
#        'quant5w', 'quant75w', 'quant9w', 'pos_consum5', 'pos_consum75',
#        'pos_consum9', 'pos_aircraft5', 'pos_aircraft75', 'pos_aircraft9',
#        'geomean_cons', 'geomean_airc', 'num_errors', 'mean_dist', 'max_dist',
#        'min_dist', 'mean_consum_cut', 'mean_consum_cut_2', 'init_cut',
#        'quant75w_cut', 'quant9w_cut', 'quant5w_cut', 'var_consum_cut',
#        'cons_min_mean_cut', 'pos_consum5_cut', 'pos_consum9_cut']

def plotting(data_frame, x_name, y_name, y_pred_name, graph_name):
    # Plot!
    # result = pd.DataFrame(x_test)
    # result['y_test'] = y_test
    # result['y_pred'] = y_pred
    plot = ggplot2.ggplot(data_frame) + \
           ggplot2.aes_string(x=x_name, y=y_name, color='status') + \
           ggplot2.geom_jitter(alpha=0.8, height=0.1) + \
           ggplot2.geom_line(ggplot2.aes_string(y=y_pred_name), color='blue') + \
           ggplot2.facet_grid(ro.Formula('init_cut ~ .')) + \
           ggplot2.theme_minimal()

    path_out = path_graphs + r'{}_{}.png'.format(graph_name, name)
    plot.save(path_out)


def regression_superquantiles(_var, grade=1, **kwargs):

    X = result_tab[['mean_consum', 'init']].copy()
    Y = result_tab[_var]
    for g in range(1, grade+1):
        X['mean_consum'+str(g)] = X.mean_consum**g
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    coef0, coefs = mub.regression_VaR(X_train, y_train, **kwargs)
    print(coefs)

    X_out = X_test.copy()
    X_out[_var] = y_test
    X_out['pred'] = y_pred = np.sum([v*X_out[k] for k, v in coefs.items()], axis=0) + coef0
    args = (y_test, y_pred)
    print(metrics.mean_absolute_error(*args))
    print(metrics.mean_squared_error(*args))
    X_out = X_out.join(result_tab[['init_cut', 'name']])
    X_out = X_out.merge(status_df, on='name', how='left')
    graph_name = 'regression_mean_consum_g{}_init_{}'.format(grade, _var)
    plotting(X_out, 'mean_consum', _var, 'pred', graph_name)

for _var in ['maints', 'mean_2maint', 'mean_dist']:
    for grade in range(1, 4):
        regression_superquantiles(_var, grade, _lambda=0.1)


# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render(filename=path_out)
#
# Y_out = y_test.to_frame()
# Y_out.columns.values[0] = 'y_real'
# Y_out['y_pred'] = y_pred
# Y_out['dif'] = (Y_out.y_pred + 0 - Y_out.y_real + 0)
# Y_out['dif'].sum()/len(Y_out)
# worst_predicted = Y_out[Y_out['dif']**2>5]
# bad_indeces = worst_predicted.index
# result_tab[
#     ['name', 'mean_consum', 'pos_consum5', 'pos_consum9']
#     ].join(worst_predicted['dif'], how='inner').\
#     round().sort_values(['mean_consum', 'pos_consum5'])
# worst_predicted.join(result_tab['name'], how='inner')

# for classification:
# worst_predicted = Y_out[Y_out['dif']<0]
# worst_predicted.join(result_tab['name'], how='inner')
# len(Y_out[Y_out['dif']>3])

# 3       201905011903        160.0           40           80 -7.0
# 146     201905021153        160.0           45           78  5.0
import package.experiment as exp
import pandas as pd
import package.params as params
import package.superdict as sd
import package.tuplist as tl
import os
import numpy as np
import package.rpy_graphs as rpyg
from rpy2.robjects import pandas2ri
import package.reports as rep
import rpy2.robjects.lib.ggplot2 as ggplot2
import rpy2.robjects as ro
import package.data_input as di

pandas2ri.activate()

# name = 'dell_20190502_num_maint_2'
# name = 'dell_20190501'
name = 'dell_20190505'
name = 'dell_20190507'
path = params.PATHS['results'] + name +'/base/'
path_graphs = r'\\luq\franco.peschiera.fr$\MyDocs\graphs/'
os.environ['path'] += r';C:\Program Files (x86)\Graphviz2.38\bin'

experiments = [os.path.join(path, i) for i in os.listdir(path)]

# e = experiments[0]
# case = exp.Experiment.from_dir(e)


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


result = []
for e in experiments:
    # print(e)
    case = exp.Experiment.from_dir(e)
    # we clean errors.
    if case is None:
        continue
    consumption = get_consumptions(case)
    aircraft_use = get_consumptions(case, hours=False)
    rel_consumption = get_rel_consumptions(case)
    cycle_size = get_cycle_sizes(case)
    init_hours = get_init_hours(case)
    cy_sum = cycle_size.agg(['mean', 'max', 'min']).tolist()
    airc_sum = aircraft_use.agg(['mean', 'max', 'var']).tolist()
    cons_sum = consumption.agg(['mean', 'max', 'var']).tolist()
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
    _case_name = os.path.basename(e)
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
                  cy_sum)

names = ['name', 'init',
         'mean_consum', 'max_consum', 'var_consum',
         'mean_airc', 'max_airc', 'var_airc',
         'maints',
         'cons_min_mean', 'cons_min_max', 'quant5w', 'quant75w', 'quant9w',
         'pos_consum5', 'pos_consum75', 'pos_consum9',
         'pos_aircraft5', 'pos_aircraft75', 'pos_aircraft9',
         'geomean_cons', 'geomean_airc', 'num_errors',
         'mean_dist', 'max_dist', 'min_dist']

renames = {p: n for p, n in enumerate(names)}
result_tab = pd.DataFrame(result).rename(columns=renames)
# result_tab = result_tab[result_tab.num_errors==0]

result_tab['mean_consum_cut'] = pd.qcut(result_tab.mean_consum, q=10).astype(str)
indeces = pd.DataFrame({'mean_consum_cut': sorted(result_tab['mean_consum_cut'].unique())}).\
    reset_index().rename(columns={'index': 'mean_consum_cut_2'})
result_tab = result_tab.merge(indeces, on='mean_consum_cut')

for col in ['init', 'quant75w', 'quant9w', 'quant5w',
            'var_consum', 'cons_min_mean', 'pos_consum5', 'pos_consum9']:
    result_tab[col+'_cut'] = pd.qcut(result_tab[col], q=2).astype(str)

result_tab_summ = \
    result_tab.groupby(['mean_consum_cut_2', 'init_cut']).\
    agg({'mean_dist': ['mean', 'var'],
         'maints': ['mean', 'var']})

# result_tab.columns
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

#####################
var = 'mean_consum_vs_mean_dist_by_pos_consum'
#####################
def test1():
    plot = ggplot2.ggplot(result_tab) + \
           ggplot2.aes_string(x='mean_consum', y='mean_dist') + \
           ggplot2.geom_point(alpha=0.8) + \
           ggplot2.geom_smooth(method = 'lm') + \
           ggplot2.theme_minimal()

    path_out = path_graphs + r'{}_{}.png'.format(var, name)
    plot.save(path_out)

#####################
# consumption vs main_dist
#####################
var = 'consumption_vs_maints'
def test2():
    plot = ggplot2.ggplot(result_tab) + \
           ggplot2.aes_string(x='mean_consum', y='maints') + \
           ggplot2.geom_jitter(alpha=0.8, height=0.2) + \
           ggplot2.geom_smooth(method = 'loess') + \
           ggplot2.facet_grid(ro.Formula('init_cut ~ .'))+\
           ggplot2.theme_minimal()

    path_out = path_graphs + r'{}_{}.png'.format(var, name)
    plot.save(path_out)

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
# Trees
# #

from sklearn import tree
import graphviz
from sklearn import metrics
from sklearn.model_selection import train_test_split
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
Y = result_tab[var]
X.columns[18]
X.columns[14]
X.columns[0]
X.columns[1]
X.columns[18]
X.columns[20]

# X.init
# 70% training and 30% test
X_train, X_test, y_train, y_test = \
    train_test_split(X, Y, test_size=0.3, random_state=1)


clf = tree.DecisionTreeRegressor(max_depth=10)
clf.fit(X=X_train, y=y_train)
y_pred = clf.predict(X_test)
metrics.mean_absolute_error(y_test, y_pred)
metrics.mean_squared_error(y_test, y_pred)

def classify():
    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf.fit(X=X_train, y=y_train, sample_weight=y_train*10 +1)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    metrics.confusion_matrix(y_test, y_pred)

dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render(filename=path_out)

Y_out = y_test.to_frame()
Y_out.columns.values[0] = 'y_real'
Y_out['y_pred'] = y_pred
Y_out['dif'] = (Y_out.y_pred + 0 - Y_out.y_real + 0)
Y_out['dif'].sum()/len(Y_out)
worst_predicted = Y_out[Y_out['dif']**2>5]
bad_indeces = worst_predicted.index
result_tab[
    ['name', 'mean_consum', 'pos_consum5', 'pos_consum9']
    ].join(worst_predicted['dif'], how='inner').\
    round().sort_values(['mean_consum', 'pos_consum5'])
worst_predicted.join(result_tab['name'], how='inner')

# for classification:
worst_predicted = Y_out[Y_out['dif']<0]
worst_predicted.join(result_tab['name'], how='inner')
# len(Y_out[Y_out['dif']>3])

# 3       201905011903        160.0           40           80 -7.0
# 146     201905021153        160.0           45           78  5.0
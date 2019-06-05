import scripts.COR2019 as cor
import package.reports as rep
import pandas as pd
import dfply as dp
from dfply import X

# import os

input = ['dell_20190327_cbc_old',
         'dell_20190327_cbc_new',
         'dell_20190402_cbc_new_new',
         'dell_20190404_new_new_cbc210_6months',
         'dell_20190410_new_new_cbc_sanshours',
         None,
         None,
         'dell_20190426_new_new_cbc_newobjs_rut']

input = [
    'dell_20190326_cplex_old_only5',
    'dell_20190327_cplex_new_only5',
    'dell_20190403_new_new_cplex',
    'dell_20190408_new_new_cplex_6months',
    'dell_20190410_new_new_cplex_sanshours',
    'dell_20190419_new_new_cplex_newobjs_sanshours',
    'dell_20190419_new_new_cplex_newobjs',
    'dell_20190422_new_new_cplex_newobjs_rut',
    'dell_20190430_new_new_gurobi_newobjs_rut'
         ]


input = ['dell_20190529_allcuts',
         'dell_20190529_strcuts',
         'dell_20190529_nocuts',
         'dell_20190529_strANDmaxcuts',
         'dell_20190529_strcutsANDmaints',
         'dell_20190529_strcutsANDmaxmaints']
# file_path = r'\\luq\franco.peschiera.fr$\MyDocs\graphs\cplex_cut_comparison.html'
# file_path = r'\\luq\franco.peschiera.fr$\MyDocs\graphs\cbc_comparison.tex'
# file_path = r'\\luq\franco.peschiera.fr$\MyDocs\graphs\cbc_comparison_2.html'
file_path = r'\\luq\franco.peschiera.fr$\MyDocs\graphs\cuts_comparison.tex'
# file_path = r'\\luq\franco.peschiera.fr$\MyDocs\graphs\cplex_vs_cbc.html'
# input = [('new', 'dell_20190401_maintenances'), ('old', 'dell_20190327_cbc_old')]
#####################
# INFO

results_dict = {}
raw_results_dict = {}
for name, exp in enumerate(input):
    if exp is None:
        continue
    table = rep.get_simulation_results(exp)
    table_errors = rep.get_experiment_errors(exp)
    table_n = table >> dp.left_join(table_errors, by=['scenario', 'instance'])
    raw_results_dict[name] = table_n
    results_dict[name] = \
        table_n >> \
        dp.mutate(gap_abs = X.objective - X.bound) >> \
            dp.rename(t= 'time_out', g = 'gap_abs') >> \
            dp.group_by(X.scenario) >> \
            dp.mutate(sizet = dp.n(X.t)) >> \
            dp.filter_by(~X.inf) >> \
            dp.summarize(g_med=X.g.median(),
                         g_avg=X.g.mean(),
                         t_max=X.t.max(),
                         t_min=X.t.min(),
                         t_med=X.t.median(),
                         t_avg=X.t.mean(),
                         cons=X.cons.mean(),
                         vars=X.vars.mean(),
                         non_0=X.nonzeros.mean(),
                         no_int=X.no_int.sum(),
                         inf=dp.first(X.sizet) - dp.n(X.t),
                         errs=X.errors.sum()
                    )

stats_order = ['cons', 'vars', 'non_0', 'inf', 'no_int', 'errs'] \
              + ['t_{}'.format(time) for time in ['min', 'med', 'avg', 'max']]\
              + ['g_{}'.format(time) for time in ['min', 'med', 'avg', 'max']]
stats_order_dict = {stat: pos for pos, stat in enumerate(stats_order)}

results = pd.concat(results_dict)
df = pd.concat(raw_results_dict).reset_index()
df[df.inf].sort_values(['level_1', 'level_0'])

results2 = results.reset_index().drop(['level_1'], axis=1).\
    rename(columns=dict(level_0='experiment'))

results3 = results2[results2.scenario != 'numperiod_140'].\
    set_index(['scenario', 'experiment']).stack().\
    unstack(0)
results3['order'] = results3.index.get_level_values(1).map(stats_order_dict)

results4 = results3.reset_index().\
    rename(columns={'experiment': 'exp'}).\
    set_index(['level_1', 'exp']).\
    sort_values(['order', 'exp']).\
    drop('order', axis=1)

text = results4.to_latex(bold_rows=True, float_format='%.1f', longtable=True)

with open(file_path, 'w') as f:
    f.write(text)

#####################
# RELAXATIONS

# results_dict = {}
# for name, exp in input:
#     results_dict[name] = cor.statistics_relaxations(exp, write=False)
#
# results = pd.concat(results_dict)
#
# results2 = results.reset_index().drop(['level_1'], axis=1).\
#     rename(columns=dict(level_0='experiment'))
#
# html = results2.set_index(['case', 'experiment']).\
#     unstack(1).to_html(float_format='%.1f')
#
# with open(file_path, 'w') as f:
#     f.write(html)
#

#####################

# cor.statistics_experiment(experiment)

# scenarios = \
#     dict(
#         MIN_HOUR_5="dell_20190327_2/numperiod_90",
#         MIN_HOUR_15="dell_20190327_2/numperiod_90",
#         MIN_HOUR_20="dell_20190327_2/numperiod_90",
#         BASE="clust_params2_cplex/base",
#     )
#
# cor.get_scenarios_to_compare(scenarios)


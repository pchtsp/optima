import scripts.COR2019 as cor
import package.reports as rep
import pandas as pd
import dfply as dp
from dfply import X

input = [('0', 'dell_20190327_cbc_old'),
         ('1', 'dell_20190327_cbc_new'),
         ('2', 'dell_20190402_cbc_new_new'),
         ('3', 'dell_20190404_new_new_cbc210_6months'),
         ('4', 'dell_20190410_new_new_cbc_sanshours')]
input = [('0', 'dell_20190326_cplex_old'),
         ('1', 'dell_20190327_cplex_new_only5'),
         ('2', 'dell_20190403_maintenances_cplex'),
         ('3', 'dell_20190408_new_new_cplex_6months'),
         ('4', 'dell_20190410_new_new_cplex_sanshours')]
file_path = r'\\luq\franco.peschiera.fr$\MyDocs\graphs\cplex_cut_comparison.html'
file_path = r'\\luq\franco.peschiera.fr$\MyDocs\graphs\cbc_comparison.html'
file_path = r'\\luq\franco.peschiera.fr$\MyDocs\graphs\cplex_comparison.html'
# file_path = r'\\luq\franco.peschiera.fr$\MyDocs\graphs\cplex_vs_cbc.html'
# input = [('new', 'dell_20190401_maintenances'), ('old', 'dell_20190327_cbc_old')]
#####################
# INFO

results_dict = {}
for name, exp in input:
    table = rep.get_simulation_results(exp)
    results_dict[name] = \
        table >> \
        dp.mutate(gap_abs = X.objective - X.bound) >> \
            dp.rename(t= 'time_out', g = 'gap_abs') >> \
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
                         inf = dp.first(X.sizet) - dp.n(X.t)
                    )

results = pd.concat(results_dict)

results2 = results.reset_index().drop(['level_1'], axis=1).\
    rename(columns=dict(level_0='experiment'))

html = results2[results2.scenario != 'numperiod_140'].\
    set_index(['scenario', 'experiment']).\
    unstack(1).to_html(float_format='%.1f')

with open(file_path, 'w') as f:
    f.write(html)

#####################
# RELAXATIONS

results_dict = {}
for name, exp in input:
    results_dict[name] = cor.statistics_relaxations(exp, write=False)

results = pd.concat(results_dict)

results2 = results.reset_index().drop(['level_1'], axis=1).\
    rename(columns=dict(level_0='experiment'))

html = results2.set_index(['case', 'experiment']).\
    unstack(1).to_html(float_format='%.1f')

with open(file_path, 'w') as f:
    f.write(html)


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


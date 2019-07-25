import stochastic.graphs as graphs
import stochastic.params as sto_params
import scripts.stochastic_analysis as sto_funcs
import stochastic.models as models

import rpy2.robjects.lib.ggplot2 as ggplot2


name = sto_params.name
use_zip = sto_params.use_zip
relpath = name + '/base/'

if use_zip:
    cases, log_info = sto_funcs.get_cases_logs_zip(name, relpath)
else:
    cases, log_info = sto_funcs.get_cases_logs(relpath)

result_tab = sto_funcs.get_table(cases, 1)
result_tab = sto_funcs.treat_table(result_tab)
result_tab = sto_funcs.merge_table_status(result_tab, log_info)


##################
# Number of maintenances
##################

y = 'maints'
x = 'mean_consum'
graph_name = '{}_vs_{}_nocolor'.format(x, y)
equiv = \
    {'(304.732, 554.756]': 'Low initial status',
     '(554.756, 614.911]': 'Medium initial status',
     '(614.911, 812.467]': 'High initial status'}
result_tab['init_cut_pretty'] = result_tab['init_cut'].map(equiv)
plot = graphs.plotting(result_tab, x=x, y=y, graph_name=graph_name, smooth=False, save=False,
                       facet='init_cut_pretty ~ .')
theme_params = {
    # angle = 90,
    'axis.text.x': ggplot2.element_text(size=10, hjust=0),
    'axis.text.y': ggplot2.element_text(size=10),
    'axis.title': ggplot2.element_text(size=10, face="bold")
}
plot += ggplot2.labs(x='Average consumption in flight hours per period', y='Total number of maintenances')
plot += ggplot2.theme(**theme_params)
path_out = sto_params.path_graphs + r'{1}/{0}_{1}.png'.format(graph_name, name)
plot.save(path_out)

##################
# QuantReg!
##################
y = 'mean_dist'
opt = dict(q=0.1, max_iter=10000, bound='lower', y_var=y)
x_vars = ['mean_consum', 'mean_consum2', 'mean_consum3', 'mean_consum4', 'mean_consum5', 'init', 'var_consum', 'spec_tasks', 'geomean_cons']
method='QuantReg'
clf, mean_std = models.test_regression(result_tab, x_vars, method=method, **opt, plot_args=dict(facet=None))

X_all_norm = models.normalize_variables(result_tab[x_vars], mean_std)
y_pred = clf.predict(X_all_norm)
X = result_tab.copy()
X['pred'] = y_pred

x = 'mean_consum'
plot = graphs.plotting(X, x='mean_consum', y=y, y_pred='pred', graph_name=graph_name, smooth=False, facet=None)
theme_params = {
    'axis.text.x': ggplot2.element_text(size=10, hjust=0),
    'axis.text.y': ggplot2.element_text(size=10),
    'axis.title': ggplot2.element_text(size=10, face="bold")
}
plot += ggplot2.labs(x='Average consumption in flight hours per period', y='Average distance between maintenances')
plot += ggplot2.theme(**theme_params)
graph_name = '{}_mean_consum_{}_{}'.format(method, 'lower', y)
path_out = sto_params.path_graphs + r'{1}/{0}_{1}.png'.format(graph_name, name)
plot.save(path_out)
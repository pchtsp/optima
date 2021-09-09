import stochastic.graphs as graphs
import stochastic.params as sto_params
import scripts.stochastic_analysis as sto_funcs

import pandas as pd

import rpy2.robjects.lib.ggplot2 as ggplot2

batch = sto_funcs.get_batch()
result_tab = sto_funcs.get_table_semi_treated(batch)
result_tab = sto_funcs.treat_table(result_tab)
table = result_tab[(result_tab.gap_abs < 100) & (result_tab.num_errors == 0)]


##################
# Number of maintenances
##################
def draw_maintenances():
    y = 'maints'
    x = 'mean_consum'
    table1 = result_tab.copy().reset_index()
    labels = ['Init: {}/3'.format(p) for p in range(1, 4)]
    table1['init_cut_pretty'] = pd.qcut(table1.init, q=3, duplicates='drop', labels=labels)
    plot = graphs.plotting(table1, x=x, y=y, graph_name='', smooth=False, save=False,
                           facet='init_cut_pretty ~ .')
    theme_params = {
        # angle = 90,
        'axis.text.x': ggplot2.element_text(size=10, hjust=0),
        'axis.text.y': ggplot2.element_text(size=10),
        'axis.title': ggplot2.element_text(size=10, face="bold"),
        'strip.text.y': ggplot2.element_text(angle=0)
    }
    plot += ggplot2.labs(x='Average consumption in flight hours per period', y='Total number of maintenances')
    plot += ggplot2.theme(**theme_params)
    graph_name = 'NPS_article_{}_vs_{}_nocolor'.format(x, y)
    path_out = sto_params.path_graphs + r'{1}/{0}_{1}.png'.format(graph_name, sto_params.name)
    plot.save(path_out)

##################
# QuantReg!
##################

def draw_quantiles():
    # from rpy2.robjects.packages import importr
    # from rpy2.robjects.vectors import StrVector
    # l2p = importr('latex2exp')
    labels1 = dict(x='Sum of all remaining flight hours at the beginning of first period', y='Average distance between maintenances')
    labels2 = dict(x='Sum of all remaining flight hours at the beginning of first period', y='Total numbar of maintenances')
    options = [dict(q=0.9, bound='upper', y_var='maints', method='QuantReg', max_iter = 10000, test_perc= 0.3, labels=labels2),
               dict(q=0.9, bound='upper', y_var='mean_dist_complete', method='QuantReg', max_iter = 10000, test_perc= 0.3, labels=labels1)
               ]
    options[0]['plot_args'] = dict(facet='init_cut ~ geomean_cons_cut', x='mean_consum', y=options[0]['y_var'])
    options[1]['plot_args'] = dict(facet='mean_consum_cut ~ geomean_cons_cut', x='init', y=options[1]['y_var'])
    for opt in options:
        y_pred = sto_funcs.predict_from_table(table, opt)
        X = table.copy()
        labels = ['mu_{{WC}}: {}/3'.format(p) for p in range(1, 4)]
        X['geomean_cons_cut'] = pd.qcut(X['geomean_cons'], q=3, duplicates='drop', labels=labels)
        # values_nn = StrVector(X['geomean_cons_cut'])
        # X['geomean_cons_cut'] = l2p.TeX(values_nn)
        labels = ['mu_{{C}}: {}/3'.format(p) for p in range(1, 4)]
        X['mean_consum_cut'] = pd.qcut(X['mean_consum'], q=3, duplicates='drop', labels=labels)
        X['pred'] = y_pred
        plot = graphs.plotting(X, **opt['plot_args'], y_pred='pred', graph_name='', smooth=False, save=False)
        theme_params = {
            'axis.text.x': ggplot2.element_text(size=10, hjust=0),
            'axis.text.y': ggplot2.element_text(size=10),
            'axis.title': ggplot2.element_text(size=10, face="bold"),
            'strip.text.y': ggplot2.element_text(angle = 0)
        }
        # ggplot2.label_parsed
        plot += ggplot2.labs(**opt['labels'])
        plot += ggplot2.theme(**theme_params)
        graph_name = 'NPS_article_{}_{}_{}_{}'.format(opt['method'], opt['plot_args']['x'], opt['bound'], opt['y_var'])
        path_out = sto_params.path_graphs + r'{1}/{0}_{1}.png'.format(graph_name, sto_params.name)
        plot.save(path_out)


if __name__ == '__main__':
    draw_quantiles()

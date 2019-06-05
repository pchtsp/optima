from rpy2.robjects import pandas2ri
import rpy2.robjects.lib.ggplot2 as ggplot2
import rpy2.robjects as ro

import pandas as pd
import pytups.superdict as sd
import numpy as np

import stochastic.solution_stats as sol_status
import stochastic.params as params

pandas2ri.activate()
path_graphs = params.path_graphs
name = params.name

#####################
# histograms
#####################

def draw_hist(var='maints', bar=True):
    if bar:
        _func = ggplot2.geom_bar(position="identity")
    else:
        _func = ggplot2.geom_histogram(position="identity")
    plot = ggplot2.ggplot(result_tab) + \
           ggplot2.aes_string(x=var) + \
           _func + \
           ggplot2.theme_minimal()

    path_out = path_graphs + r'hist_{}_{}.png'.format(var, name)
    plot.save(path_out)


def hist_no_agg(basenames, cases):
    # basenames= 1
    var = 'sec_maint'
    cases_dict = sd.SuperDict(zip(basenames, cases))
    tt = cases_dict.clean(func=lambda v: v is not None).\
        vapply(sol_status.get_last_maint_date)
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
def plotting(table, graph_name, facet='init_cut ~ .', y_pred=None, smooth=True, jitter=True, **kwargs):
    # x='mean_consum', y=y,
    plot = ggplot2.ggplot(table) + \
           ggplot2.aes_string(**kwargs) + \
           ggplot2.theme_minimal()

    if jitter:
        plot += ggplot2.geom_jitter(alpha=0.8, height=0.2)
    else:
        plot += ggplot2.geom_point(alpha=0.8, height=0.2)

    if facet:
        plot += ggplot2.facet_grid(ro.Formula(facet))

    if smooth:
        plot += ggplot2.geom_smooth(method = 'loess')

    if y_pred:
        plot += ggplot2.geom_line(ggplot2.aes_string(y=y_pred), color='blue')

    path_out = path_graphs + r'{}_{}.png'.format(graph_name, name)
    plot.save(path_out)



def test3(var = 'cons_min_max_vs_maints'):
    plot = ggplot2.ggplot(result_tab) + \
           ggplot2.aes_string(x='cons_min_max', y='maints') + \
           ggplot2.geom_jitter(alpha=0.8, height=0.1) + \
           ggplot2.geom_smooth(method = 'loess') + \
           ggplot2.theme_minimal()

    path_out = path_graphs + r'{}_{}.png'.format(var, name)
    plot.save(path_out)


def test4(var = 'mean_dist_vs_maints'):
    plot = ggplot2.ggplot(result_tab) + \
           ggplot2.aes_string(x='mean_dist', y='maints') + \
           ggplot2.geom_jitter(alpha=0.8, height=0.1) + \
           ggplot2.theme_minimal()

    path_out = path_graphs + r'{}_{}.png'.format(var, name)
    plot.save(path_out)


def test6(var = 'cons_min_max_vs_max_dist_by_pos_consum'):
    plot = ggplot2.ggplot(result_tab) + \
           ggplot2.aes_string(x='cons_min_max', y='max_dist') + \
           ggplot2.geom_jitter(alpha=0.8, height=0.1) + \
           ggplot2.facet_grid(ro.Formula('pos_consum9_cut ~ .'))+\
           ggplot2.theme_minimal()

    path_out = path_graphs + r'{}_{}.png'.format(var, name)
    plot.save(path_out)


def cycle_sizes(cases):
    cycles =[sol_status.get_1M_2M_dist(case) for case in cases if case is not None]
    all_cycles = np.asarray(cycles).flatten()

    plot = ggplot2.ggplot(pd.DataFrame(all_cycles, columns=['cycle'])) + \
           ggplot2.aes_string(x='cycle') + \
           ggplot2.geom_bar(position="identity") + \
           ggplot2.theme_minimal()

    path_out = path_graphs + r'hist_{}_{}.png'.format('all_cycles', name)
    plot.save(path_out)


if __name__ == '__main__':
    result_tab= None
    # some more graphs.
    var = 'cycle_2M_min'
    table = result_tab
    table = table[table.mean_consum.between(150, 300)]
    plot = ggplot2.ggplot(table) + \
           ggplot2.aes_string(x='mean_consum', y=var) + \
           ggplot2.geom_jitter(alpha=0.8, height=0.2) + \
           ggplot2.geom_smooth() + \
           ggplot2.facet_grid(ro.Formula('init_cut ~ spec_tasks_cut')) + \
           ggplot2.theme_minimal()
    path_out = path_graphs + r'mean_consum_init_vs_{}_{}_spectasks_nocolor.png'.format(var, name)
    plot.save(path_out)

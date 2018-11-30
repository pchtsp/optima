import scripts.graphs2 as gp
import os
import package.experiment as exp
import pandas as pd
import package.params as pm
import package.superdict as sd
import scripts.names as na
import dfply as dp
from dfply import X
from ggplot import *

path = '/home/pchtsp/Documents/projects/COR2019/'
experiment = "clust1_20181107"
# experiment2 = "clust1_20181107"
# experiment = "hp_20181102"
def test1():
    table = gp.get_simulation_results(experiment)
    # table2 = get_simulation_results(experiment2)
    # table = table.append(table2)
    table_summary = gp.summary_table(table)
# nsp = na.names_no_spaces()
# names_fr = na.simulation_params_fr()
# nsp_fr = {v: names_fr[k] for k, v in nsp.items()}
# pd.DataFrame.from_dict(nsp, orient='index')
# vars = ['minavailpercent', 'minavailvalue', 'minhoursperc', 'minusageperiod', 'numparalleltasks', 'numperiod',
#         'tminassign']

# sd.SuperDict(nsp_fr).filter(vars)

# plot = gp.boxplot_instances(table >> dp.filter_by(~X.inf))
# plot = gp.boxplot_instances(table >> dp.filter_by(~X.inf), column='gap_out')
# plot.save(path + 'img/' + experiment1 + '_gap.png')
# print(plot)

# name = experiment1 + '_' + experiment
    name = experiment
    gp.summary_to_latex(experiment=name, table=table_summary, path=path + 'tables/')

############################ TEST:

def test2():

    path_exps = pm.PATHS['results'] + experiment
    exps = {p: os.path.join(path_exps, p) + '/' for p in os.listdir(path_exps)}

    # results_list = {k: get_results_table(v, get_exp_info=False) for k, v in exps.items()}
    exps = {k: exp.list_experiments(path_abs, get_exp_info=False) for k, path_abs in exps.items()}
    exps = sd.SuperDict.from_dict(exps)


    inf = {k: v.get_property('status') for k, v in exps.items()}
    cut_times = {k: v.get_property('cut_info').get_property('time') for k, v in exps.items()}
    cuts = {k: v.get_property('cut_info').get_property('cuts') for k, v in exps.items()}

    data = sd.SuperDict(inf).to_dictup().to_tuplist().to_list()
    data = sd.SuperDict(cuts).to_dictup().to_tuplist().to_list()
    data = sd.SuperDict(cut_times).to_dictup().to_tuplist().to_list()

    base_names = ['scenario', 'instance']
    names = base_names + ['status']
    names = base_names + ['cut', 'num']
    names = base_names + ['num']
    table = pd.DataFrame.from_records(data, columns=names).reset_index()

    table_sum  = \
        table >> \
        dp.group_by(X.scenario, X.instance) >> \
        dp.summarize(num = X.num.sum())

    scenarios = table >> dp.distinct(X.scenario) >> dp.select(X.scenario)
    scenarios = scenarios.reset_index(drop=True) >> dp.mutate(code=X.index)

    table_sum = table_sum >> dp.left_join(scenarios, on='scenario')

    ggplot(aes(x='code', y='num'), data=table_sum) + geom_boxplot() + \
    theme(axis_text_x  = element_text("Cut", angle = 45, hjust = 1))

    t = table >> dp.group_by(X.scenario, X.cut) >> dp.summarize(num = X.num.sum())
    t >> \
    dp.spread(X.scenario, X.num) >> \
    dp.select(X.cut, X.base, X.maxelapsedtime_40, X.maxelapsedtime_80, X.elapsedtimesize_40, X.elapsedtimesize_20) >> \
    dp.rename(mt80=X.maxelapsedtime_80, mt40=X.maxelapsedtime_40, es40=X.elapsedtimesize_40, es20=X.elapsedtimesize_20)

    ggplot(aes(x='cut', y='num', color='scenario'), data=t) + geom_bar() + \
    xlab(element_text("Scenario", size=10, vjust=-0.05, angle=100))
    # theme(axis_text_x=element_text("Cut", angle = 45, hjust = 1, size=10))
    # xlab(element_text("Scenario", size=10, vjust=-0.05, angle=100))

    # xlab(element_text("Scenario", size=20, vjust=-0.05, angle=100))
    #


    # ylab(element_text("Solving time (in seconds)", size=20, vjust=0.15))
    #

        # dp.mutate(code = range(X.scenario.n()))

    plot = gp.boxplot_instances(table_sum, column='num_cuts')
    print(plot)
    # pd.io.json.json_normalize(cuts, record_path=['*', '*'])

if __name__ == "__main__":
    test2()
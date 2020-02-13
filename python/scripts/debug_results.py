import scripts.compare_stochastic as compare_sto
import orloge as ol
import pytups as pt
import package.experiment as exp
import zipfile
import package.params as params
import data.data_input as di
import execution.exec  as exec
import os
# import reports.gantt as gantt

def ___():
    exp_list = ['IT000125_20191017', 'IT000125_20190917']
    df = compare_sto.get_df_comparison(exp_list, scenarios=['numparalleltasks_3'], get_progress=False)
    df.first_solution[1]
    df.cut_info[1]
    df[['instance', 'experiment', 'first_relaxed']].head()
    # df = df[df.scenario=='numparalleltasks_2']
    df = df[df.status_code == ol.LpSolutionOptimal]
    df.set_index(['instance', 'experiment'])['best_solution'].unstack('experiment')
    df.best_solution

def run():
    experiment = 'IT000125_20190729'  # new model
    instance_path = 'IT000125_20190729/numparalleltasks_2/201907290916_1'
    zipobj = zipfile.ZipFile(params.PATHS['results'] + experiment + '.zip')
    ddd = zipobj.read(instance_path + '/results.log')
    return ol.get_info_solver(ddd.decode("utf-8"), 'CPLEX', content=True)

data = run()
# we want to debug some specific instances.
"""
              name instance experiment
1 201907290916_102     8103       base
2 201907290916_267     8268       base
3 201907290917_197     8778       base
4 201910250956_102     8103       cuts
5 201910250956_267     8268       cuts
6 201910250956_777     8778       cuts
"""
experiment = 'IT000125_20191025_2'  # old model
experiment = 'IT000125_20190729'  # new model
zipobj = zipfile.ZipFile(params.PATHS['results'] + experiment + '.zip')
instance_path = 'IT000125_20191025_2/numparalleltasks_2/201910250956_432'
instance_path = 'IT000125_20190729/numparalleltasks_2/201907290916_5'
# instance_path = params.PATHS['results'] + r'DESKTOP-9NAIFBG_20191030\numparalleltasks_1\201910300840_6'
case = exp.Experiment.from_zipfile(zipobj, instance_path)
# gantt.make_gantt_from_experiment(experiment=case)
ddd = zipobj.read(instance_path+'/results.log')
options = di.load_data_zip(zipobj, instance_path + '/options.json')
options = di.load_data(instance_path + '/options.json')
path_remake = os.path.join(params.PATHS['experiments'], 'test_remake/')
path_remake2 = os.path.join(params.PATHS['experiments'], 'test_remake2/')
new_options = dict(timeLimit=60, path=path_remake, mip_start=True,
                   fix_vars= [], solver="Model.CPLEX")
options.update(new_options)
case = exp.Experiment.from_dir(instance_path)
case2 = exp.Experiment.from_dir(path_remake2)
maints2 = case2.get_all_maintenance_cycles().to_tuplist().to_set()
maints1 = case.get_all_maintenance_cycles().to_tuplist().to_set()
# maints1 ^ maints2
import reports.gantt as gantt
gantt.make_gantt_from_experiment(experiment=case)
gantt.make_gantt_from_experiment(experiment=case2)
exec.execute_solve(case.instance.data, options, case.solution.data)
# anor = md_anor.ModelANOR(case.instance, case.solution)

# solution = model.solve(options)

case.check_solution()
#
# ll = zipobj.namelist()
#
# ll.index('IT000125_20191025_2/numparalleltasks_2/2019')
# pt.TupList(ll).vfilter(lambda v: instance_path in v)
ddd = zipobj.read(instance_path+'/results.log')
with open('results.log', 'wb') as f:
    f.write(ddd)




# import data.data_input as di
# di.load_data_zip(zipobj, files[0])
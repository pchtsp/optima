import data.data_input as di
import solvers.model as md
import package.experiment as exp
import os

path_root = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/"
path_in = path_root + "MOSIM2018/"
path_out = path_root + "experiments/"

weight_options = [round(0.1 * value, 1) for value in range(11)]
experiments = \
    ['201801141705', '201801141331', '201801141334', '201801141706',
     '201801141646', '201801131813', '201801102127', '201801131817',
     '201801141607', '201801102259']
experiments = \
    ['201801141705', '201801141334', '201801141706',
     '201801131813', '201801131817', '201801102259']
paths = {k: path_in + k for k in experiments}
instances = {k: exp.Experiment.from_dir(v).instance for k, v in paths.items()}
options_all = {k: di.load_data(v + '/options.json') for k, v in paths.items()}


def compare_two_weights(instance, def_options, weight_options):

    for pos, w in enumerate(weight_options):
        name = str(int(w * 10))
        instance.data["parameters"]['maint_weight'] = w
        instance.data["parameters"]['unavail_weight'] = 1 - w
        options = dict(def_options)
        options["weights"] = {'maint_weight': w, 'unavail_weight': 1 - w}
        options["path"] += name + '/'
        di.export_data(options['path'], instance.data, name="data_in", file_type='json')
        di.export_data(options['path'], options, name="options", file_type='json')

        # solving part:
        solution = md.solve_model(instance, options)
        if solution is not None:
            di.export_data(options['path'], solution.data, name="data_out", file_type='json')


def test_all_multiobjective():

    for k, v in options_all.items():
        v['path'] = path_out + 'weights_all/' + k + '/'
        if not os.path.exists(v['path']):
            os.mkdir(v['path'])

    # instance = instances['201801141607']
    # option = options['201801141607']

    for _id in experiments:
        # if _id != '201801141607':
        instance = instances[_id]
        options = options_all[_id]
        compare_two_weights(instance, options, weight_options)


def test_all_solver_change():

    for k, v in options_all.items():
        v['path'] = path_out + 'CPLEX128/' + k + '/'
        v['comment'] = 'CPLEX 12.8'
        if not os.path.exists(v['path']):
            os.mkdir(v['path'])

    for _id in instances:
        options = options_all[_id]
        instance = instances[_id]
        di.export_data(options['path'], instance.data, name="data_in", file_type='json')
        di.export_data(options['path'], options, name="options", file_type='json')

        solution = md.solve_model(instance, options)
        if solution is not None:
            di.export_data(options['path'], solution.data, name="data_out", file_type='json')



if __name__ == "__main__":
    # test_all_multiobjective()
    test_all_solver_change()

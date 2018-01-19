import package.aux as aux
import package.data_input as di
import package.model as md
import package.instance as inst
import package.tests as exp
import os


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
        solution = md.model_no_states(instance, options)
        if solution is not None:
            di.export_data(options['path'], solution.data, name="data_out", file_type='json')


if __name__ == "__main__":

    path_abs = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/"
    weight_options = [round(0.1 * value, 1) for value in range(11)]

    experiments =\
        ['201801141705', '201801141331', '201801141334', '201801141706',
         '201801141646', '201801131813', '201801102127', '201801131817',
         '201801141607', '201801102259']

    paths = {k: path_abs + k for k in experiments}

    instances = {k: exp.Experiment.from_dir(v).instance for k, v in paths.items()}
    options = {k: di.load_data(v + '/options.json') for k, v in paths.items()}
    for k, v in options.items():
        v['path'] = path_abs + 'weights_all/' + k + '/'
        if not os.path.exists(v['path']):
            os.mkdir(v['path'])

    # instance = instances['201801141607']
    # option = options['201801141607']

    for _id in experiments:
        if _id != '201801141607':
            instance = instances[_id]
            option = options[_id]
            compare_two_weights(instance, option, weight_options)

if __name__ == "__main__AAAA":

    # instance = inst.Instance(model_data)

    def_options = {
        'timeLimit': 3600
        , 'gap': 0
        , 'solver': "CPLEX"
        , 'path':
            '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/weights3/'
        , "model": "no_states"
        , "timestamp": aux.get_timestamp()
        # , "comments": "periods 0 to 30 without tasks: O10, O8"
    }


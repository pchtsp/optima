import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import package.auxiliar as aux
import package.data_input as di
import package.instance as inst
import package.model as md
import package.model_cp as md_cp
import package.experiment as exp
import importlib
import argparse
import package.heuristics as heur
import package.simulation as sim
import package.superdict as sd
import datetime as dt


def config_and_solve(params):

    options = params.OPTIONS
    if options.get('simulate', False):
        model_data = sim.create_dataset(options)
    else:
        model_data = di.get_model_data(params.PATHS['input'])
        historic_data = di.generate_solution_from_source(params.PATHS['hist'])
        model_data = di.combine_data_states(model_data, historic_data)
        model_data['parameters']['start'] = options['start']
        model_data['parameters']['end'] = \
            aux.shift_month(model_data['parameters']['start'], options['num_period'] - 1)

    white_list = options.get('white_list', [])
    black_list = options.get('black_list', [])

    tasks = model_data['tasks']
    if len(black_list) > 0:
        tasks = {k: v for k, v in model_data['tasks'].items() if k not in black_list}
    if len(white_list) > 0:
        tasks = {k: v for k, v in model_data['tasks'].items() if k in white_list}

    model_data['tasks'] = tasks

    execute_solve(model_data, options)


def re_execute_instance(directory, new_options=None):

    model_data = di.load_data(os.path.join(directory, 'data_in.json'))
    options = di.load_data(os.path.join(directory, 'options.json'))
    if new_options is not None:
        options.update(new_options)
    execute_solve(model_data, options)


def execute_solve(model_data, options):
    instance = inst.Instance(model_data)

    output_path = options['path']

    di.export_data(output_path, instance.data, name="data_in", file_type='json')
    di.export_data(output_path, options, name="options", file_type='json')

    # solving part:
    solver = options.get('solver', 'CPLEX')
    if solver == 'CPO':
        solution = md_cp.solve_model2(instance, options)
    elif solver == 'HEUR':
        heur_obj = heur.GreedyByMission(instance)
        solution = heur_obj.solve(options)
    else:
        solution = md.solve_model(instance, options)

    if solution is None:
        return None

    experiment = exp.Experiment(instance, solution)
    errors = experiment.check_solution()
    errors = sd.SuperDict.from_dict(errors)
    errors = {k: v.to_dictdict() for k, v in errors.items()}

    di.export_data(output_path, experiment.solution.data, name="data_out", file_type='json')
    if len(errors):
        di.export_data(output_path, errors, name='errors', file_type="json")


if __name__ == "__main__":

    import json
    parser = argparse.ArgumentParser(description='Solve an instance MFMP.')
    parser.add_argument('-c', dest='file', default="package.params",
                        help='config file (default: package.params)')
    parser.add_argument('-d', '--options', dest='config_dict', type=json.loads)
    parser.add_argument('-p', '--paths', dest='paths_dict', type=json.loads)

    args = parser.parse_args()
    # if not os.path.exists(args.file):
    #     raise FileNotFoundError("{} was not found".format(args.file))

    print('Using config file in {}'.format(args.file))
    params = importlib.import_module(args.file)
    if args.config_dict:
        params.OPTIONS.update(args.config_dict)

    if args.paths_dict:
        params.PATHS.update(args.paths_dict)
        params.OPTIONS['path'] = \
            os.path.join(params.PATHS['experiments'], dt.datetime.now().strftime("%Y%m%d%H%M") ) + '/'

    # import package.params as params
    config_and_solve(params)
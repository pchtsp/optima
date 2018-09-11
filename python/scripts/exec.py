import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import package.auxiliar as aux
import package.data_input as di
import package.instance as inst
import package.model as md
import package.model_cp as md_cp
import importlib
import argparse
import package.heuristics as heur
import package.simulation as sim


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
    if solution is not None:
        di.export_data(output_path, solution.data, name="data_out", file_type='json')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solve an instance MFMP.')
    parser.add_argument('-c', dest='file', default="package.params",
                        help='config file (default: package.params)')

    args = parser.parse_args()
    # if not os.path.exists(args.file):
    #     raise FileNotFoundError("{} was not found".format(args.file))

    print('Using config file in {}'.format(args.file))
    params = importlib.import_module(args.file)
    # import package.params as params
    config_and_solve(params)

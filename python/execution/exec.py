import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import data.dates as aux
import data.data_input as di
import package.instance as inst
import package.solution as sol
import package.experiment as exp
import data.simulation as sim
import pytups.superdict as sd

import datetime as dt
import importlib
import argparse


def config_and_solve(options):

    # options = params.OPTIONS
    if options.get('simulate', False):
        model_data = sim.create_dataset(options)
    else:
        model_data = di.get_model_data(options['PATHS']['input'])
        historic_data = di.generate_solution_from_source(options['PATHS']['hist'])
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


def re_execute_instance_errors(directory, new_options=None, **kwags):
    destination = directory
    if new_options is not None:
        if 'path' in new_options:
            destination = new_options['path']
    try:
        print('path is : {}'.format(directory))
        re_execute_instance(directory, new_options=new_options, **kwags)
    except Exception as e:
        str_fail = "Unexpected error in case: \n{}".format(repr(e))
        path_out = os.path.join(destination, 'failure.txt')
        with open(path_out, 'w') as f:
            f.write(str_fail)


def re_execute_instance(directory, new_options=None, new_input=None):
    if not os.path.exists(directory):
        raise ValueError('path {} does not exist'.format(directory))
    model_data = di.load_data(os.path.join(directory, 'data_in.json'))
    model_data = sd.SuperDict.from_dict(model_data)
    if new_input is not None:
        model_data.update(new_input)
    solution_data = None
    options = di.load_data(os.path.join(directory, 'options.json'))
    options = sd.SuperDict.from_dict(options)
    if new_options is not None:
        options.update(new_options)
    warm_start = options.get('mip_start', False)
    if warm_start:
        solution_path = os.path.join(directory, 'data_out.json')
        if os.path.exists(solution_path):
            solution_data = di.load_data(solution_path)
    execute_solve(model_data, options, solution_data)


def execute_solve(model_data, options, solution_data=None):

    import solvers.model as md

    instance = inst.Instance(model_data)
    solution = None

    if solution_data is not None:
        solution = sol.Solution(solution_data)

    exclude_aux = options.get('exclude_aux', True)
    output_path = options['path']
    di.export_data(output_path, instance.data, name="data_in", file_type='json', exclude_aux=exclude_aux)
    di.export_data(output_path, options, name="options", file_type='json')

    # solving part:
    engine = options.get('solver', 'CPLEX')
    # there is the possibilty to have two solvers separated by .
    solver = None
    if '.' in engine:
        engine, solver = engine.split('.')
        options['solver'] = solver
    if engine == 'CPO':
        raise NotImplementedError("The CPO model is not supported for the time being")
    if engine == 'HEUR':
        import solvers.heuristics as heur
        experiment = heur.GreedyByMission(instance, solution=solution)
    elif engine == 'HEUR_mf':
        import solvers.heuristics_maintfirst as mf
        experiment = mf.MaintenanceFirst(instance, solution=solution)
        if solver is not None:
            # if we added a solver, we want a two-step solving
            solution = experiment.solve(options)
            experiment = md.Model(instance, solution=solution)
            options.update(dict(mip_start= True))
    elif engine == 'FixLP':
        import solvers.model_fixingLP as FixLP
        experiment = FixLP.ModelFixLP(instance, solution=solution)
    elif engine == 'FlexFixLP':
        import solvers.model_fixing_flexibleLP as FlexFixLP
        experiment = FlexFixLP.ModelFixFlexLP(instance, solution=solution)
    else:
        # model with solver
        experiment = md.Model(instance, solution=solution)

    solution = experiment.solve(options)

    if solution is None:
        return None

    experiment = exp.Experiment(instance, solution)
    errors = experiment.check_solution()
    errors = {k: v.to_dictdict() for k, v in errors.items()}

    di.export_data(output_path, experiment.solution.data, name="data_out", file_type='json', exclude_aux=exclude_aux)
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

    options = params.OPTIONS
    options['PATHS'] = params.PATHS
    # import package.params as params
    config_and_solve(options)
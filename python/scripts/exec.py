import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import importlib
import argparse
import datetime as dt
import json
import package.superdict as sd
import package.auxiliar as aux
import package.data_input as di
import package.instance as inst
import package.solution as sol
import package.model as md
import package.experiment as exp
import package.heuristics as heur
import package.heuristics_maintfirst as mf
import package.simulation as sim
import package.template_data as td


def config_and_solve(options):

    # options = params.OPTIONS
    if options.get('simulate', False):
        model_data = sim.create_dataset(options)
    elif options.get('template', False):
        model_data = td.import_input_template(options['input_template_path'])
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

    if options.get('R_HOME'):
        os.environ['R_HOME'] = options.get('R_HOME')

    execute_solve(model_data, options)


def re_execute_instance(directory, new_options=None):

    model_data = di.load_data(os.path.join(directory, 'data_in.json'))
    solution_data = None
    options = di.load_data(os.path.join(directory, 'options.json'))
    if new_options is not None:
        options.update(new_options)
    warm_start = options.get('mip_start', False)
    if warm_start:
        solution_data = di.load_data(os.path.join(directory, 'data_out.json'))
    # print(options)
    execute_solve(model_data, options, solution_data)


def execute_solve(model_data, options, solution_data=None):
    instance = inst.Instance(model_data)
    solution = None

    if solution_data is not None:
        solution = sol.Solution(solution_data)

    output_path = options['path']
    # print(output_path)
    di.export_data(output_path, instance.data, name="data_in", file_type='json', exclude_aux=True)
    di.export_data(output_path, options, name="options", file_type='json')

    # solving part:
    solver = options.get('solver', 'CPLEX')
    if solver == 'CPO':
        raise("The CPO model is not supported for the time being")
    if solver == 'HEUR':
        experiment = heur.GreedyByMission(instance, solution=solution)
    elif solver == 'HEUR_mf':
        experiment = mf.MaintenanceFirst(instance, solution=solution)
    elif solver == 'HEUR_mf_CPLEX':
        experiment = mf.MaintenanceFirst(instance, solution=solution)
        solution = experiment.solve(options)
        experiment = md.Model(instance, solution=solution)
        options.update(dict(mip_start= True, solver='CPLEX'))
    else:
        # model with solver
        experiment = md.Model(instance, solution=solution)

    solution = experiment.solve(options)

    if solution is None:
        return None

    experiment = exp.Experiment(instance, solution)
    errors = experiment.check_solution()
    errors = {k: v.to_dictdict() for k, v in errors.items()}

    di.export_data(output_path, experiment.solution.data, name="data_out", file_type='json', exclude_aux=True)
    if len(errors):
        di.export_data(output_path, errors, name='errors', file_type="json")

    if options.get('template', False):
        td.export_output_template(options['output_template_path'], experiment.solution.data)
        input_path = options['input_template_path']
        # if it doesnt exist: we also export the input
        if not os.path.exists(input_path):
            td.export_input_template(input_path, experiment.instance.data)

    if options.get('graph', False):
        try:
            import package.rpy_graphs as rg
            possible_path = options['root'] + 'R/functions/import_results.R'
            # print('possible path for script: {}'.format(possible_path))
            # os.listdir(options['root'] + 'python/')
            # os.listdir(options['root'])
            if os.path.exists(possible_path):
                # print('file exists')
                rg.gantt_experiment(options['path'], possible_path)
            else:
                # print('file doesnt exists')
                rg.gantt_experiment(options['path'])
        except:
            print("No support for R graph functions!")


def update_case_path(options, path):
    options['path'] = path
    options['input_template_path'] = path + 'template_in.xlsx'
    options['output_template_path'] = path + 'template_out.xlsx'
    return options


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Solve an instance MFMP.')
    parser.add_argument('-c', dest='file', default="package.params",
                        help='config file (default: package.params)')
    parser.add_argument('-d', '--options', dest='config_dict', type=json.loads)
    parser.add_argument('-df', '--options-file', dest='config_file')
    parser.add_argument('-p', '--paths', dest='paths_dict', type=json.loads)
    parser.add_argument('-it', '--input-template', dest='input_template')
    parser.add_argument('-id', '--input-template-dir', dest='input_template_dir')

    args = parser.parse_args()
    print('Using config file in {}'.format(args.file))
    params = importlib.import_module(args.file)

    args = parser.parse_args()
    if getattr(sys, 'frozen', False):
        # we are running in a bundle
        root = sys._MEIPASS + '/'
    else:
        # we are running in a normal Python environment
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/'

    params.OPTIONS['root'] = root

    new_options = None
    if args.config_dict:
        new_options = args.config_dict
    elif args.config_file:
        new_options = di.load_data(args.config_file)
    if new_options:
        params.OPTIONS.update(new_options)

    if args.paths_dict:
        params.PATHS.update(args.paths_dict)
        path = os.path.join(params.PATHS['experiments'], dt.datetime.now().strftime("%Y%m%d%H%M")) + '/'
        update_case_path(params.OPTIONS, path)

    if args.input_template_dir:
        path = args.input_template_dir
        update_case_path(params.OPTIONS, path)
        possible_option_path = path + 'options_in.json'
        if os.path.exists(possible_option_path):
            new_options = di.load_data(possible_option_path)
            params.OPTIONS.update(new_options)

    if args.input_template:
        params.OPTIONS['input_template_path'] = args.input_template

    options = params.OPTIONS
    options['PATHS'] = params.PATHS
    config_and_solve(options)
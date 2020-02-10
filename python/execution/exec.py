import os, sys
import subprocess
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import data.dates as aux
import data.data_input as di
import data.data_dga as dga
import data.template_data as td

import package.instance as inst
import package.solution as sol
import package.experiment as exp
import data.simulation as sim
import pytups.superdict as sd


def config_and_solve(options):

    if options.get('simulate', False):
        model_data = sim.create_dataset(options)
    elif options.get('template', False):
        model_data = td.import_input_template(options['input_template_path'])
    else:
        model_data = dga.get_model_data(options['PATHS']['input'])
        historic_data = dga.generate_solution_from_source(options['PATHS']['hist'])
        model_data = dga.combine_data_states(model_data, historic_data)
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


def engine_factory(engine):
    if engine == 'CPO':
        raise NotImplementedError("The CPO model is not supported for the time being")
    elif engine == 'HEUR':
        import solvers.heuristics as heur
        return heur.GreedyByMission
    elif engine == 'HEUR_mf':
        import solvers.heuristics_maintfirst as mf
        return mf.MaintenanceFirst
    elif engine == 'FixLP':
        import solvers.model_fixingLP as model
        return model.ModelFixLP
    elif engine == 'FlexFixLP':
        import solvers.model_fixingLP as model
        return model.ModelFixFlexLP
    elif engine == 'FlexFixLP_3':
        import solvers.model_fixingLP as model
        return model.ModelFixFlexLP_3
    elif engine == 'ModelANOR':
        import solvers.model_anor as model_anor
        return model_anor.ModelANOR
    else:
        import solvers.model as md
        return md.Model


def execute_solve(model_data, options, solution_data=None):

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
    if '.' in engine:
        engine, solver = engine.split('.')
        options['solver'] = solver

    engine_obj = engine_factory(engine)
    experiment = engine_obj(instance, solution=solution)
    solution = experiment.solve(options)

    if solution is None:
        return None

    experiment = exp.Experiment(instance, solution)
    errors = experiment.check_solution()
    errors = {k: v.to_dictdict() for k, v in errors.items()}

    exclude_aux = options.get('exclude_aux', True)
    di.export_data(output_path, experiment.solution.data, name="data_out", file_type='json', exclude_aux=exclude_aux)
    if len(errors):
        di.export_data(output_path, errors, name='errors', file_type="json")

    if options.get('template', False):
        td.export_output_template(options['output_template_path'], experiment)
        input_path = options['input_template_path']
        # if it doesnt exist: we also export the input
        if not os.path.exists(input_path):
            td.export_input_template(input_path, experiment.instance.data)


def copy_file_temp(path):
    if not os.path.exists(path):
        return None
    path_dir = os.path.join(os.environ['TEMP'], 'OPA')
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    _file_name = os.path.basename(path)
    filename = os.path.join(path_dir, _file_name)
    with open(path) as f:
        with open(filename, "w") as f1:
            for line in f:
                f1.write(line)
    return filename


def update_case_path(options, path):
    options['path'] = path
    options['input_template_path'] = path + 'template_in.xlsx'
    options['output_template_path'] = path + 'template_out.xlsx'
    return options


def udpdate_case_read_options(options, path):
    possible_option_path = path + 'options_in.json'
    if os.path.exists(possible_option_path):
        new_options = di.load_data(possible_option_path)
        options.update(new_options)
    update_case_path(options, path)
    return options

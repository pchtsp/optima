import os, sys
import subprocess
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import package.auxiliar as aux
import data.data_input as di
import data.data_dga as dga
import package.instance as inst
import package.solution as sol
import solvers.model as md
import package.experiment as exp
import solvers.heuristics as heur
import solvers.heuristics_maintfirst as mf
import package.simulation as sim
import data.template_data as td


def config_and_solve(options):

    # options = params.OPTIONS
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

    if options.get('R_HOME'):
        os.environ['R_HOME'] = options.get('R_HOME')

    execute_solve(model_data, options)


def re_execute_instance(directory, new_options=None):

    model_data = di.load_data(os.path.join(directory, 'data_in.json'))
    solution_data = None
    options = di.load_data(os.path.join(directory, 'options.json'))
    if new_options is not None:
        options.update(new_options)
    warm_start = options.get('warm_start', False)
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
        options.update(dict(warm_start= True, solver='CPLEX'))
    else:
        # model with solver
        experiment = md.Model(instance, solution=solution)

    if options.get('solve', True):
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
        td.export_output_template(options['output_template_path'], experiment.instance.data, experiment.solution.data)
        input_path = options['input_template_path']
        # if it doesnt exist: we also export the input
        if not os.path.exists(input_path):
            td.export_input_template(input_path, experiment.instance.data)

    possible_path = options['root'] + 'R/functions/import_results.R'
    if options.get('graph', False) == 1:
        import reports.gantt as gantt
        gantt.make_gantt_from_experiment(path=options['path'])
        # try:
            # import reports.rpy_graphs as rg
            # print('possible path for script: {}'.format(possible_path))
            # os.listdir(options['root'] + 'python/')
            # os.listdir(options['root'])
            # if os.path.exists(possible_path):
                # print('file exists')
                # rg.gantt_experiment(options['path'], possible_path)
            # else:
                # print('file doesnt exists')
                # rg.gantt_experiment(options['path'])
        # except:
        #     print("No support for R graph functions!")
    elif options.get('graph', False) == 2:
        _file = copy_file_temp(possible_path)
        rscript = options['R_HOME'] + '/bin/Rscript.exe'
        a = subprocess.run([rscript, _file, options['path']], stdout=subprocess.PIPE)


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
import package.auxiliar as aux
import package.data_input as di
import package.instance as inst
import package.model as md
import copy
# import package.model as md
# import multiprocessing as multi
# import datetime
import package.tests as Exp


if __name__ == "__main__":

    model_data = di.get_model_data()
    historic_data = di.generate_solution_from_source()
    model_data = di.combine_data_states(model_data, historic_data)

    # this is for testing purposes:
    num_start_period = 0
    num_max_periods = 30
    model_data['parameters']['start'] = \
        aux.shift_month(model_data['parameters']['start'], num_start_period)
    model_data['parameters']['end'] = \
        aux.shift_month(model_data['parameters']['start'], num_max_periods)
    # black_list = ['O10', 'O8', 'O6']
    # black_list = ['O10', 'O8']
    black_list = ['O8']  # this task has less candidates than what it asks.
    model_data['tasks'] = \
        {k: v for k, v in model_data['tasks'].items() if k not in black_list}
    # this was for testing purposes

    task_type = aux.get_property_from_dic(model_data['tasks'], 'type_resource')
    type_tasks = {type: [] for type in task_type.values()}
    for task, type in task_type.items():
        type_tasks[type].append(task)

    type_inst = {}
    for type, tasks in type_tasks.items():
        model_data_n = copy.deepcopy(model_data)
        model_data_n['tasks'] =\
            {k: v for k, v in model_data['tasks'].items() if k in tasks}
        type_inst[type] = inst.Instance(model_data_n)

    options_d = {
        'timeLimit': 3600
        , 'gap': 0
        , 'solver': "CPLEX"
        , 'path':
            '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/pieces30/'
        , "model": "no_states"
    }

    for type, instance in type_inst.items():
        options = {**options_d, **{'type': type, 'path': options_d['path']+type+'/'}}
        di.export_data(options['path'], instance.data, name="data_in", file_type='json')
        di.export_data(options['path'], options, name="options", file_type='json')
        # solving part:
        solution = md.solve_model(instance, options)

        if solution is not None:
            di.export_data(options['path'], solution.data, name="data_out", file_type='json')


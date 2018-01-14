import package.aux as aux
import package.data_input as di
import package.instance as inst
import package.model as md
# import package.model as md
# import multiprocessing as multi
# import datetime


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
    forbidden_tasks = ['O10', 'O8', '06']
    # forbidden_tasks = ['O8']  # this task has less candidates than what it asks.
    model_data['tasks'] = \
        {k: v for k, v in model_data['tasks'].items() if k not in forbidden_tasks}
    # this was for testing purposes

    instance = inst.Instance(model_data)

    options = {
        'timeLimit': 3600
        , 'gap': 0
        , 'solver': "CPLEX"
        , 'path':
            '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/{}/'.
                format(aux.get_timestamp())
        , "model": "no_states"
    }

    di.export_data(options['path'], instance.data, name="data_in", file_type='json')
    di.export_data(options['path'], options, name="options", file_type='json')

    # solving part:
    solution = md.model_no_states(instance, options)
    if solution is not None:
        di.export_data(options['path'], solution.data, name="data_out", file_type='json')


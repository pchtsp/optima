import package.aux as aux
import package.data_input as di
import package.instance as inst
import package.model as md
import package.params as params
import os
# import package.model as md
# import multiprocessing as multi
# import datetime


if __name__ == "__main__":

    input_file = params.PATHS['data'] + 'raw/parametres_DGA_final.xlsm'
    model_data = di.get_model_data(input_file)
    historic_data = di.generate_solution_from_source()
    model_data = di.combine_data_states(model_data, historic_data)

    # this is for testing purposes:
    num_start_period = 0
    num_max_periods = 40
    model_data['parameters']['start'] = \
        aux.shift_month(model_data['parameters']['start'], num_start_period)
    model_data['parameters']['end'] = \
        aux.shift_month(model_data['parameters']['start'], num_max_periods)
    # black_list = []
    white_list = []
    # white_list = ['O1', 'O5']
    # black_list = ['O10', 'O8', 'O6']
    # black_list = ['O10', 'O8']
    black_list = ['O8']  # this task has less candidates than what it asks.
    if len(black_list) > 0:
        model_data['tasks'] = \
            {k: v for k, v in model_data['tasks'].items() if k not in black_list}
    if len(white_list) > 0:
        model_data['tasks'] = \
            {k: v for k, v in model_data['tasks'].items() if k in white_list}
    # this was for testing purposes

    instance = inst.Instance(model_data)
    # num_res = instance.get_tasks('num_resource')
    # candidates = aux.dict_filter(instance.get_task_candidates(), white_list)
    # candidates = [c for k, l in candidates.items() for c in l[:num_res[k]+15]]
    # candidates = list(np.unique(candidates))
    # # candidates = list(candidates[:len(candidates)//3])
    # instance.data['resources'] = aux.dict_filter(instance.data['resources'], candidates)
    # instance.data['parameters']['maint_capacity'] /= 3

    options = {
        'timeLimit': 3600
        , 'gap': 0
        , 'solver': "CPLEX"
        , 'path':
            os.path.join(params.PATHS['experiments'], aux.get_timestamp()) + '/'
    }

    di.export_data(options['path'], instance.data, name="data_in", file_type='json')
    di.export_data(options['path'], options, name="options", file_type='json')

    # solving part:
    solution = md.solve_model(instance, options)
    if solution is not None:
        di.export_data(options['path'], solution.data, name="data_out", file_type='json')
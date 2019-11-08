import package.auxiliar as aux
import data.data_input as di
import package.instance as inst
import solvers.model as md
import multiprocessing as multi


def solve_write(instance, options):

    solution = md.solve_model(instance, options)
    if solution is not None:
        di.export_data(options['path'], instance.data, name="data_in", file_type='json')
        di.export_data(options['path'], solution.data, name="data_out", file_type='json')
        di.export_data(options['path'], options, name="options", file_type='json')
    return True


def working(instance, def_options, weight_options):

    pool = multi.Pool(processes=3)

    results = {}

    for pos, w in enumerate(weight_options):
        name = str(int(w * 10))
        instance.data["parameters"]['maint_weight'] = w
        instance.data["parameters"]['unavail_weight'] = 1 - w
        options = dict(def_options)
        options["weights"] = {'maint_weight': w, 'unavail_weight': 1 - w}
        options["path"] += name + '/'
        results[pos] = pool.apply_async(solve_write, [instance, options])

    # solutions = {}
    for pos, result in results.items():
        timeout = def_options['timeLimit']*2 + 500
        result = result.get(timeout=timeout)

    return True

model_data = di.get_model_data()
historic_data = di.generate_solution_from_source()
model_data = di.combine_data_states(model_data, historic_data)

# this is for testing purposes:
num_start_period = 0
num_max_periods = 20
model_data['parameters']['start'] = \
    aux.shift_month(model_data['parameters']['start'], num_start_period)
model_data['parameters']['end'] = \
    aux.shift_month(model_data['parameters']['start'], num_max_periods)
black_list = ['O10', 'O8', '06']
# black_list = ['O8']  # this task has less candidates than what it asks.
model_data['tasks'] = \
    {k: v for k, v in model_data['tasks'].items() if k not in black_list}
# this was for testing purposes

instance = inst.Instance(model_data)

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

# solving part:
weight_options = [round(0.1*value, 1) for value in range(11)]
# weight_options = [0.1]

if __name__ == "__main__":

    # time = datetime.datetime.now()
    # working(instance, def_options, weight_options)
    # duration = datetime.datetime.now() - time
    # print("The time it took = {} seconds".format(duration.seconds))
    pass

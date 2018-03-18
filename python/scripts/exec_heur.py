import package.solution as sol
import package.heuristics as heur
import package.data_input as di
import package.instance as inst
import pprint as pp
import package.tests as exp
import package.params as params
import package.auxiliar as aux


if __name__ == "__main__":

    input_file = params.PATHS['data'] + 'raw/parametres_DGA_final.xlsm'
    model_data = di.get_model_data(input_file)
    historic_data = di.generate_solution_from_source(params.PATHS['data'] + 'raw/Planifs M2000.xlsm')
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
    black_list = ['O10', 'O8', 'O6']
    # black_list = ['O10', 'O8']
    # black_list = ['O8']  # this task has less candidates than what it asks.
    if len(black_list) > 0:
        model_data['tasks'] = \
            {k: v for k, v in model_data['tasks'].items() if k not in black_list}
    if len(white_list) > 0:
        model_data['tasks'] = \
            {k: v for k, v in model_data['tasks'].items() if k in white_list}

    instance = inst.Instance(model_data)
    heur_obj = heur.Greedy(instance)
    heur_obj.solve()
    check = heur_obj.check_solution()
    pp.pprint(check)
    # heur.solution.print_solution("/home/pchtsp/Downloads/calendar_temp1.html")

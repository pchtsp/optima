import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import scripts.exec as exec
import datetime as dt
import package.superdict as sd
import package.simulation as sim
import argparse
import re
import copy
import multiprocessing as multi

def merge_resources(model_data, initial_data):
    model_data = sd.SuperDict.from_dict(model_data)
    initial_data = sd.SuperDict.from_dict(initial_data)
    init_elapsed = initial_data['resources'].get_property('initial_elapsed')
    init_used = initial_data['resources'].get_property('initial_used')

    # of the resources has no previous assignments,
    # we'll get its stats to the "fixed" ones
    res_with_previous_states = model_data['resources'].get_property('states').clean(func=lambda x: len(x)).keys_l()
    for k, v in model_data['resources'].items():
        if k not in res_with_previous_states:
            v['initial_elapsed'] = init_elapsed[k]
            v['initial_used'] = init_used[k]

    return model_data

if __name__ == "__main__":

    import package.params as params
    import json
    parser = argparse.ArgumentParser(description='Solve an instance MFMP.')
    parser.add_argument('-d', '--options', dest='config_dict', type=json.loads)
    parser.add_argument('-p', '--paths', dest='paths_dict', type=json.loads)
    parser.add_argument('-s', '--simulation', dest='sim_dict', type=json.loads)
    parser.add_argument('-q', '--num_instances', dest='num_inst', type=int, required=True)
    parser.add_argument('-c', '--case_options', dest='case_opt', type=json.loads, required=True)

    args = parser.parse_args()

    if args.config_dict:
        params.OPTIONS.update(args.config_dict)

    if args.paths_dict:
        params.PATHS.update(args.paths_dict)

    if args.sim_dict:
        params.OPTIONS['simulation'].update(args.sim_dict)

    if not os.path.exists(params.PATHS['results']):
        os.mkdir(params.PATHS['results'])

    num_instances = args.num_inst

    options = params.OPTIONS = sd.SuperDict(params.OPTIONS)
    sim_data = options['simulation'] = sd.SuperDict(options['simulation'])
    options['PATHS'] = params.PATHS


    case_options = args.case_opt

    case_data = [{k: vv, 'name': '{}_{}'.format(re.sub('_', '', k), vv)}
                 for k, v in case_options.items() for vv in v] + \
                [{'name': 'base'}]

    seed_backup = sim_data['seed']
    multiproc = options.get('multiprocess')
    results = {}
    pos = 0
    if multiproc:
        pool = multi.Pool(processes=multiproc)


    for case in case_data:
        sim_data['seed'] = seed_backup
        path_exp = params.PATHS['experiments'] = \
            params.PATHS['results'] + case['name']
        if not os.path.exists(path_exp):
            os.mkdir(path_exp)

        case_opt = sd.SuperDict(case).filter(options.keys_l(), check=False)
        _options = copy.deepcopy(options)
        _options.update(case_opt)
        case_sim = sd.SuperDict(case).filter(sim_data.keys_l(), check=False)
        _sim_data = _options['simulation']
        _sim_data.update(case_sim)
        # this needs to be enforced so feasible instances can be obtained:
        _sim_data['num_resources'] = 15 * _sim_data['num_parallel_tasks']

        # this data is the reference data for the scenario. Specifically for resources.
        initial_data = sim.create_dataset(options)

        for _ in range(num_instances):
            if _sim_data['seed']:
                _sim_data['seed'] += 1

            _path_instance = path_instance = _options['path'] = \
                os.path.join(path_exp, dt.datetime.now().strftime("%Y%m%d%H%M"))
            num = 1

            while os.path.exists(_path_instance):
                _path_instance = path_instance + "_{}".format(num)
                num += 1
            _options['path'] = path_instance = _path_instance + '/'


            try:
                model_data = sim.create_dataset(_options)
                model_data = merge_resources(model_data, initial_data)
                exec.execute_solve(model_data, _options)
            except Exception as e:
                if not os.path.exists(path_instance):
                    os.mkdir(path_instance)
                str_fail = "Unexpected error in case: \n{}".format(repr(e))
                with open(path_instance + 'failure.txt', 'w') as f:
                    f.write(str_fail)

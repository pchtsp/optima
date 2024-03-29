import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import execution.exec as exec
import datetime as dt
import pytups.superdict as sd
import argparse
import re
import copy
import multiprocessing as multi

if __name__ == "__main__":

    import package.params as params
    import json
    parser = argparse.ArgumentParser(description='Solve an instance MFMP.')
    parser.add_argument('-d', '--options', dest='config_dict', type=json.loads)
    parser.add_argument('-p', '--paths', dest='paths_dict', type=json.loads)
    parser.add_argument('-s', '--simulation', dest='sim_dict', type=json.loads)
    parser.add_argument('-q', '--num_instances', dest='num_inst', type=int)
    parser.add_argument('-c', '--case_options', dest='case_opt', type=json.loads)

    args = parser.parse_args()

    if args.config_dict:
        params.OPTIONS.update(args.config_dict)

    if args.paths_dict:
        params.PATHS.update(args.paths_dict)

    if args.sim_dict:
        params.OPTIONS['simulation'].update(args.sim_dict)

    if not os.path.exists(params.PATHS['results']):
        os.mkdir(params.PATHS['results'])

    if args.num_inst:
        num_instances = args.num_inst
    else:
        num_instances = 10

    options = params.OPTIONS = sd.SuperDict.from_dict(params.OPTIONS)
    sim_data = options['simulation'] = sd.SuperDict.from_dict(options['simulation'])
    options['PATHS'] = params.PATHS

    if args.case_opt:
        case_options = args.case_opt
    else:
        case_options = {
            'maint_duration': [4, 8],
            'perc_capacity': [0.1, 0.2],
            'max_used_time': [800, 1200],
            'max_elapsed_time': [40, 80],
            'elapsed_time_size': [20, 40]
        }

    case_data = [{k: vv, 'name': '{}_{}'.format(re.sub(r'[_\}]', '', k), vv)}
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

        for sim in range(num_instances):
            if sim_data['seed']:
                sim_data['seed'] += 1
            case_sim = sd.SuperDict.from_dict(case).filter(sim_data.keys_l(), check=False)
            case_opt = sd.SuperDict.from_dict(case).filter(options.keys_l(), check=False)
            _options = copy.deepcopy(options)
            _sim_data = _options['simulation']
            _sim_data.update(case_sim)
            _options.update(case_opt)
            # this needs to be enforced so feasible instances can be obtained:
            _sim_data['num_resources'] = 15*_sim_data['num_parallel_tasks']
            _path_instance = path_instance = _options['path'] = \
                os.path.join(path_exp, dt.datetime.now().strftime("%Y%m%d%H%M"))

            num = 1
            while os.path.exists(_path_instance):
                _path_instance = path_instance + "_{}".format(num)
                num += 1
            _options['path'] = path_instance = _path_instance + '/'

            if not os.path.exists(_path_instance):
                os.mkdir(_path_instance)

            if multiproc:
                results[pos] = pool.apply_async(exec.solve_errors, [_options, path_instance])
                pos += 1
            else:
                exec.solve_errors(_options, path_instance)
            # time.sleep(60)

    for pos, result in results.items():
        result = result.get(timeout=10000)

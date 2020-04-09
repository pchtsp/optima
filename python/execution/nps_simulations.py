import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import execution.exec as exec
import datetime as dt
import pytups.superdict as sd
import data.simulation as sim
import argparse
import re
import copy
import multiprocessing as multi

def solve_errors(_option):
    try:
        print('path is : {}'.format(_option['path']))
        model_data = sim.create_dataset(_option)
        exec.execute_solve(model_data, _option)
    except Exception as e:
        str_fail = "Unexpected error in case: \n{}".format(repr(e))
        path_out = os.path.join(_option['path'], 'failure.txt')
        with open(path_out, 'w') as f:
            f.write(str_fail)

if __name__ == "__main__":

    import package.params as params
    import json
    parser = argparse.ArgumentParser(description='Solve an instance MFMP.')
    parser.add_argument('-d', '--options', dest='config_dict', type=json.loads)
    parser.add_argument('-p', '--paths', dest='paths_dict', type=json.loads)
    parser.add_argument('-s', '--simulation', dest='sim_dict', type=json.loads)
    parser.add_argument('-q', '--num_instances', dest='num_inst', type=int, required=True)
    parser.add_argument('-c', '--case_options', dest='case_opt', type=json.loads, required=True)
    parser.add_argument('-nb', '--no_base_case', dest='no_base_case', action='store_true')
    parser.add_argument('-nmp', '--no_multiprocess', dest='no_multiprocess', action='store_true')

    args = parser.parse_args()

    options = params.OPTIONS = sd.SuperDict.from_dict(params.OPTIONS)

    if args.config_dict:
        params.OPTIONS.update(args.config_dict)

    if args.paths_dict:
        params.PATHS.update(args.paths_dict)

    if args.sim_dict:
        params.OPTIONS['simulation'].update(args.sim_dict)

    if not os.path.exists(params.PATHS['results']):
        os.mkdir(params.PATHS['results'])

    num_instances = args.num_inst

    sim_data = options['simulation'] = sd.SuperDict(options['simulation'])
    options['PATHS'] = params.PATHS


    case_options = args.case_opt

    case_data = [{k: vv, 'name': '{}_{}'.format(re.sub('_', '', k), vv)}
                 for k, v in case_options.items() for vv in v]
    if not args.no_base_case:
        case_data += [{'name': 'base'}]

    seed_backup = sim_data['seed']
    multiproc = options.get('multiprocess')
    if args.no_multiprocess:
        multiproc = False
    time_limit_default = options.get('timeLimit', 3600) + 600
    results = {}
    pos = 0
    pool = None
    if multiproc:
        pool = multi.Pool(processes=multiproc)


    for case in case_data:
        sim_data['seed'] = seed_backup
        path_exp = params.PATHS['experiments'] = \
            params.PATHS['results'] + case['name']
        if not os.path.exists(path_exp):
            os.mkdir(path_exp)

        case_opt = sd.SuperDict(case).filter(options.keys_l(), check=False)
        case_opt = {k: v for k, v in case_opt.items()}
        _option = copy.deepcopy(options)
        _option.update(case_opt)
        case_sim = sd.SuperDict(case).filter(sim_data.keys_l(), check=False)
        case_sim = {k: v for k, v in case_sim.items()}
        _sim_data = _option['simulation']
        _sim_data.update(case_sim)
        # this needs to be enforced so feasible instances can be obtained:
        _sim_data['num_resources'] = 15 * _sim_data['num_parallel_tasks']

        for _ in range(num_instances):
            if _sim_data['seed']:
                _sim_data['seed'] += 1

            _path_instance = path_instance = _option['path'] = \
                os.path.join(path_exp, dt.datetime.now().strftime("%Y%m%d%H%M"))
            num = 1

            while os.path.exists(_path_instance):
                _path_instance = path_instance + "_{}".format(num)
                num += 1
            _option['path'] = path_instance = _path_instance + '/'

            if not os.path.exists(_path_instance):
                os.mkdir(_path_instance)

            args = [copy.deepcopy(_option)]
            if multiproc:
                # print('create poolasync')
                results[pos] = pool.apply_async(solve_errors, args)
                pos += 1
            else:
                solve_errors(*args)

    for pos, result in results.items():
        # print('actually running functions')
        try:
            result.get(timeout=time_limit_default)
        except multi.TimeoutError:
            print('We lacked patience and got a multiprocessing.TimeoutError')


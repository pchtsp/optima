import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import execution.exec as exec
import execution.exec_batch as exec_batch

import pytups.superdict as sd
import argparse
import re

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

    options['simulation'] = sd.SuperDict(options['simulation'])
    options['PATHS'] = params.PATHS

    num_instances = args.num_inst
    case_options = args.case_opt
    no_base_case = args.no_base_case
    multiproc = options.get('multiprocess')
    if args.no_multiprocess:
        multiproc = False

    case_data = [{k: vv, 'name': '{}_{}'.format(re.sub('_', '', k), vv)}
                 for k, v in case_options.items() for vv in v]
    if not no_base_case:
        case_data += [{'name': 'base'}]

    options_list = exec_batch.prepare_directories_and_optionlist(case_data, options, num_instances)
    func = exec.config_and_solve
    # func = solve_errors

    # if no multiprocess, we execute everything and exit
    if not multiproc:
        for args in options_list:
            func(args)
    else:
        # in case we do multiprocessing
        time_limit_default = options.get('timeLimit', 3600) + 600
        exec_batch.execute_with_multiprocessing(func, multiproc, options_list, time_limit_default)


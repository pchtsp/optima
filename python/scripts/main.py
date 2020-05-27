import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import pytups.superdict as sd
import argparse
import datetime as dt
import json
import re
import gc

import execution.exec as exec
import execution.exec_batch as exec_batch
import data.data_input as di


if __name__ == "__main__":

    import package.params as params
    parser = argparse.ArgumentParser(description='Solve an instance MFMP.')

    # config related:
    parser.add_argument('-df', '--options-file', dest='config_file')
    parser.add_argument('-d', '--options', dest='config_dict', type=json.loads)
    parser.add_argument('-p', '--paths', dest='paths_dict', type=json.loads)

    # simulation related:
    parser.add_argument('-s', '--simulation', dest='sim_dict', type=json.loads)
    parser.add_argument('-q', '--num_instances', dest='num_inst', type=int, default=0)
    parser.add_argument('-c', '--case_options', dest='case_opt', type=json.loads, default=None)
    parser.add_argument('-nb', '--no_base_case', dest='no_base_case', action='store_true')
    parser.add_argument('-nmp', '--no_multiprocess', dest='no_multiprocess', action='store_true')

    # Exceptions
    parser.add_argument('-mm', '--max_memory', dest='max_memory', type=float, default=0.5)

    # GUI related:
    parser.add_argument('-it', '--input-template', dest='input_template')
    parser.add_argument('-id', '--input-template-dir', dest='input_template_dir')
    parser.add_argument('-gui', '--desktop-app', dest='open_desktop_app', action='store_true')

    if getattr(sys, 'frozen', False):
        # we are running in a bundle
        root = sys._MEIPASS + '/'
    else:
        # we are running in a normal Python environment
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/'

    args = parser.parse_args()

    options = params.OPTIONS = sd.SuperDict.from_dict(params.OPTIONS)

    params.OPTIONS['root'] = root

    new_options = None
    if args.config_dict:
        new_options = args.config_dict
    elif args.config_file:
        new_options = di.load_data(args.config_file)
    if new_options:
        params.OPTIONS.update(new_options)

    if args.paths_dict:
        params.PATHS.update(args.paths_dict)
        path = os.path.join(params.PATHS['experiments'], dt.datetime.now().strftime("%Y%m%d%H%M")) + '/'
        exec.update_case_path(params.OPTIONS, path)

    if args.sim_dict:
        params.OPTIONS['simulation'].update(args.sim_dict)

    if not os.path.exists(params.PATHS['results']):
        os.mkdir(params.PATHS['results'])

    # num_instances = args.num_inst

    sim_data = options['simulation'] = sd.SuperDict(options['simulation'])
    options['PATHS'] = params.PATHS

    if args.input_template_dir:
        exec.udpdate_case_read_options(params.OPTIONS, args.input_template_dir)

    if args.input_template:
        params.OPTIONS['input_template_path'] = args.input_template

    if args.max_memory:
        exec.memory_limit(args.max_memory)

    num_instances = args.num_inst
    case_options = args.case_opt
    no_base_case = args.no_base_case
    multiproc = options.get('multiprocess')
    if args.no_multiprocess:
        multiproc = False

    # batch mode uses slightly more complicated functions
    batch = num_instances or case_options

    if args.open_desktop_app:
        try:
            import desktop_app.app as gui_app
        except Exception as e:
            print('GUI needs additional libraries.')
            raise e
        options['template'] = True
        gui_app.MainWindow_EXCEC(options)
        sys.exit()
    if not batch:
        exec.config_and_solve(options)
        sys.exit()

    # batch mode
    case_data = [{k: vv, 'name': '{}_{}'.format(re.sub('_', '', k), vv)}
                 for k, v in case_options.items() for vv in v]
    if not no_base_case:
        case_data += [{'name': 'base'}]

    options_list = exec_batch.prepare_directories_and_optionlist(case_data, options, num_instances)

    # if no multiprocess, we execute everything and exit
    if not multiproc:
        for args in options_list:
            exec_batch.solve_errors(args)
            gc.collect()
    else:
        # in case we do multiprocessing
        time_limit_default = options.get('timeLimit', 3600) + 600
        exec_batch.execute_with_multiprocessing(multiproc, options_list, time_limit_default)


import os
import pytups.superdict as sd
import datetime as dt
import copy
import multiprocessing as multi
import execution.exec as exec


def solve_errors(_option):
    try:
        print('path is : {}'.format(_option['path']))
        exec.config_and_solve(_option)
    except Exception as e:
        str_fail = "Unexpected error in case: \n{}".format(repr(e))
        path_out = os.path.join(_option['path'], 'failure.txt')
        with open(path_out, 'w') as f:
            f.write(str_fail)


def prepare_options(options, case):
    case_opt = sd.SuperDict(case).filter(options.keys_l(), check=False)
    case_opt = {k: v for k, v in case_opt.items()}
    _option = copy.deepcopy(options)
    _option.update(case_opt)
    case_sim = sd.SuperDict(case).filter(_option['simulation'].keys_l(), check=False)
    case_sim = {k: v for k, v in case_sim.items()}
    _option['simulation'].update(case_sim)
    # this needs to be enforced so feasible instances can be obtained:
    _option['simulation']['num_resources'] = 15 * _option['simulation']['num_parallel_tasks']
    return _option


def get_path_instance(path_exp):
    _path_instance = path_instance = os.path.join(path_exp, dt.datetime.now().strftime("%Y%m%d%H%M"))
    num = 1

    while os.path.exists(_path_instance):
        _path_instance = path_instance + "_{}".format(num)
        num += 1
    return _path_instance + '/'


def prepare_directories_and_optionlist(case_data, options, num_instances):
    options_list = []
    for case in case_data:
        _option = prepare_options(options, case)

        path_exp = _option['PATHS']['experiments'] = _option['PATHS']['results'] + case['name']
        if not os.path.exists(path_exp):
            os.mkdir(path_exp)

        for _ in range(num_instances):
            if _option['simulation']['seed']:
                _option['simulation']['seed'] += 1

            _option['path'] = path_instance = get_path_instance(path_exp)

            if not os.path.exists(path_instance):
                os.mkdir(path_instance)

            options_list.append(copy.deepcopy(_option))
    return options_list


def execute_with_multiprocessing(multiproc, options_list, time_limit_default):
    pool = multi.Pool(processes=multiproc)
    results = {}
    pos = 0

    for args in options_list:
        # print('create poolasync')
        results[pos] = pool.apply_async(solve_errors, [args])
        pos += 1

    for pos, result in results.items():
        # print('actually running functions')
        try:
            result.get(timeout=time_limit_default)
        except multi.TimeoutError:
            print('We lacked patience and got a multiprocessing.TimeoutError')


import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import scripts.exec as exec
import datetime as dt
import package.superdict as sd
import argparse

if __name__ == "__main__":

    import package.params as params
    import json
    parser = argparse.ArgumentParser(description='Solve an instance MFMP.')
    parser.add_argument('-d', '--options', dest='config_dict', type=json.loads)
    parser.add_argument('-p', '--paths', dest='paths_dict', type=json.loads)

    args = parser.parse_args()

    if args.config_dict:
        params.OPTIONS.update(args.config_dict)

    if args.paths_dict:
        params.PATHS.update(args.paths_dict)

    if not os.path.exists(params.PATHS['results']):
        os.mkdir(params.PATHS['results'])

    options = params.OPTIONS = sd.SuperDict(params.OPTIONS)
    sim_data = options['simulation'] = sd.SuperDict(options['simulation'])

    case_data = [{
        'num_period': periods,
        'num_parallel_tasks': num_tasks,
        'num_resources': num_tasks * res_per_task,
        # 'solver': solver,
        'name': 'task_periods_minusage_pricerutend_respertask_{}_{}_{}_{}_{}/'.format(num_tasks, periods, min_usage, price_rut_end, res_per_task),
        'min_usage_period': min_usage,
        'price_rut_end': price_rut_end
    }
        for num_tasks in range(1, 2)
        for periods in [90]
        for min_usage in [0]
        for price_rut_end in [0, 1]
        for res_per_task in [15, 30]
    ]

    for case in case_data:
        path_exp = params.PATHS['experiments'] = \
            params.PATHS['results'] + case['name']
        if not os.path.exists(path_exp):
            os.mkdir(path_exp)

        for sim in range(10):
            sim_data['seed'] = None
            case_sim = sd.SuperDict(case).filter(sim_data.keys_l(), check=False)
            case_opt = sd.SuperDict(case).filter(options.keys_l(), check=False)
            sim_data.update(case_sim)
            options.update(case_opt)
            options['path'] = \
                os.path.join(path_exp, dt.datetime.now().strftime("%Y%m%d%H%M"))
            num = 1
            path_instance = options['path']
            while os.path.exists(path_instance):
                path_instance = options['path'] + "_{}".format(num)
                num += 1
            path_instance += '/'
            options['path'] = path_instance
            try:
                exec.config_and_solve(params)
            except Exception as e:
                if not os.path.exists(path_instance):
                    os.mkdir(path_instance)
                str_fail = "Unexpected error in case: \n{}".format(repr(e))
                with open(path_instance + 'failure.txt', 'w') as f:
                    f.write(str_fail)
            # time.sleep(60)

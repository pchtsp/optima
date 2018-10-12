import scripts.exec as exec
import os
import datetime as dt
import time
import sys
import package.superdict as sd


if __name__ == "__main__":
    import package.params as params
    options = params.OPTIONS = sd.SuperDict(params.OPTIONS)
    sim_data = options['simulation'] = sd.SuperDict(options['simulation'])

    case_data = [{
        'num_period': periods,
        'num_parallel_tasks': num_tasks,
        'num_resources': num_tasks * 30,
        # 'solver': solver,
        'name': 'simulated_data/task_periods_solv_{}_{}_{}/'.format(num_tasks, periods, solver)
    }
        for num_tasks in range(1, 4)
        for solver in ['GUROBI']
        for periods in [10, 30, 50]
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
            except:
                if not os.path.exists(path_instance):
                    os.mkdir(path_instance)
                str_fail = "Unexpected error in case: \n{}".format(sys.exc_info()[0])
                with open(path_instance + 'failure.txt', 'w') as f:
                    f.write(str_fail)
            # time.sleep(60)
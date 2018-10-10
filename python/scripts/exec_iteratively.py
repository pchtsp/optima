import scripts.exec as exec
import os
import datetime as dt
import time
import sys
import package.superdict as sd


if __name__ == "__main__":
    import package.params as params
    options = sd.SuperDict(params.OPTIONS)
    sim_data = sd.SuperDict(options['simulation'])

    case_data = [{
        'num_period': periods,
        'num_parallel_tasks': num_tasks,
        'num_resources': num_tasks * 30,
        # 'solver': solver,
        'name': 'simulated_data/task_types_capa_{}_{}/'.format(num_tasks, periods)
    }
        for num_tasks in range(1, 4)
        # for solver in ['GUROBI', 'CPLEX']
        for periods in [10, 30, 50]
    ]

    for case in case_data:
        for sim in range(10):
            path_exp = params.PATHS['experiments'] = \
                params.PATHS['results'] + case['name']
            if not os.path.exists(path_exp):
                os.mkdir(path_exp)
            sim_data['seed'] = None
            case_sim = sd.SuperDict(case).filter(sim_data.keys_l(), check=False)
            case_opt = sd.SuperDict(case).filter(options.keys_l(), check=False)
            sim_data.update(case_sim)
            options.update(case_opt)
            # sim_data['num_parallel_tasks'] = num_tasks
            options['path'] = \
                os.path.join(path_exp, dt.datetime.now().strftime("%Y%m%d%H%M"))
            num = 1
            while os.path.exists(options['path']):
                options['path'] += "_{}".format(num)
                num += 1

            options['path'] += '/'

            try:
                exec.config_and_solve(params)
            except:
                print("Unexpected error in case {}:", options['path'], sys.exc_info()[0])
            # time.sleep(60)
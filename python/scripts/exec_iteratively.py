import scripts.exec as exec
import os
import datetime as dt
import time


if __name__ == "__main__":
    import package.params as params
    sim_data = params.OPTIONS['simulation']

    for num_tasks in range(1, 3):
        for sim in range(30):
            params.PATHS['experiments'] = \
                params.PATHS['results'] + 'simulated_data/{}_task_slack/'.format(num_tasks)
            sim_data['seed'] = None
            sim_data['num_resources'] = num_tasks * 50
            sim_data['num_parallel_tasks'] = num_tasks
            # we don't care about the original params object.
            # so we do not copy it.
            # params.OPTIONS['end_pos'] = period
            # params.OPTIONS['solver'] = solver
            params.OPTIONS['path'] = os.path.join(
                params.PATHS['experiments'],
                dt.datetime.now().strftime("%Y%m%d%H%M")
            ) + '/'

            exec.config_and_solve(params)
            time.sleep(60)
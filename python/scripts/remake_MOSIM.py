import os
# import package.instance as inst
import package.params as params
import package.experiment as exp
import solvers.model as md
import package.data_input as di


if __name__ == "__main__":

    path = params.PATHS['results'] + 'MOSIM2018_copy/'
    cases = [c for c in os.listdir(path) if c.startswith("2018")]
    for case in cases:
        # case = '201801141646'
        output_path = path + case + '/'
        experiment = exp.Experiment.from_dir(output_path)
        instance = experiment.instance
        options = di.load_data(output_path + 'options.json')
        options['path'] = output_path
        options['gap'] = 0.00049
        solution = md.solve_model(instance, options=options)
        if solution is not None:
            di.export_data(output_path, solution.data, name="data_out", file_type='json')

    # sim_data = params.OPTIONS['simulation']
    # params.PATHS['experiments'] = params.PATHS['results'] + 'simulated_data/2_task_slack/'
    #
    # for num_tasks in range(2, 3):
    #     for sim in range(30):
    #         sim_data['seed'] = None
    #         sim_data['num_resources'] = num_tasks * 50
    #         sim_data['num_parallel_tasks'] = num_tasks
    #         # we don't care about the original params object.
    #         # so we do not copy it.
    #         # params.OPTIONS['end_pos'] = period
    #         # params.OPTIONS['solver'] = solver
    #         params.OPTIONS['path'] = os.path.join(
    #             params.PATHS['experiments'],
    #             dt.datetime.now().strftime("%Y%m%d%H%M")
    #         ) + '/'
    #
    #         exec.config_and_solve(params)
    #         time.sleep(60)
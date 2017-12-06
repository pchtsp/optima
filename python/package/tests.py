# /usr/bin/python3
import package.aux as aux

"""

## tasks

* are covered: number of resources
* covered by correct resources and no other
* covered; minimum of hours

## resources

* balance is well done
* at most one task in each period.
* maintenance scheduled correctly: duration.
* no task+maintenance at the same time.

"""

def check_task_numresources(model_data, solution):
    initial_period = model_data['parameters']['start']
    last_period = model_data['parameters']['end']
    periods = aux.get_months(initial_period, last_period)
    task_reqs = aux.get_property_from_dic(model_data['tasks'], 'num_resource')
    task_start = aux.get_property_from_dic(model_data['tasks'], 'start')
    task_start = {task: max(task_start[task], initial_period) for task in task_start}
    task_end = aux.get_property_from_dic(model_data['tasks'], 'end')

    task_periods = {task: [period for period in aux.get_months(task_start[task], task_end[task])] for task in task_reqs}
    task_reqs_month = {(task, period): task_reqs[task]
                       for task in task_reqs
                       for period in periods }

    task_assigned = {task: 0 for task in task_reqs}
    for (task, period) in solution['task']:
        task_assigned[task] += 1

    for task in task_periods:
        for period in task_periods[task]:
            return
    # assigned
    # solution['task']

    return


if __name__ == "__main__":
    path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712051356/"
    model_data = aux.load_data(path + "data_in.pickle")
    solution = aux.load_data(path + "data_out.pickle")

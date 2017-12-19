# /usr/bin/python3
import package.aux as aux
import numpy as np
import pandas as pd
import package.data_input as di
import package.solution as sol
import package.instance as inst
import os
import shutil

"""

## tasks

[*] are covered: number of resources
[*] covered by correct resources and no other

## resources

[*] balance is well done
[*] consumption
[*] at most one task in each period => 
    this I cannot check because of the structure of the data.
    But it could be detected if tasks' needs are not taken into account.
[*] maintenance scheduled correctly: duration.
[*] no task+maintenance at the same time.
"""


class Experiment(object):

    def __init__(self, instance, solution):
        self.instance = instance
        self.solution = solution

    @classmethod
    def from_dir(cls, path, format='json', prefix="data_"):
        files = [os.path.join(path, prefix + f + "." + format) for f in ['in', 'out']]
        if not np.all([os.path.exists(f) for f in files]):
            return None
        instance = di.load_data(files[0])
        solution = di.load_data(files[1])
        return cls(inst.Instance(instance), sol.Solution(solution))

    def check_solution(self):
        func_list = {
            'duration':     self.check_maintenance_duration
            ,'candidates':  self.check_resource_in_candidates
            ,'state':       self.check_resource_state
            ,'resources':   self.check_task_num_resources
            ,'usage':       self.check_resource_consumption
        }
        return {k: v() for k, v in func_list.items()}

    def check_task_num_resources(self):
        task_reqs = self.instance.get_tasks('num_resource')

        task_assigned = \
            aux.fill_dict_with_default(
                self.solution.get_task_num_resources(),
                self.instance.get_task_period_list()
            )
        task_under_assigned = {
            (task, period): task_reqs[task] - task_assigned[(task, period)]
            for (task, period) in task_assigned
            if task_reqs[task] - task_assigned[(task, period)] != 0
        }

        return task_under_assigned

    def check_resource_in_candidates(self):
        task_data = self.instance.get_tasks()
        task_solution = self.solution.get_tasks()

        task_candidates = aux.get_property_from_dic(task_data, 'candidates')

        bad_assignment = {
            (resource, period): task
            for (resource, period), task in task_solution.items()
            if resource not in task_candidates[task]
        }
        return bad_assignment

    def get_consumption(self):
        maint_hours = self.instance.get_param('max_used_time')
        hours = self.instance.get_tasks("consumption")
        demand = {k: hours[v] for k, v in self.solution.get_tasks().items()}
        supply = {k: maint_hours for k in self.solution.get_maintenance_starts()}
        netgain = {(resource, period): 0
                   for resource in self.instance.get_resources()
                   for period in self.instance.get_periods()}
        for k, v in demand.items():
            netgain[k] -= v
        for k, v in supply.items():
            netgain[k] += v
        return netgain

    def get_remaining_usage_time(self):
        prev_month = aux.get_prev_month(self.instance.get_param('start'))
        table_initial = \
            pd.DataFrame(
                aux.dict_to_tup(
                    self.instance.get_resources('initial_used')
                ),
                columns=['resource', 'netgain']).assign(period=prev_month)
        netgain = self.get_consumption()
        table = pd.DataFrame(aux.dict_to_tup(netgain), columns=['resource', 'period', 'netgain'])
        table = pd.concat([table_initial, table])
        table.sort_values(['resource', 'period'], inplace=True)
        table['netgain_c'] = \
            table.groupby('resource').netgain.cumsum()
        return table.set_index(['resource', 'period'])['netgain_c'].to_dict()

    def check_resource_consumption(self):
        rut = self.get_remaining_usage_time()
        return {k: v for k, v in rut.items() if v < 0}

    def check_resource_state(self):
        task_solution = self.solution.get_tasks()
        state_solution = self.solution.get_state()

        task_solution_k = np.fromiter(task_solution.keys(),
                                      dtype=[('A', '<U6'), ('T', 'U7')])
        state_solution_k = np.fromiter(state_solution.keys(),
                                      dtype=[('A', '<U6'), ('T', 'U7')])
        duplicated_states = \
            np.intersect1d(task_solution_k, state_solution_k)

        return [tuple(item) for item in duplicated_states]

    def check_maintenance_duration(self):
        maintenances = self.solution.get_maintenance_periods()
        first_period = self.instance.get_param('start')
        last_period = self.instance.get_param('end')
        duration = self.instance.get_param('maint_duration')

        maintenance_duration_incorrect = {}
        for (resource, start, finish) in maintenances:
            size_period = len(aux.get_months(start, finish))
            if size_period > duration:
                maintenance_duration_incorrect[(resource, start)] = size_period
            if size_period < duration and start != first_period and \
                            finish != last_period:
                maintenance_duration_incorrect[(resource, start)] = size_period
        return maintenance_duration_incorrect

    def get_objective_function(self):
        weight = self.instance.get_param("maint_weight")
        unavailable = max(self.solution.get_unavailable().values())
        in_maint = max(self.solution.get_in_maintenance().values())
        return in_maint * weight + unavailable


def clean_experiments(path, clean=True):
    """
    loads and cleans all experiments that are incomplete
    :param path: path to experiments
    :param clean: if set to false it only does not delete them
    :return: deleted experiments
    """
    exps_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    to_delete = []
    for e in exps_paths:
        exp = Experiment.from_dir(e, format="json")
        if exp is None:
            exp = Experiment.from_dir(e, format="pickle")
        to_delete.append(exp is None)
    exps_to_delete = np.array(exps_paths)[to_delete]
    if clean:
        for ed in exps_to_delete:
            shutil.rmtree(ed)
    return exps_to_delete
    # np.array(exps_paths).__len__()
    #
    # return 0


if __name__ == "__main__":

    path_states = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712190002/"
    path_nostates = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712182321/"

    sol_states = Experiment.from_dir(path_states)
    sol_nostates = Experiment.from_dir(path_nostates)

    t = aux.tup_to_dict(aux.dict_to_tup(sol_nostates.solution.get_tasks()),
                    result_col= [0], is_list=True, indeces=[2, 1]).items()
    # {k: len(v) for k, v in t}
    sol_nostates.instance.get_tasks('num_resource')

    # sol_nostates.check_solution()
    # sol_states.check_solution()

    # exp = Experiment(inst.Instance(model_data), sol.Solution(solution))
    # exp = Experiment.from_dir(path)
    # results = exp.check_solution()

    # [k for (k, v) in sol_nostates.check_solution().items() if len(v) > 0]
    # [k for (k, v) in sol_states.check_solution().items() if len(v) > 0]

    print(sol_states.get_objective_function())
    print(sol_nostates.get_objective_function())
    sol_nostates.check_task_num_resources()

    # sol_states.solution.get_state()[('A2', '2017-03')]
    l = sol_states.instance.get_domains_sets()
    # l['v_at']
    checks = sol_states.check_solution()

    sol_nostates.check_solution()
    # sol_states.solution.print_solution("/home/pchtsp/Documents/projects/OPTIMA/img/calendar.html")
    sol_nostates.solution.print_solution("/home/pchtsp/Documents/projects/OPTIMA/img/calendar.html")
    #
    # [k for (k, v) in results["duration"].items() if v < 6]
    # results["usage"]
    #
    #
    # consum = exp.get_consumption()
    # aux.dicttup_to_dictdict(exp.get_remaining_usage_time())['A46']
    # aux.dicttup_to_dictdict(consum)["A46"]
    #
    # results["resources"]
    #
    # aux.dicttup_to_dictdict(exp.solution.get_task_num_resources())['O10']['2017-09']
    # exp.instance.get_tasks("num_resource")

    path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments"
    exps = clean_experiments(path, clean=True)
    len(exps)

    # sol.Solution(solution).get_schedule()
    # check.check_resource_consumption()
    # check.check_resource_state()
    # pp.pprint(results)
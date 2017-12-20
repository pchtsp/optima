# /usr/bin/python3

import package.aux as aux
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import plotly.figure_factory as ff
import plotly


class Solution(object):
    """
    These objects represent the solution to the assignment problem
    It does not include the initial data.
    The methods are made to make it easy to get information.
    They also draw graphs.
    """

    def __init__(self, solution):
        self.data = solution

    def get_periods(self):
        resource_period = list(self.get_tasks().keys())
        return sorted(aux.tup_to_dict(resource_period, result_col=0).keys())

    def get_tasks(self):
        return aux.dictdict_to_dictup(self.data['task'])

    def get_state(self):
        return aux.dictdict_to_dictup(self.data['state'])

    def get_maintenance_periods(self):
        return [(k[0], k[1], k[3]) for k in self.get_schedule() if k[2] == 'M']

    def get_maintenance_starts(self):
        maintenances = self.get_maintenance_periods()
        return [(r, s) for (r, s, e) in maintenances]

    def get_task_resources(self):
        task_solution = self.get_tasks()
        task_resources = aux.tup_to_dict(aux.dict_to_tup(task_solution), 0, is_list=True)
        return {(a, t): v for (t, a), v in task_resources.items()}

    def get_task_num_resources(self):
        return {key: len(value) for key, value in self.get_task_resources().items()}

    def get_schedule(self):
        """
        returns periods of time where a resources has a specific state
        In a way, collapses single period assignments into a start-finish
        :return: a (resource, start, finish, task) tuple
        """
        statesMissions = aux.dict_to_tup(self.get_state()) + aux.dict_to_tup(self.get_tasks())
        result = aux.tup_to_start_finish(statesMissions)
        return result

    def get_unavailable(self):
        num_tasks = self.get_in_mission()
        in_maint = self.get_in_maintenance()
        return {k: in_maint[k] + num_tasks[k] for k in in_maint}  # in_maint has all periods already

    def get_in_mission(self):
        tasks = [(t, r) for (r, t) in self.get_tasks()]
        in_mission = {k: len(v) for k, v in aux.tup_to_dict(tasks, 1, is_list=True).items()}
        return aux.fill_dict_with_default(in_mission, self.get_periods())

    def get_in_maintenance(self):
        states = [(t, r) for (r, t), v in self.get_state().items() if v == 'M']
        in_maint = {k: len(v) for k, v in aux.tup_to_dict(states, 1, is_list=True).items()}
        # TODO: add fixed maintenances? Or maybe it should be included in the states already
        # fixed_maints = self.instance.get_fixed_maintenances()
        return aux.fill_dict_with_default(in_maint, self.get_periods())

    def graph_maintenances(self, path, **kwags):
        """
        uses bokeh
        :param path: path to generate image
        :return: graph object
        """
        return aux.graph_dict_time(self.get_in_maintenance(), path, **kwags)

    def graph_unavailable(self, path, **kwags):
        return aux.graph_dict_time(self.get_unavailable(), path, **kwags)

    def print_solution(self, path, max_tasks=None):
        cols = ['group', 'start', 'content', 'end']
        table = pd.DataFrame(self.get_schedule(),
                             columns=cols
                             )
        table.end = table.end.apply(lambda x: aux.get_next_month(x))
        table['status'] = table.content.str.replace(r"\d+", "")
        colors = {'A': "white", 'M': '#0055FF', 'O': "#BD0026"}
        colors_content = table.set_index('content').status.replace(colors).to_dict()

        table['style'] = \
            table.status.replace(colors). \
                map("background-color:{0};border-color:{0}".format)
        table = table.sort_values("group").reset_index().rename(columns={'index': "id"})
        cols2 = cols + ['style', 'id']
        table[cols2] = table[cols2].astype(str)
        table = table[['group', 'start', 'content', 'end']]
        table.columns = ["Task", 'Start', 'Resource', 'Finish']
        if max_tasks is not None:
            # TODO: filter correctly tasks to show only the maximum number
            table = table[table.Task.str.len() < 3].reset_index(drop=True)
        fig = ff.create_gantt(table, colors=colors_content, index_col='Resource', show_colorbar=True, group_tasks=True)
        plotly.offline.plot(fig, filename=path)

        return

    def print_solution_r(self, path):
        # TODO: this is almost correct but some names are not correctly written.
        cols = ['group', 'start', 'content', 'end']
        table = pd.DataFrame(self.get_schedule(),
                             columns=cols
                             )
        table.end = table.end.apply(lambda x: aux.get_next_month(x))
        table['status'] = table.content.str.replace(r"\d+", "")
        colors = {'A': "white", 'M': '#0055FF', 'O': "#BD0026"}

        table['style'] = \
            table.status.replace(colors). \
                map("background-color:{0};border-color:{0}".format)

        groups = pd.DataFrame({'id': table.group.unique(), 'content': table.group.unique()})
        timevis = importr('timevis')
        htmlwidgets = importr('htmlwidgets')
        rdf = pandas2ri.py2ri(table)
        rdfgroups = pandas2ri.py2ri(groups)

        options = ro.ListVector({
            "stack": False,
            "editable": True,
            "align": "center",
            "orientation": "top",
            # "snap": None,
            "margin": 0
        })

        graph = timevis.timevis(rdf, groups=rdfgroups, options=options, width="100%")
        htmlwidgets.saveWidget(graph, file=path, selfcontained=False)
        print(graph)
        return graph


if __name__ == "__main__":
    path_states = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712190002/"
    path_nostates = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712181704/"
    import package.data_input as di
    sol_states = Solution(di.load_data(path_states + "data_out.json"))
    sol_nostates = Solution(di.load_data(path_nostates + "data_out.json"))
    # sol_nostates.graph_maintenances(path="/home/pchtsp/Documents/projects/OPTIMA/img/maintenances.html",
    #                                 title="Maintenances")
    # sol_nostates.graph_unavailable(path="/home/pchtsp/Documents/projects/OPTIMA/img/unavailable.html",
    #                                title="Affectations")
    sol_nostates.print_solution("/home/pchtsp/Documents/projects/OPTIMA/img/calendar.html")

    # sol.print_solution("/home/pchtsp/Downloads/calendar_temp3.html")
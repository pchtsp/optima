# /usr/bin/python3

import package.aux as aux
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri


class Solution(object):

    def __init__(self, solution):
        self.data = solution

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

    def print_solution(self, path):
        cols = ['group', 'start', 'content', 'end']
        table = pd.DataFrame(self.get_schedule(),
                             columns=cols
                             )
        table.end = table.end.apply(lambda x: aux.get_next_month(x))
        table['status'] = table.content.str.replace(r"\d+", "")
        colors = {'A': "white", 'M': '#FFFFB2', 'O': "#BD0026"}
        table['style'] = \
            table.status.replace(colors). \
                map("background-color:{0};border-color:{0}".format)
        table = table.sort_values("group").reset_index().rename(columns={'index': "id"})
        cols2 = cols + ['style', 'id']
        table[cols2] = table[cols2].astype(str)
        # table = table.astype(
        #     {'content': 'U5', 'id': 'int64', 'group': 'U5', 'start': 'U7', 'end': 'U7', 'status': 'U5', 'style': 'U40'})
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
        return
        # print(graph)


if __name__ == "__main__":
    path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712121208/"
    path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712142345/"
    import package.data_input as di
    sol = Solution(di.load_data(path + "data_out.json"))
    sol.print_solution("/home/pchtsp/Downloads/calendar_temp3.html")
import patterns.node as node


class Node(node.Node):

    def get_adjacency_list_tasks(self, maint_options_ret):
        # 1. get compatible tasks
        tasks = self.instance.get_task_candidates(resource=self.resource)
        # 2. budget

        max_num_periods = self.get_tasks_data('consumption').vapply(lambda v: min(rut // v for rut in self.rut.values()))
        max_num_periods2 = self.get_tasks_data('end').vapply(self.dif_period)
        max_num_periods = max_num_periods.sapply(min, max_num_periods2).vapply(lambda v: v + 1)
        min_num_periods = self.get_tasks_data('min_assign')
        possible_tasks = min_num_periods.sapply(range, max_num_periods).to_tuplist()

        # TODO: do not go too far with assignments, filter if maintenance is coming with periods
        pass




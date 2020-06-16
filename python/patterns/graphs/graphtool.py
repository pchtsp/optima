import graph_tool.all as gr
from pytups import superdict as sd, tuplist as tl
import numpy as np
import data.data_input as di
import os
from .core import DAG, get_artificial_nodes
import patterns.node as nd
import random as rn
import package.instance as inst
import logging as log
import itertools
import ujson as json

# installing graph-tool and adding it to venv:
# https://git.skewed.de/count0/graph-tool/wikis/installation-instructions
# https://jolo.xyz/blog/2018/12/07/installing-graph-tool-with-virtualenv


class GraphTool(DAG):

    def __init__(self, instance: inst.Instance, resource=None, nodes_ady=None, empty=False):
        if nodes_ady is not None or empty:
            self.instance = instance
            self.sink = nd.get_sink_node(instance)
        else:
            if not resource:
                raise ValueError('resource argument needs to be filled if no adjacency is given')
            super().__init__(instance, resource)
            nodes_ady = self.g
        if empty:
            return
        self.g = gr.Graph()
        edges = nodes_ady.to_tuplist()
        nodes = (edges.take(0) + edges.take(1)).unique2()
        vertices = self.g.add_vertex(len(nodes))
        self.refs = {node: int(v) for node, v in zip(nodes, vertices)}
        self.refs = sd.SuperDict(self.refs)
        self.refs_inv = self.refs.reverse()

        edges_list = edges.vapply(lambda v: (self.refs[v[0]], self.refs[v[1]]))
        self.g.add_edge_list(edges_list)

        # delete nodes with infinite distance to sink:
        self.g.set_reversed(is_reversed=True)
        distances = self.shortest_path(node1=self.sink)
        max_dist = instance.get_dist_periods(*instance.get_first_last_period()) + 5
        nodes = [n for n in self.g.vertices()
                 if distances[n] > max_dist and
                 self.refs_inv[n].rut is not None]
        self.g.set_reversed(is_reversed=False)
        self.g.remove_vertex(nodes, fast=True)
        self.g.shrink_to_fit()
        self.g.reindex_edges()

        self.edges = self.g.get_edges()
        self.vp_not_task = self.g.new_vp('bool', val=1)
        self.period_vp = self.g.new_vp('int')
        self.period_end_vp = self.g.new_vp('int')
        self.type_vp = self.g.new_vp('int')
        self.assgin_vp = self.g.new_vp('int')
        self.rut_first_vp = self.g.new_vp('int')
        self.duration_vp = self.g.new_vp('int')
        self.rut_vp = self.g.new_vp('int')
        self.ret_vp = self.g.new_vp('int')
        self.weights = self.g.new_ep('int')
        self._equiv_task = {k: v +1 for v, k in enumerate(self.instance.get_tasks())}
        self._equiv_task[''] = 0
        self._equiv_task['M'] = -1
        num = self.instance.get_param('num_period')
        self.period_ruts = sd.SuperDict({k: [] for k in range(-1, num+1)})
        # after the graph is generated, we initialize the constants:
        self.initialize_graph()
        self.set_weights(self.weights)
        self.resource_nodes = sd.SuperDict()

    def initialize_graph(self):
        # create dictionary to filter nodes that have no tasks
        positions = self.instance.get_period_positions()
        consum = self.instance.get_tasks('consumption')
        for v in self.g.vertices():
            node = self.refs_inv[v]
            self.type_vp[v] = node.type
            start = positions[node.period]
            end = positions[node.period_end]
            self.period_vp[v] = start
            self.period_end_vp[v] = end
            self.duration_vp[v] = end - start + 1
            if node.type == nd.EMPTY_TYPE:
                self.assgin_vp[v] = 0
            else:
                self.assgin_vp[v] = self._equiv_task[node.assignment]
            self.rut_vp[v] = node.rut['M'] if node.rut else 0
            self.ret_vp[v] = node.ret['M'] if node.ret else 0
            # it a task, we need to calculate the average rut
            # else: it's the last period
            if node.type == nd.TASK_TYPE:
                self.rut_first_vp[v] = self.rut_vp[v] + (self.duration_vp[v] - 1)*consum[node.assignment]
            else:
                self.rut_first_vp[v] = self.rut_vp[v]
            rut_first = self.rut_first_vp[v]
            rut_end = self.rut_vp[v]
            size = self.duration_vp[v]
            edges = self.g.get_in_edges(v, [self.g.edge_index])[:, 2]
            if size > 1:
                for pos, p in enumerate(range(start, end+1)):
                    _rut = pos * (rut_end - rut_first)/(size - 1) + rut_first

                    # option1:
                    # self.period_ruts[p].append(np.column_stack((int(v), int(_rut))))

                    # option 2:
                    # edges = (above)
                    ruts = np.full_like(edges, _rut)
                    self.period_ruts[p].append(np.column_stack((edges, ruts)))
            else:
                _rut = rut_end
                p = start
                # option1:
                # self.period_ruts[p].append(np.column_stack((int(v), int(_rut))))

                # option 2:
                ruts = np.full_like(edges, _rut)
                self.period_ruts[p].append(np.column_stack((edges, ruts)))

        self.period_ruts = self.period_ruts.vapply(np.concatenate)
        self.vp_not_task.a[self.type_vp.get_array() == nd.TASK_TYPE] = 0

    def set_weights(self, weigths_ep):

        targets = self.edges[:, 1]
        # assign small default weights with rut (and durations)
        sum_rut = (self.rut_first_vp.get_array() + self.rut_vp.get_array())/2 * \
                  self.duration_vp.get_array() * 0.01
        weigths_ep.a = -np.floor(sum_rut[targets])

        # give weight to maintenances:
        edge_target_type = self.type_vp.get_array()[targets]
        positions = self.instance.get_period_positions()
        last = positions[self.instance.get_param('end')] + 1
        maint_edge = edge_target_type == nd.MAINT_TYPE
        maint_weight = (last - self.period_vp.get_array()[targets[maint_edge]]) + \
                       (last - self.period_end_vp.get_array()[targets[maint_edge]])
        weigths_ep.a[maint_edge] += maint_weight

    def shortest_path(self, node1=None, node2=None, **kwargs):
        target, source = None, None
        if node1 is not None:
            source = self.refs[node1]
        if node2 is not None:
            target = self.refs[node2]
        return gr.shortest_distance(self.g, source=source, target=target, dag=True, **kwargs)

    def filter_by_tasks(self, node1, node2, g=None):
        # returns a graph without task assignments.
        # Except the node1 and node2 and previous nodes to node2
        nodes = [self.refs[node1], self.refs[node2]] + list(self.g.vertex(self.refs[node2]).in_neighbors())
        _temp_vp = self.vp_not_task.copy()
        _temp_vp.a[nodes] = 1
        if g is None:
            g = self.g
        return gr.GraphView(g, vfilt=_temp_vp)

    def filter_by_mask(self, mask, node2, resource):
        vfilt = self.g.new_vp('bool', val=0)
        # only nodes for that resource:
        vfilt.a[self.resource_nodes[resource]] = 1
        index_node = self.refs[node2]
        # we take all predecessors that do not pass the mask function:
        predecessors = self.g.vertex(index_node).in_neighbors()
        _func = lambda n: mask(self.refs_inv[n])
        nodes = [int(n) for n in predecessors if not _func(n)]
        vfilt.a[nodes] = 0
        # all nodes after node2 we take them away too:
        take_out = self.period_end_vp.get_array() > self.period_end_vp[index_node]
        vfilt.a[take_out] = 0
        return gr.GraphView(self.g, vfilt=vfilt)

    def nodes_to_patterns(self, node1, node2, resource, cutoff=1000, max_paths=1000, add_empty=True, mask=None, **kwargs):
        refs = self.refs
        refs_inv = self.refs_inv

        # we shuffle the graph.
        edges = self.edges
        self.g.clear_edges()
        np.random.shuffle(edges)
        self.g.add_edge_list(edges)
        self.g.shrink_to_fit()
        self.g.reindex_edges()
        self.edges = self.g.get_edges()
        self.set_weights(self.weights)

        # we take a view if we have some nodes to take out
        if mask:
            graph = self.filter_by_mask(mask, node2, resource)
        else:
            graph = self.g

        # we take a sample of paths
        log.debug("cutoff size: {}".format(cutoff))
        paths_iterator = gr.all_paths(graph, source=refs[node1], target=refs[node2], cutoff=cutoff)
        # log.debug("got the iterator, now proceding to sample")
        sample = self.iter_sample_fast(paths_iterator, max_paths, max_paths * 10)
        # sample = self.iter_sample_fast(paths_iterator, max_paths, max_paths * 100)
        log.debug("sample size: {}".format(len(sample)))
        if add_empty:
            gfilt = self.filter_by_tasks(node1, node2, g=graph)
            paths_iterator = gr.all_paths(gfilt, source=refs[node1], target=refs[node2])
            # sample += self.iter_sample_fast(paths_iterator, max_paths, max_paths * 100)
            sample += self.iter_sample_fast(paths_iterator, max_paths, max_paths * 10)

        return tl.TupList(sample).vapply(lambda v: [refs_inv[vv] for vv in v])

    def nodes_to_patterns2(self, node1, node2, resource, max_paths=1000, add_empty=True, mask=None, **kwargs):
        refs = self.refs
        refs_inv = self.refs_inv
        if mask:
            graph = self.filter_by_mask(mask, node2, resource)
        else:
            graph = self.g
        # get edges between node1 and node 2 only
        # edges = self.get_edges_between_nodes(node1, node2)

        sample = tl.TupList()
        weights = self.weights.copy()
        arr = self.weights.get_array()
        nodes_window = self.get_nodes_in_window(node1, node2, resource)
        edges_all = self.edges
        targets = edges_all[:, 1]
        sources = edges_all[:, 0]
        relevant_edge = nodes_window[sources] & nodes_window[targets]
        num_edges = np.sum(relevant_edge)
        source = graph.vertex(refs[node1])
        target = graph.vertex(refs[node2])
        for i in range(max_paths):
            weights.a[relevant_edge] = np.floor(arr[relevant_edge] * (1 + np.random.random(num_edges)))
            pattern, edges = gr.shortest_path(graph, source=source, target=target, weights=weights, dag=True)
            _paths = [pattern]
            # log.debug("number of patterns for resource {}: {}".format(node1.resource, len(_paths)))
            sample.extend(_paths)
        if add_empty:
            gfilt = self.filter_by_tasks(node1, node2, g=graph)
            paths_iterator2 = gr.all_paths(gfilt, source=refs[node1], target=refs[node2])
            sample += self.iter_sample_fast(paths_iterator2, max_paths, max_paths * 100)
        return sample.vapply(lambda v: [refs_inv[vv] for vv in v])

    def nodes_to_pattern1(self, node1, node2, all_weights, cutoff=1000, **kwargs):

        # input:
        # graph initialization
        node1 = self.g.vertex(self.refs[node1])  # example source
        node2 = self.g.vertex(self.refs[node2])
        # all_weights = {}  # edge => weight
        # cutoff = 10

        current = node1
        path = []
        visited = set()
        while True:
            if current == node2:
                # we finished!
                return path + [current]
            visited.add(current)  # maybe we should add (current, len(path))
            neighbors = set(current.out_neighbors()) - visited
            if len(path) >= cutoff or not len(neighbors):
                # this path does not reach node2
                # we backtrack
                # and we will never visit this node again
                current = path.pop()
                continue
            if not len(path) and not len(neighbors):
                # there is no path to node2
                # this should not be possible
                return None
            # we haven't reached node2
            # but there is still hope
            path.append(current)
            weights = [all_weights.get((current, n), 1) for n in neighbors]
            _sum = sum(weights)
            weights = [w / _sum for w in weights]
            current = np.random.choice(a=list(neighbors), p=weights)
        return None

    def nodes_to_pattern2(self, node1, node2, resource, mask, errors, **kwargs):
        refs = self.refs
        refs_inv = self.refs_inv
        if mask:
            graph = self.filter_by_mask(mask, node2, resource)
        else:
            graph = self.g

        weights = self.get_weights(node1, node2, errors, resource)

        def find_vertex(node):
            while node.rut is None or node.rut['M'] >= 0:
                try:
                    return graph.vertex(refs[node])
                except (KeyError, ValueError):
                    node.rut['M'] -= 10
                    data = node.get_data()
                    node.jsondump = json.dumps(data, sort_keys=True)
                    node.hash = hash(node.jsondump)
                    log.warning('had to correct node')
            return None

        # TODO: this is failing, sometimes?
        source = find_vertex(node1)
        target = find_vertex(node2)
        if source is None or target is None:
            log.error("There was a problem finding node {} or node {}".format(node1, node2))
            return None
        nodes, edges = gr.shortest_path(graph, source=source, target=target, weights=weights, dag=True)
        return [refs_inv[n] for n in nodes]

    def get_nodes_in_window(self, node1, node2, resource):
        positions = self.instance.get_period_positions()
        first, last = self.instance.get_first_last_period()
        min_period = max(positions[node1.period], positions[first]-1)
        max_period = min(positions[node2.period_end], positions[last]+1)
        return \
            (self.resource_nodes[resource]) & \
            (self.period_vp.get_array() >= min_period) & \
            (self.period_end_vp.get_array() <= max_period)

    # @profile
    def get_weights(self, node1, node2, errors, resource):
        nodes_window = self.get_nodes_in_window(node1, node2, resource)
        edges_all = self.edges
        targets = edges_all[:, 1]
        sources = edges_all[:, 0]
        relevant_edge = nodes_window[sources] & nodes_window[targets]
        w_array = self.weights.get_array()

        # resources:
        #   for periods where there are missing resources:
        #   penalize the nodes that do not assign a mission.
        # TODO: errors could be prefiltered for a given resource (hours and resource)
        #  then I would not need this filter
        tasks_for_resource = self.instance.get_task_candidates(resource=resource)
        resources = errors.get('resources', sd.SuperDict()).\
            keys_tl().vfilter(lambda v: v[0] in tasks_for_resource)
        weights_resource = np.zeros_like(w_array)
        arr_per_end = self.period_end_vp.get_array()
        arr_assign = self.assgin_vp.get_array()
        arr_per = self.period_vp.get_array()
        positions = self.instance.get_period_positions()
        tasks = self._equiv_task
        for task, period in resources:
            t = tasks[task]
            p = positions[period]
            relevant_node = nodes_window & (arr_per <= p) & (arr_per_end >= p) & (arr_assign != t)
            edges = relevant_edge & relevant_node[targets]
            weights_resource[edges] += 1
        #  hours:
        #   add weight for negative rut.
        clusters = self.instance.get_cluster_candidates(resource=resource).list_reverse()[resource]
        clusters = set(clusters)
        hours_periods = \
            errors.get('hours', sd.SuperDict()).\
            kfilter(lambda k: k[0] in clusters)

        weights_hours = np.zeros_like(w_array)
        arr_per = self.period_vp.get_array()
        for (k, period), v in hours_periods.items():
            refs = self.period_ruts[positions[period]][:,0]
            ruts = self.period_ruts[positions[period]][:,1]
            final_rut = ruts + v
            negative = final_rut < 0

            # option 2:
            edges = refs
            extra_hours = np.zeros_like(edges)
            extra_hours[negative] = -final_rut[negative]
            weights_hours[edges] += extra_hours

            # option 1:
            # nodes_arr = refs
            # extra_hours = np.zeros_like(arr_per)
            # extra_hours[nodes_arr[negative]] = -final_rut[negative]
            # weights_hours[relevant_edge] += extra_hours[targets[relevant_edge]]


        # maintenances!
        #   add weight for extra maintenance over limit.
        capacity = errors.get('capacity', sd.SuperDict())
        positions = self.instance.get_period_positions()
        arr_per_end = self.period_end_vp.get_array()
        arr_type = self.type_vp.get_array()
        arr_per = self.period_vp.get_array()
        weights_maints = np.zeros_like(w_array)
        for _, period in capacity:
            p = positions[period]
            relevant_node = nodes_window \
                    & (arr_type == nd.MAINT_TYPE)\
                    & (arr_per <= p)\
                    & (arr_per_end >= p)
            edges = relevant_edge & relevant_node[targets]
            weights_maints[edges] += 1

        # summarize
        # TODO: pass these weights from somewhere
        weights = self.g.new_ep('int')
        self.set_weights(weights)
        weights_arr = weights.get_array()
        weights.a[relevant_edge] += weights_arr[relevant_edge] + \
                                   weights_resource[relevant_edge] * 20000 + \
                                   weights_hours[relevant_edge] * 100 + \
                                   weights_maints[relevant_edge] * 30000
        return weights

    def get_weights_old(self, node1, node2, errors, resource):
        # get edges between node1 and node 2 only
        nodes_window = self.get_nodes_in_window(node1, node2, resource)
        edges_all = self.edges
        targets = edges_all[:, 1]
        sources = edges_all[:, 0]
        relevant_edge = nodes_window[sources] & nodes_window[targets]
        weigths_frac = np.ones_like(self.weights.get_array(), dtype='float')
        hours_periods = errors.get('hours', sd.SuperDict())
        clust_cand = self.instance.get_cluster_candidates().vapply(set)
        clust_hours = \
            hours_periods.\
            kfilter(lambda k: resource in clust_cand[k[0]]).\
            to_tuplist().\
            to_dict(2, indices=[1]).\
            vapply(min)
        weigths_frac = self.modify_weights_by_hours(weigths_frac, nodes_window, relevant_edge, targets, clust_hours)
        weigths_frac = self.modify_weights_by_resource(weigths_frac, nodes_window, relevant_edge, targets, errors.get('resources'))
        weigths_frac = self.modify_weights_by_capacity(weigths_frac, nodes_window, relevant_edge, targets, errors.get('capacity'))
        weights = self.weights.copy()
        weights_arr = weights.get_array()
        weights.a[relevant_edge] = np.floor(weights_arr[relevant_edge] *
                                            weigths_frac[relevant_edge] +
                                            (
                                                    weights_arr[relevant_edge] *
                                                    np.random.random(np.sum(relevant_edge))
                                             )
                                            )
        return weights

    def modify_weights_by_capacity(self, weights, nodes_window, relevant_edge, targets, capacity):
        if not capacity:
            return weights
        # capacity negative is bad.
        # capacity => multiply by X to each edge if maintenance has started? or it's going on
        positions = self.instance.get_period_positions()
        arr_per_end = self.period_end_vp.get_array()
        arr_type = self.type_vp.get_array()
        arr_per = self.period_vp.get_array()
        for _, period in capacity:
            p = positions[period]
            relevant_node = nodes_window \
                    & (arr_type == nd.MAINT_TYPE)\
                    & (arr_per <= p)\
                    & (arr_per_end >= p)
            edges = relevant_edge & relevant_node[targets]
            weights[edges] = weights[edges] * 1.1
        return weights

    def modify_weights_by_hours(self, weights, nodes_window, relevant_edge, targets, hours_periods):
        if not hours_periods:
            return weights
        # negative hours are bad
        # hours => multiply by X to each edge, depending on lower rut
        positions = self.instance.get_period_positions()
        pos, hour = zip(*((positions[k], v) for k, v in hours_periods.items()))
        num_periods = self.instance.get_param('num_period')
        old_ruts = np.zeros(num_periods)
        old_ruts[np.array(pos)] = np.array(hour)
        arr_per = self.period_vp.get_array()
        # periods_pos = tl.TupList(periods).vapply(lambda v: positions[v])
        old_rut_node = np.zeros_like(arr_per)
        old_rut_node[nodes_window] = old_ruts[arr_per[nodes_window]]
        ruts = self.rut_vp.get_array()
        final_rut = old_rut_node + ruts
        less_than_zero = final_rut < 0
        weight_hours = np.ones_like(final_rut, dtype='float')
        weight_hours[less_than_zero] = final_rut[less_than_zero]**2
        _max = np.max(weight_hours)
        weight_hours[less_than_zero] = weight_hours[less_than_zero] / _max + 1
        weights[relevant_edge] = weights[relevant_edge] * weight_hours[targets[relevant_edge]]
        return weights

    def modify_weights_by_resource(self, weights, nodes_window, relevant_edge, targets, resources):
        if not resources:
            return weights
        # positive resources are bad
        # resources => divide by X to each edge if mission is assigned
        # TODO: filter tasks to candidate's task?
        # task, period = zip(*((tasks[k], positions[v]) for k, v in resources.keys()))
        # deficit = resources.values()
        arr_per_end = self.period_end_vp.get_array()
        arr_assign = self.assgin_vp.get_array()
        arr_per = self.period_vp.get_array()

        positions = self.instance.get_period_positions()
        tasks = self._equiv_task
        for task, period in resources:
            t = tasks[task]
            p = positions[period]
            relevant_node = \
                nodes_window & (arr_assign == t) & (arr_per <= p) & (arr_per_end >= p)
            edges = relevant_edge & relevant_node[targets]
            weights[edges] = weights[edges] / 1.1
        return weights


    def to_file(self, path):
        raise NotImplementedError("does not work now")

    @classmethod
    def from_file(cls, path, instance, resource):
        graph = DAG(instance, resource)
        _path = os.path.join(path, 'cache_info_{}.json'.format(resource))
        graph.refs_inv = di.load_data(_path)
        graph.refs = graph.refs_inv.reverse()
        graph.g = gr.load_graph(os.path.join(path, 'graph_{}_.gt'.format(resource)))
        return graph

    def draw(self, not_show_None=True, edge_label=None, node_label=None, tikz=False, filename=None):
        y_ranges = dict(VG=lambda: rn.uniform(10, 15) * (1 - 2 * (rn.random() > 0.5)),
                        VI=lambda: rn.uniform(5, 10) * (1 - 2 * (rn.random() > 0.5)),
                        VS=lambda: rn.uniform(0, 5) * (1 - 2 * (rn.random() > 0.5))
                        )
        instance = self.instance
        refs_inv = self.refs_inv
        g = self.g
        max_rut = instance.get_maintenances('max_used_time')['M']

        def get_y_mission(v):
            if not refs_inv[v].rut:
                return 0
            return -(refs_inv[v].rut['M'] / max_rut - 0.5) * 20

        colors = \
            {'VG': '#4cb33d',
             'VI': '#00c8c3',
             'VS': '#31c9ff',
             'M': '#878787',
             'VG+VI': '#EFCC00',
             'VG+VS': '#EFCC00',
             'VG+VI+VS': '#EFCC00',
             'VI+VS': '#EFCC00'}
        extra_colors = {refs_inv[v].assignment: rn.choice(['#31c9ff', '#00c8c3', '#4cb33d']) for v in g.vertices()}
        # empty assignments we put in white:
        extra_colors[''] = '#ffffff'
        colors = {**extra_colors, **colors}

        first, last = instance.get_first_last_period()

        if not_show_None:
            keep_node = g.new_vp('bool')
            for v in g.vertices():
                if refs_inv[v].rut is None and first <= refs_inv[v].period <= last:
                    keep_node[v] = 0
                else:
                    keep_node[v] = 1
            g_filt = gr.GraphView(g, vfilt=keep_node)
        else:
            g_filt = g

        pos = g_filt.new_vp('vector<float>')
        size = g_filt.new_vp('double')
        shape = g_filt.new_vp('string')
        color = g_filt.new_vp('string')
        vertex_text = g_filt.new_vp('string')
        assignment = g_filt.new_ep('string')

        if not edge_label:
            def edge_label(tail):
                a = tail.assignment
                if a is None:
                    return ""
                return a

        for e in g_filt.edges():
            assignment[e] = edge_label(refs_inv[e.target()])

        if not node_label:
            node_label = lambda node: node.period

        for v in g_filt.vertices():
            vertex_text[v] = node_label(refs_inv[v])
            x = instance.get_dist_periods(first, refs_inv[v].period)
            a = refs_inv[v].assignment
            if a in y_ranges:
                y = y_ranges.get(a, lambda: 0)()
            else:
                y = get_y_mission(v)
            pos[v] = (x, y)
            size[v] = 2
            shape[v] = 'circle'
            color[v] = colors.get(a, 'red')

        options = dict(pos=pos, vertex_text=vertex_text, edge_text=assignment,
                       vertex_shape=shape, vertex_fill_color=color)
        if not tikz:
            gr.graph_draw(g=g_filt, **options)
        else:
            self.graph_draw_tikz(g=g_filt, **options, filename=filename)

    @staticmethod
    def graph_draw_tikz(g, pos, vertex_text, edge_text, vertex_shape, vertex_fill_color, filename):
        import network2tikz as nt
        import webcolors as wc

        nodes = [int(v) for p, v in enumerate(g.vertices())]
        _edges = list(g.edges())
        edges = [(int(e.source()), int(e.target())) for e in _edges]
        visual_style = {}
        visual_style['vertex_color'] = [wc.hex_to_rgb(vertex_fill_color[v]) for v in nodes]
        visual_style['edge_label'] = [edge_text[e] for e in _edges]
        visual_style['vertex_label'] = [vertex_text[v] for v in nodes]
        visual_style['layout'] = {v: (pos[v][0], np.cbrt(-pos[v][1])) for p, v in enumerate(nodes)}
        visual_style['vertex_size'] = 1.5
        visual_style['keep_aspect_ratio'] = False
        visual_style['canvas'] = (10, 10)
        # visual_style['node_opacity'] = 0.5
        nt.plot(network=(nodes, edges), **visual_style, filename=filename)


def generate_graph_mcluster(instance, resources):
    nodes_ady = sd.SuperDict()
    res_nodes = sd.SuperDict()
    for r in resources:
        source = nd.get_source_node(instance, r)
        nodes_ady = source.walk_over_nodes(nodes_ady)
    nodes_artificial = get_artificial_nodes(nodes_ady)
    nodes_ady.kvapply(lambda k, v: v.extend(nodes_artificial.get(k, [])))
    graph = GraphTool(instance=instance, nodes_ady=nodes_ady)
    num_period = instance.get_param('num_period')
    for r in resources:
        source = nd.get_source_node(instance, r)
        shortest_path = gr.shortest_distance(g=graph.g, source=graph.refs[source], dag=True, max_dist=num_period+2)
        res_nodes[r] = shortest_path.get_array() <= num_period
    graph.resource_nodes = res_nodes
    return graph

import graph_tool.all as gr
from pytups import superdict as sd, tuplist as tl
import numpy as np
import data.data_input as di
import os
from .core import DAG
import patterns.node as nd
import random as rn
import package.instance as inst
import logging as log
import itertools

# installing graph-tool and adding it to venv:
# https://git.skewed.de/count0/graph-tool/wikis/installation-instructions
# https://jolo.xyz/blog/2018/12/07/installing-graph-tool-with-virtualenv


class GraphTool(DAG):
    # TODO: only create a graph per aircraft type to reuse some nodes??

    def get_create_node(self, node):
        try:
            return self.refs[node]
        except KeyError:
            v = self.g.add_vertex()
            self.refs[node] = int(v)
            return v

    def __init__(self, instance: inst.Instance, resource):
        super().__init__(instance, resource)
        nodes_ady = self.g
        self.g = gr.Graph()
        self.refs = sd.SuperDict()
        for n, n2_list in nodes_ady.items():
            v1 = self.get_create_node(n)
            for n2 in n2_list:
                v2 = self.get_create_node(n2)
                e = self.g.add_edge(v1, v2, add_missing=False)
        self.refs_inv = self.refs.reverse()

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

        # create dictionary to filter nodes that have no tasks
        self.vp_not_task = self.g.new_vp('bool', val=1)
        self.period_vp = self.g.new_vp('int')
        self.period_end_vp = self.g.new_vp('int')
        self.type_vp = self.g.new_vp('int')
        self.assgin_vp = self.g.new_vp('int')
        self.rut_vp = self.g.new_vp('int')
        positions = self.instance.get_period_positions()
        self._equiv_task = {k: v +1 for v, k in enumerate(self.instance.get_tasks())}
        self._equiv_task[''] = 0
        self._equiv_task['M'] = -1
        for v in self.g.vertices():
            node = self.refs_inv[v]
            self.type_vp[v] = node.type
            self.period_vp[v] = positions[node.period]
            self.period_end_vp[v] = positions[node.period_end]
            self.assgin_vp[v] = self._equiv_task[node.assignment]
            self.rut_vp[v] = node.rut['M'] if node.rut else 0

        self.vp_not_task.a[self.type_vp.get_array() == nd.TASK_TYPE] = 0
        self.set_weights()

    def set_weights(self):
        self.weights = self.g.new_ep('int')
        multiplier = 100
        durations = (self.period_end_vp.get_array() -
                     self.period_vp.get_array() + 1) * multiplier
        targets = self.g.get_edges()[:, 1]
        self.weights.a = durations[targets]
        edge_target_type = self.type_vp.get_array()[targets]
        default = 0.9 * multiplier
        empty_edge = edge_target_type == nd.EMPTY_TYPE
        self.weights.a[empty_edge] = default
        last = max(self.period_vp.get_array())
        maint_edge = edge_target_type == nd.MAINT_TYPE
        maint_weight = (last - self.period_vp.get_array()[targets] - 1)
        self.weights.a[maint_edge] += maint_weight[maint_edge]

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
        for v in nodes:
            _temp_vp[v] = 1
        if g is None:
            g = self.g
        return gr.GraphView(g, vfilt=_temp_vp)

    def filter_by_mask(self, mask, node2):
        vfilt = self.g.new_vp('bool', val=1)
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

    def nodes_to_patterns(self, node1, node2, cutoff=1000, max_paths=1000, add_empty=True, mask=None, **kwargs):
        refs = self.refs
        refs_inv = self.refs_inv

        # we shuffle the graph.
        edges = self.g.get_edges()
        self.g.clear_edges()
        np.random.shuffle(edges)
        self.g.add_edge_list(edges)
        self.g.shrink_to_fit()
        self.g.reindex_edges()
        self.set_weights()

        # we take a view if we have some nodes to take out
        if mask:
            graph = self.filter_by_mask(mask, node2)
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
        # node = node1
        # refs.keys_tl().\
        #     vfilter(lambda v: v.period==node.period and v.assignment==node.assignment).\
        #     vapply(lambda v: (v.rut, v.ret, v.type))
        # node.rut, node.ret, node.type

    def nodes_to_patterns2(self, node1, node2, max_paths=1000, add_empty=True, mask=None, **kwargs):
        refs = self.refs
        refs_inv = self.refs_inv
        if mask:
            graph = self.filter_by_mask(mask, node2)
        else:
            graph = self.g
        # get edges between node1 and node 2 only
        # edges = self.get_edges_between_nodes(node1, node2)

        sample = tl.TupList()
        for i in range(max_paths):
            weights = self.weights.copy()
            arr = weights.get_array()
            weights.a = np.floor(arr * (1 + np.random.random(self.g.num_edges())))
            patterns = gr.all_shortest_paths(graph, source=refs[node1], target=refs[node2], weights=weights, dag=True)
            _paths = list(patterns)
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

    def nodes_to_pattern2(self, node1, node2, mask, errors, **kwargs):
        refs = self.refs
        refs_inv = self.refs_inv
        if mask:
            graph = self.filter_by_mask(mask, node2)
        else:
            graph = self.g

        weights = self.get_weights(node1, node2, errors)

        # source = graph.vertex(refs[node1])
        # TODO: this is failing, sometimes?
        try:
            source = graph.vertex(refs[node1])
        except KeyError:
            log.error("There was a problem finding a node: {}".format(node1))
            return None
        target = graph.vertex(refs[node2])
        nodes, edges = gr.shortest_path(graph, source=source, target=target, weights=weights, dag=True)
        return [refs_inv[n] for n in nodes]

    def get_weights(self, node1, node2, errors):
        # get edges between node1 and node 2 only
        positions = self.instance.get_period_positions()
        first, last = self.instance.get_first_last_period()
        min_period = max(positions[node1.period], positions[first])
        max_period = min(positions[node2.period_end], positions[last])
        nodes_window = \
            (self.period_vp.get_array() >= min_period) & \
            (self.period_end_vp.get_array() <= max_period)
        edges_all = self.g.get_edges()
        relevant_edge = nodes_window[edges_all[:, 0]] & nodes_window[edges_all[:, 1]]
        weigths_frac = np.ones_like(self.weights.get_array(), dtype='float')
        hours_periods = errors.get('hours', sd.SuperDict())
        clust_cand = self.instance.get_cluster_candidates().vapply(set)
        clust_hours = \
            hours_periods.\
            kfilter(lambda k: node1.resource in clust_cand[k[0]]).\
            to_tuplist().\
            to_dict(2, indices=[1]).\
            vapply(min)
        weigths_frac = self.modify_weights_by_hours(weigths_frac, nodes_window, relevant_edge, clust_hours)
        weigths_frac = self.modify_weights_by_resource(weigths_frac, nodes_window, relevant_edge, errors.get('resources'))
        weigths_frac = self.modify_weights_by_capacity(weigths_frac, nodes_window, relevant_edge, errors.get('capacity'))
        weights = self.weights.copy()
        weights_arr = weights.get_array()
        weights.a[relevant_edge] = np.floor(weights_arr[relevant_edge] *
                                            weigths_frac[relevant_edge] +
                                            (
                                                    weights_arr[relevant_edge] *
                                                    0.5 * np.random.random(np.sum(relevant_edge))
                                             )
                                            )
        return weights

    def modify_weights_by_capacity(self, weights, nodes_window, relevant_edge, capacity):
        if not capacity:
            return weights
        # capacity negative is bad.
        # capacity => multiply by X to each edge if maintenance has started? or it's going on
        positions = self.instance.get_period_positions()
        arr_per_end = self.period_end_vp.get_array()
        arr_type = self.type_vp.get_array()
        arr_per = self.period_vp.get_array()
        edges_all = self.g.get_edges()
        for _, period in capacity:
            p = positions[period]
            relevant_node = nodes_window \
                    & (arr_type == nd.MAINT_TYPE)\
                    & (arr_per <= p)\
                    & (arr_per_end >= p)
            edges = relevant_edge & relevant_node[edges_all[:, 1]]
            weights[edges] = weights[edges] * 1.1
        return weights

    def modify_weights_by_hours(self, weights, nodes_window, relevant_edge, hours_periods):
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
        less_than_zero = final_rut<0
        weight_hours = np.ones_like(final_rut, dtype='float')
        weight_hours[less_than_zero] = final_rut[less_than_zero]**2
        _max = max(weight_hours)
        weight_hours[less_than_zero] = weight_hours[less_than_zero] / _max + 1
        targets = self.g.get_edges()[:, 1]
        weights[relevant_edge] = weights[relevant_edge] * weight_hours[targets[relevant_edge]]
        return weights

    def modify_weights_by_resource(self, weights, nodes_window, relevant_edge, resources):
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
        edges_all = self.g.get_edges()

        positions = self.instance.get_period_positions()
        tasks = self._equiv_task
        for task, period in resources:
            t = tasks[task]
            p = positions[period]
            relevant_node = \
                nodes_window & (arr_assign == t) & (arr_per <= p) & (arr_per_end >= p)
            edges = relevant_edge & relevant_node[edges_all[:, 1]]
            weights[edges] = weights[edges] / 1.1
        return weights


    def to_file(self, path):
        name = 'cache_info_{}'.format(self.resource)
        graph_file = os.path.join(path, 'graph_{}_.gt'.format(self.resource))
        _data = self.refs_inv.vapply(lambda v: v.get_data()).to_tuplist()
        di.export_data(path, _data, name=name)
        self.g.save(graph_file)

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
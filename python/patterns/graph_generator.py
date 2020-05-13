from pytups import superdict as sd, tuplist as tl
import patterns.node as nd
import random as rn
import numpy as np
import data.data_input as di
import os

def graph_factory(instance, resource, options=None):
    if not options:
        options = {}
    return GraphTool(instance, resource)


class DAG(object):

    def __init__(self, instance, resource):
        self.source = nd.get_source_node(instance, resource)
        self.sink = nd.get_sink_node(instance, resource)
        self.instance = instance
        self.resource = resource
        # We generate the graph by using "nodes" module
        # We represent the graph with an adjacency list
        nodes_ady = self.source.walk_over_nodes()

        # Here, there is some post-processing (more nodes) to use the graph better
        # 1. for each period, for each assignment,
        # 2. tie all nodes to a single node with the same period, same assignment but rut and ret to None

        # We only create artificial nodes for all not None nodes.
        nodes_artificial = self.get_artificial_nodes(nodes_ady)
        self.g = nodes_ady.kvapply(lambda k, v: v + nodes_artificial.get(k, []))

        # this should deal with almost all nodes. maybe not the last one?
        self.refs = self.g.kvapply(lambda k: k)
        self.refs_inv = self.refs

        pass

    def draw(self):
        pass

    def shortest_path(self, node1=None, node2=None, **kwargs):
        pass

    def nodes_to_patterns(self, node1, node2, vp_not_task, cutoff=1000, max_paths=1000, add_empty=True,  **kwargs):
        pass

    def notes_to_pattern(self, node1, node2, all_weights, cutoff=1000, **kwargs):
        pass

    @staticmethod
    def iter_sample_fast(iterable, samplesize, max_iterations=9999999):
        """

        :param iter iterable:
        :param samplesize:
        :return:
        # https://stackoverflow.com/questions/12581437/python-random-sample-with-a-generator-iterable-iterator
        """
        results = []
        iterator = iter(iterable)
        # Fill in the first samplesize elements:
        try:
            for _ in range(samplesize):
                results.append(next(iterator))
        except StopIteration:
            return results
        rn.shuffle(results)  # Randomize their positions
        for i, v in enumerate(iterator, samplesize):
            r = rn.randint(0, i)
            if r < samplesize:
                results[r] = v  # at a decreasing rate, replace random items
            if i >= max_iterations:
                break
        return results

    def get_all_patterns(self):
        return self.nodes_to_patterns(self.g, self.refs, self.refs_inv, self.source, self.sink)

    def get_artificial_nodes(self, nodes_ady):
        return \
            nodes_ady.g.keys_tl().\
            vfilter(lambda v: v.assignment is not None).\
            vapply(lambda v: (v.period, v.period_end, v.assignment, v.type, v)).\
            to_dict(result_col=4).list_reverse().vapply(lambda v: v[0]).\
            vapply(lambda v: dict(instance=self.instance, resource=self.resource,
                                  period=v[0], period_end=v[1], assignment=v[2],
                                  rut=None, ret=None, type=v[3])).\
                vapply(lambda v: [nd.Node(**v)])

    def to_file(self, path):
        pass

    def from_file(self, path, instance, resource):
        pass

# installing graph-tool and adding it to venv:
# https://git.skewed.de/count0/graph-tool/wikis/installation-instructions
# https://jolo.xyz/blog/2018/12/07/installing-graph-tool-with-virtualenv

try:
    import graph_tool.all as gr

except ImportError:
    pass
else:
    class GraphTool(DAG):

        def get_create_node(self, node):
            try:
                return self.refs[node]
            except KeyError:
                v = self.g.add_vertex()
                self.refs[node] = int(v)
                return v

        def __init__(self, instance, resource):
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
            keep_node = self.g.new_vp('bool', val=1)
            for v in nodes:
                keep_node[v] = 0
            self.g = gr.GraphView(self.g, vfilt=keep_node)

            # create dictionary to filter nodes that have no tasks
            self.vp_not_task = self.g.new_vp('bool', val=1)
            for v in self.g.vertices():
                if self.refs_inv[v].type == nd.TASK_TYPE:
                    self.vp_not_task[v] = 0

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

        def shortest_path(self, node1=None, node2=None, **kwargs):
            target, source = None, None
            if node1 is not None:
                source = self.refs[node1]
            if node2 is not None:
                target = self.refs[node2]
            return gr.shortest_distance(self.g, source=source, target=target, dag=True, **kwargs)

        def filter_by_tasks(self, node1, node2):
            # returns a graph without task assignments.
            # Except the node1 and node2 and previous nodes to node2
            nodes = [self.refs[node1], self.refs[node2]] + list(self.g.vertex(self.refs[node2]).in_neighbors())
            _temp_vp = self.vp_not_task.copy()
            for v in nodes:
                _temp_vp[v] = 1
            return gr.GraphView(self.g, vfilt=_temp_vp)

        def nodes_to_patterns(self, node1, node2, cutoff=1000, max_paths=1000, add_empty=True, **kwargs):
            graph = self.g
            refs = self.refs
            refs_inv = self.refs_inv
            paths_iterator = gr.all_paths(graph, source=refs[node1], target=refs[node2], cutoff=cutoff)
            sample = self.iter_sample_fast(paths_iterator, max_paths, max_paths * 100)
            sample2 = []
            if add_empty:
                gfilt = self.filter_by_tasks(node1, node2)
                paths_iterator2 = gr.all_paths(gfilt, source=refs[node1], target=refs[node2])
                sample2 = self.iter_sample_fast(paths_iterator2, max_paths, max_paths * 100)
            return tl.TupList(sample + sample2).vapply(lambda v: [refs_inv[vv] for vv in v])
            # node = node1
            # refs.keys_tl().\
            #     vfilter(lambda v: v.period==node.period and v.assignment==node.assignment).\
            #     vapply(lambda v: (v.rut, v.ret, v.type))
            # node.rut, node.ret, node.type

        def nodes_to_pattern(self, node1, node2, all_weights, cutoff=1000, **kwargs):

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
                weights = [w/_sum for w in weights]
                current = np.random.choice(a=list(neighbors), p=weights)

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

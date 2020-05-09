import package.instance as inst
import data.data_input as di

import pytups.superdict as sd
import pytups.tuplist as tl
import numpy.random as rn
import patterns.node as nd
import random
import os
import numpy as np

import graph_tool.all as gr


# installing graph-tool and adding it to venv:
# https://git.skewed.de/count0/graph-tool/wikis/installation-instructions
# https://jolo.xyz/blog/2018/12/07/installing-graph-tool-with-virtualenv


def walk_over_nodes(node: nd.Node, get_nodes_only=False):
    """

    :param node: node from where we start the DFS
    :param get_nodes_only: True if we only care about taking the arcs.
        False we return all paths
    :return: all arcs or all paths
    """

    remaining_nodes = [(node, [])]
    # we store the neighbors of visited nodes, not to recalculate them
    cache_neighbors = sd.SuperDict()
    i = 0
    final_paths = []
    last_node = nd.get_sink_node(node.instance, node.resource)
    ct = node.instance.compare_tups
    fixed = \
        node.instance.\
        get_fixed_states(resource=node.resource, filter_horizon=True).\
        to_start_finish(ct, pp=2)
    if len(fixed):
        # we start from the initial fixed nodes
        # instead of starting from the initial node
        neighbors = get_fixed_initial_nodes(node, fixed)
        cache_neighbors[node] = neighbors
        # we replace remaining_nodes, this losing the origin.
        remaining_nodes = [(n, [node]) for n in neighbors]

    while len(remaining_nodes) and i < 10000000:
        i += 1
        node, path = remaining_nodes.pop()
        # we need to make a copy of the path
        path = path + [node]
        if node == last_node:
            # if last_node reached, save the path and go back
            final_paths.append(path)
            continue
        # we're not in the last_node.
        neighbors = cache_neighbors.get(node)
        if neighbors is None:
            # I don't have any cache of the node.
            # I'll get neighbors and do cache
            cache_neighbors[node] = neighbors = node.get_adjacency_list(only_next_period=True)
        elif get_nodes_only:
            # we have the cache AND we don't want paths, we leave
            continue
        remaining_nodes += [(n, path) for n in neighbors]
        # log.debug("iteration: {}, remaining: {}, stored: {}".format(i, len(remaining_nodes), len(cache_neighbors)))
    if get_nodes_only:
        return cache_neighbors
    return final_paths


def get_fixed_initial_nodes(node, fixed):
    _, assignment, start, end = fixed[0]
    # we take out the source from the list
    if assignment == 'M':
        # if the fixed assignment is a maintenance
        # we just need to create a small(er) maintenance node
        return [node.create_adjacent(assignment, 1, node.dif_period_end(end), 0)]
    # if the fixed assignment is a task
    # we have to look for nodes that comply with the fixed period
    neighbors = node.get_adjacency_list(only_next_period=True)
    return \
        tl.TupList(neighbors). \
            vfilter(lambda v: v.period <= start and
                              v.period_end >= end and
                              v.assignment == assignment and
                              v.type == nd.TASK_TYPE)


def get_create_node(refs, g, n):
    try:
        return refs[n]
    except KeyError:
        v = g.add_vertex()
        refs[n] = int(v)
        return v


def adjacency_to_graph(nodes_ady):
    # TODO: make this faster
    g = gr.Graph()
    refs = sd.SuperDict()
    for n, n2_list in nodes_ady.items():
        v1 = get_create_node(refs, g, n)
        for n2 in n2_list:
            v2 = get_create_node(refs, g, n2)
            e = g.add_edge(v1, v2, add_missing=False)
    return g, refs


def get_all_patterns(graph, refs, refs_inv, instance, resource):
    source = nd.get_source_node(instance, resource)
    sink = nd.get_sink_node(instance, resource)

    return nodes_to_patterns(graph, refs, refs_inv, source, sink)


def draw_graph(instance, g, refs_inv=None, not_show_None=True,
               edge_label=None, node_label=None,
               tikz=False, filename=None):
    y_ranges = dict(VG=lambda: rn.uniform(10, 15) * (1 - 2 * (rn.random() > 0.5)),
                    VI=lambda: rn.uniform(5, 10) * (1 - 2 * (rn.random() > 0.5)),
                    VS=lambda: rn.uniform(0, 5) * (1 - 2 * (rn.random() > 0.5))
                    )
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
    extra_colors['']  = '#ffffff'
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
    assignment = g.new_ep('string')

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
        graph_draw_tikz(g=g_filt, **options, filename=filename)

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


    nt.plot(network = (nodes, edges), **visual_style, filename=filename)


def shortest_path(graph, refs, node1=None, node2=None, distances=None, **kwargs):
    target, source = None, None
    if node1 is not None:
        source = refs[node1]
    if node2 is not None:
        target = refs[node2]
    if source and target and distances:
        return distances[source][target]
    return gr.shortest_distance(graph, source=source, target=target, dag=True, **kwargs)

def get_cleaned_graph(g, node1, node2, refs, refs_inv, vp_not_task):
    # returns a graph without task assignments.
    # Except the node1 and node2 and previous nodes to node2
    nodes = [refs[node1], refs[node2]] + list(g.vertex(refs[node2]).in_neighbors())
    _temp_vp = vp_not_task.copy()
    for v in nodes:
        _temp_vp[v] = 1
    return gr.GraphView(g, vfilt=_temp_vp)

def nodes_to_patterns(graph, refs, refs_inv, node1, node2, vp_not_task,
                      cutoff=1000, max_paths=1000, add_empty=True,  **kwargs):
    paths_iterator = gr.all_paths(graph, source=refs[node1], target=refs[node2], cutoff=cutoff)
    sample = iter_sample_fast(paths_iterator, max_paths, max_paths*100)
    sample2 = []
    if add_empty:
        gfilt = get_cleaned_graph(graph, node1, node2, refs, refs_inv, vp_not_task)
        paths_iterator2 = gr.all_paths(gfilt, source=refs[node1], target=refs[node2])
        sample2 = iter_sample_fast(paths_iterator2, max_paths, max_paths * 100)
    return tl.TupList(sample + sample2).vapply(lambda v: [refs_inv[vv] for vv in v])
    node = node1
    refs.keys_tl().\
        vfilter(lambda v: v.period==node.period and v.assignment==node.assignment).\
        vapply(lambda v: (v.rut, v.ret, v.type))
    node.rut, node.ret, node.type

def nodes_to_pattern(graph, refs, refs_inv, node1, node2, all_weights,
                      cutoff=1000, **kwargs):

    # input:
    # graph = gr.Graph()
    # graph initialization
    node1 = graph.vertex(refs[node1])  # example source
    node2 = graph.vertex(refs[node2])
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


def state_to_node(instance, resource, state):
    return nd.Node(instance=instance, resource=resource, **state)


def get_graph_of_resource(instance, resource):
    source = nd.get_source_node(instance, resource)
    sink = nd.get_sink_node(instance, resource)

    # We generate the graph by using "nodes" module
    # We represent the graph with an adjacency list
    nodes_ady = walk_over_nodes(source, get_nodes_only=True)

    # Here, there is some post-processing (more nodes) to use the graph better
    # 1. for each period, for each assignment,
    # 2. tie all nodes to a single node with the same period, same assignment but rut and ret to None

    # We only create artificial nodes for all not None nodes.
    nodes_artificial = \
        nodes_ady.keys_tl().\
        vfilter(lambda v: v.assignment is not None).\
        vapply(lambda v: (v.period, v.period_end, v.assignment, v.type, v)).\
        to_dict(result_col=4).list_reverse().vapply(lambda v: v[0]).\
        vapply(lambda v: dict(instance=instance, resource=resource,
                              period=v[0], period_end=v[1], assignment=v[2],
                              rut=None, ret=None, type=v[3])).\
            vapply(lambda v: [nd.Node(**v)])

    nodes_ady_2 = nodes_ady.kvapply(lambda k, v: v + nodes_artificial.get(k, []))

    # Then, when exploiting this, we will filter nodes with low enough ret and rut.
    # compared to the cycle limit on the next maintenance.

    # We create a graph-tool version of the graph
    # and links between the original nodes and the graph tool ones
    g, refs = adjacency_to_graph(nodes_ady_2)
    refs_inv = refs.reverse()

    # delete nodes with infinite distance to sink:
    g.set_reversed(is_reversed=True)
    distances = shortest_path(graph=g, refs=refs, node1=sink)
    max_dist = instance.get_dist_periods(*instance.get_first_last_period()) + 5
    nodes = [n for n in g.vertices()
             if distances[n] > max_dist and
             refs_inv[n].rut is not None]
    g.set_reversed(is_reversed=False)
    keep_node = g.new_vp('bool', val=1)
    for v in nodes:
        keep_node[v] = 0
    g = gr.GraphView(g, vfilt=keep_node)

    vp_not_task = g.new_vp('bool', val=1)
    for v in g.vertices():
        if refs_inv[v].type == nd.TASK_TYPE:
            vp_not_task[v] = 0

    return sd.SuperDict(graph=g, refs=refs, refs_inv=refs_inv,
                        source=source, sink=sink, distances=None, vp_not_task=vp_not_task)


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
    random.shuffle(results)  # Randomize their positions
    for i, v in enumerate(iterator, samplesize):
        # log.debug('In node number: {}'.format(i))
        r = random.randint(0, i)
        if r < samplesize:
            results[r] = v  # at a decreasing rate, replace random items
        if i >= max_iterations:
            break
    return results


def export_graph_data(path, data, resource):

    name = 'cache_info_{}'.format(resource)
    graph_file = os.path.join(path, 'graph_{}_.gt'.format(resource))
    _data = data[resource]['refs_inv'].vapply(lambda v: v.get_data()).to_tuplist()
    di.export_data(path, _data, name = name)
    data[resource]['graph'].save(graph_file)


def import_graph_data(path, resource):
    _path = os.path.join(path, 'cache_info_{}.json'.format(resource))
    refs_inv = di.load_data(_path)
    graph = gr.load_graph(os.path.join(path, 'graph_{}_.gt'.format(resource)))
    return sd.SuperDict(graph=graph, refs_inv=refs_inv)


def get_patterns_into_dictup(res_patterns):
    return \
        {(res, p): pattern
         for res, _pat_list in res_patterns.items()
         for p, pattern in enumerate(_pat_list)
         }


def get_assignments_from_patterns(instance, combos, maint='M'):
    _range = instance.get_periods_range
    _dist = instance.get_dist_periods
    rut_or_None = lambda rut: rut[maint] if rut is not None else None
    _range_backup = sd.SuperDict(combos).to_tuplist().\
        take(2).unique2().\
        vapply(lambda v: (v.period, v.period_end)).unique2().\
        to_dict(None).\
        vapply(lambda v: _range(*v))

    info = tl.TupList(
        (res, p, period, e.assignment, e.type, rut_or_None(e.rut), pos, _dist(e.period, e.period_end))
        for (res, p), pattern in combos.items()
        for e in pattern for pos, period in enumerate(_range_backup[e.period, e.period_end])
    )
    return info

# def random_items(iterable, k=1):
#     result = [None] * k
#     for i, item in enumerate(iterable):
#         if i < k:
#             result[i] = item
#         else:
#             j = int(random() * (i+1))
#             if j < k:
#                 result[j] = item
#     shuffle(result)
#     return result

if __name__ == '__main__':
    import package.params as params
    import data.template_data as td
    import data.simulation as sim
    import data.test_data as test_d

    def temp():
        from importlib import reload
        reload(test_d)

    path = params.PATHS['data'] + 'template/201903120540/template_in.xlsx'
    # three options to import data: dassault template, toy-dataset, simulator.
    data_in = td.import_input_template(path)
    data_in = test_d.dataset3()
    # data_in = sim.create_dataset(params.OPTIONS)
    # Interesting. It works but only with a small maintenance window.

    instance = inst.Instance(data_in)
    res = instance.get_resources().keys_l()[0]
    info = get_graph_of_resource(instance, res)

    # Example creating paths between nodes:

    # node1 = rn.choice([n for n in nodes_ady.keys() if n.period=='2017-12'])
    # node1 comes from the status of the aircraft in the beginning of the window:

    # last_prev_assign = dict(period='2017-12', period_end='2017-12', assignment=None, type=0, rut=sd.SuperDict(),
    #                         ret=sd.SuperDict())

    # node_gr = options[0][1]

    # len(nodes_ady)

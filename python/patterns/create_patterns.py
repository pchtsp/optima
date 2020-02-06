import package.instance as inst
import pytups.superdict as sd
import numpy.random as rn
import patterns.node as nd

# installing graph-tool and adding it to venv:
# https://git.skewed.de/count0/graph-tool/wikis/installation-instructions
# https://jolo.xyz/blog/2018/12/07/installing-graph-tool-with-virtualenv


def walk_over_nodes(node: nd.Node, get_nodes_only=False):
    remaining_nodes = [(node, [])]
    # we store the neighbors of visited nodes, not to recalculate them
    cache_neighbors = {}
    i = 0
    final_paths = []
    # this code gets all combinations
    last_node = nd.get_sink_node(node.instance, node.resource)
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
            cache_neighbors[node] = neighbors = node.get_adjacency_list()
            if not len(neighbors):
                # no neighbors means I need to go to the last_node
                cache_neighbors[node] = neighbors = [last_node]
        elif get_nodes_only:
            # we have the cache AND we don't want paths, we leave
            continue
        remaining_nodes += [(n, path) for n in neighbors]
        print("iteration: {}, remaining: {}, stored: {}".format(i, len(remaining_nodes), len(cache_neighbors)))
    if get_nodes_only:
        return cache_neighbors
    return final_paths


def get_create_node(refs, g, n):
    v = refs.get(n)
    if v is not None:
        return v
    v = g.add_vertex()
    g.vp.period[v] = n.period
    g.vp.assignment[v] = n.assignment
    refs[n] = v
    return v


def adjacency_to_graph(nodes_ady):
    try:
        import graph_tool.all as gr
    except:
        print('graph-tool is not available')
        return None

    g = gr.Graph()
    g.vp.period = g.new_vp('string')
    g.vp.assignment = g.new_vp('string')
    g.ep.assignment = g.new_ep('string')
    refs = {}
    for n, n2_list in nodes_ady.items():
        v1 = get_create_node(refs, g, n)
        for n2 in n2_list:
            v2 = get_create_node(refs, g, n2)
            e = g.add_edge(v1, v2)
            if n2.assignment is None:
                g.ep.assignment[e] = ""
            else:
                g.ep.assignment[e] = n2.assignment

    return g, refs


def get_all_paths(g, refs):
    try:
        import graph_tool.all as gr
    except:
        print('graph-tool is not available')
        return None

    g_source = refs[source]
    g_sink = refs[sink]

    return [p for p in gr.all_paths(g, source=g_source, target=g_sink)]


def draw_graph(g):
    try:
        import graph_tool.all as gr
    except:
        print('graph-tool is not available')
        return None

    y_ranges = dict(M=lambda: 0,
                    VG=lambda: rn.uniform(10, 15) * (1 - 2 * (rn.random() > 0.5)),
                    VI=lambda: rn.uniform(5, 10) * (1 - 2 * (rn.random() > 0.5)),
                    VS=lambda: rn.uniform(0, 5) * (1 - 2 * (rn.random() > 0.5))
                    )
    extra_dist = {g.vp.assignment[v]: lambda: rn.uniform(-15, 15) for v in g.vertices()}
    y_ranges = {**extra_dist, **y_ranges}
    colors = \
        {'VG': '#4cb33d',
         'VI': '#00c8c3',
         'VS': '#31c9ff',
         'M': '#878787',
         'VG+VI': '#EFCC00',
         'VG+VS': '#EFCC00',
         'VG+VI+VS': '#EFCC00',
         'VI+VS': '#EFCC00'}
    extra_colors = {g.vp.assignment[v]: rn.choice(['#31c9ff', '#00c8c3', '#4cb33d']) for v in g.vertices()}
    colors = {**extra_colors, **colors}
    pos = g.new_vp('vector<float>')
    size = g.new_vp('double')
    shape = g.new_vp('string')
    color = g.new_vp('string')
    first = instance.get_param('start')

    for v in g.vertices():
        x = instance.get_dist_periods(first, g.vp.period[v])
        assignment = g.vp.assignment[v]
        y = y_ranges.get(assignment, lambda: 0)()
        pos[v] = (x, y)
        size[v] = 2
        shape[v] = 'circle'
        color[v] = colors.get(assignment, 'red')

    gr.graph_draw(g, pos=pos, vertex_text=g.vp.period,
                  edge_text=g.ep.assignment, vertex_shape=shape, vertex_fill_color=color)



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
    data_in = sim.create_dataset(params.OPTIONS)
    # Interesting. It works but only with a small maintenance window.

    instance = inst.Instance(data_in)
    res = instance.get_resources().keys_l()[0]
    source = nd.get_source_node(instance, res)
    sink = nd.get_sink_node(instance, res)
    # final_paths = walk_over_nodes(source)
    nodes_ady = walk_over_nodes(source, get_nodes_only=True)
    # stop
    g, refs = adjacency_to_graph(nodes_ady)

    import graph_tool.all as gr
    node1 = rn.choice([n for n in nodes_ady.keys() if n.assignment=='M'])
    node2 = rn.choice([n for n in nodes_ady.keys() if n.period=='2021-01'])
    # nd.Node(instance=instance, resource=res, period='2020-01', )
    # refs[node1]
    [p for p in gr.all_paths(g, source=refs[node1], target=refs[node2])]

    # draw_graph(g)
    final_paths = get_all_paths(g, refs)

    print(len(final_paths))
    # len(nodes_ady)

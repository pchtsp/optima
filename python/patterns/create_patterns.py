import package.instance as inst
import pytups.superdict as sd
import pytups.tuplist as tl
import numpy.random as rn
import patterns.node as nd

# installing graph-tool and adding it to venv:
# https://git.skewed.de/count0/graph-tool/wikis/installation-instructions
# https://jolo.xyz/blog/2018/12/07/installing-graph-tool-with-virtualenv


def walk_over_nodes(node: nd.Node, get_nodes_only=False):
    remaining_nodes = [(node, [])]
    # we store the neighbors of visited nodes, not to recalculate them
    cache_neighbors = sd.SuperDict()
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
            cache_neighbors[node] = neighbors = node.get_adjacency_list(only_next_period=True)
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
    refs[n] = int(v)
    return v


def adjacency_to_graph(nodes_ady):
    try:
        import graph_tool.all as gr
    except:
        print('graph-tool is not available')
        return None, None

    g = gr.Graph()
    g.vp.period = g.new_vp('string')
    g.vp.assignment = g.new_vp('string')
    g.ep.assignment = g.new_ep('string')
    refs = sd.SuperDict()
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


def get_all_patterns(graph, refs, refs_inv, instance, resource):
    try:
        import graph_tool.all as gr
    except:
        print('graph-tool is not available')
        return None

    source = nd.get_source_node(instance, resource)
    sink = nd.get_sink_node(instance, resource)

    return nodes_to_patterns(graph, refs, refs_inv, source, sink)


def draw_graph(instance, g, refs_inv=None):
    try:
        import graph_tool.all as gr
    except:
        print('graph-tool is not available')
        return None

    y_ranges = dict(VG=lambda: rn.uniform(10, 15) * (1 - 2 * (rn.random() > 0.5)),
                    VI=lambda: rn.uniform(5, 10) * (1 - 2 * (rn.random() > 0.5)),
                    VS=lambda: rn.uniform(0, 5) * (1 - 2 * (rn.random() > 0.5))
                    )
    max_rut = instance.get_maintenances('max_used_time')['M']

    def get_y_mission(v):
        if not refs_inv[v].rut:
            return - 15
        return (refs_inv[v].rut['M'] / max_rut - 0.5) * 20

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
    # empty assignments we put in white:
    extra_colors['']  = '#ffffff'
    colors = {**extra_colors, **colors}
    pos = g.new_vp('vector<float>')
    size = g.new_vp('double')
    shape = g.new_vp('string')
    color = g.new_vp('string')
    first = instance.get_param('start')

    for v in g.vertices():
        x = instance.get_dist_periods(first, g.vp.period[v])
        assignment = g.vp.assignment[v]
        if assignment in y_ranges:
            y = y_ranges.get(assignment, lambda: 0)()
        else:
            y = get_y_mission(v)
        pos[v] = (x, y)
        size[v] = 2
        shape[v] = 'circle'
        color[v] = colors.get(assignment, 'red')

    gr.graph_draw(g, pos=pos, vertex_text=g.vp.period,
                  edge_text=g.ep.assignment, vertex_shape=shape, vertex_fill_color=color)

def nodes_to_patterns(graph, refs, refs_inv, node1, node2, cutoff=1000, **kwargs):
    import graph_tool.all as gr
    # TODO: there is something going on with default consumption
    # TODO: there is something going on with initial states
    all_paths = gr.all_paths(graph, source=refs[node1], target=refs[node2], cutoff=cutoff)
    return tl.TupList(all_paths).vapply(lambda v: tl.TupList(v).vapply(lambda vv: refs_inv[vv]))
    refs.keys_tl().\
        vfilter(lambda v: v.period=='2019-01' and v.period_end=='2019-01' and v.assignment=='').\
        vapply(lambda v: (v.rut, v.ret))
    node1.rut, node1.ret

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

    # TODO: fix assignment periods should only leave the possibility of the assignment
    # maybe changing the origin.

    nodes_ady_2 = nodes_ady.kvapply(lambda k, v: v + nodes_artificial.get(k, []))

    # Then, when exploiting this, we will filter nodes with low enough ret and rut.
    # compared to the cycle limit on the next maintenance.

    # We create a graph-tool version of the graph
    # and links between the original nodes and the graph tool ones
    g, refs = adjacency_to_graph(nodes_ady_2)
    return sd.SuperDict(graph=g, refs=refs, refs_inv=refs.reverse(), source=source, sink=sink)


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
    draw_graph(instance, info['graph'])

    # Example creating paths between nodes:

    # node1 = rn.choice([n for n in nodes_ady.keys() if n.period=='2017-12'])
    # node1 comes from the status of the aircraft in the beginning of the window:

    # last_prev_assign = dict(period='2017-12', period_end='2017-12', assignment=None, type=0, rut=sd.SuperDict(),
    #                         ret=sd.SuperDict())

    # node_gr = options[0][1]

    # len(nodes_ady)

import package.instance as inst
import pytups.superdict as sd
import ujson as json


# installing graph-tool and adding it to venv:
# https://git.skewed.de/count0/graph-tool/wikis/installation-instructions
# https://jolo.xyz/blog/2018/12/07/installing-graph-tool-with-virtualenv


class Node(object):
    """
    corresponds to a combination of: a period t, a maintenance m, and the remaining rets and ruts for all maintenances
    """

    def __init__(self, instance, resource, period, ret, rut, maint):
        """

        :param inst.Instance instance:
        :param period:
        :param sd.SuperDict ret:
        :param sd.SuperDict rut:
        :param maint:
        """
        self.instance = instance
        self.ret = ret
        self.rut = rut
        self.maint = maint
        self.period = period
        self.resource = resource
        self.maint_data = instance.data['maintenances']
        self.hash = hash(json.dumps(self.get_data(), sort_keys=True))
        return

    def __repr__(self):
        return repr('{} => {}'.format(self.period, self.maint))

    def get_maints_rets(self):
        return self.ret.keys()

    def get_maints_ruts(self):
        return self.rut.keys()

    # @profile
    def get_adjacency_list(self):
        """
        gets all nodes that are reachable from this node.
        :return: a list of nodes.
        """
        last = self.instance.get_param('end')
        if self.period >= last:
            return []

        data_m = self.maint_data
        maints_ret = self.get_maints_rets()
        opts_ret = sd.SuperDict()
        for m in maints_ret:
            start = max(self.ret[m] - data_m[m]['elapsed_time_size'], 0)
            end = self.ret[m]
            _range = range(int(start), int(end))
            if _range:
                opts_ret[m] = _range
                opts_ret[m] = set(opts_ret[m])

        acc_cons = 0
        maints_rut = self.get_maints_ruts()
        opts_rut = sd.SuperDict.from_dict({m: set() for m in maints_rut})
        maints_rut = set(maints_rut)
        dif_m = {m: info['used_time_size'] for m, info in data_m.items()}
        max_period = None
        # duration_past_maint = data_m.get_m(self.maint, 'duration_periods')
        # if duration_past_maint is None:
        # duration_past_maint = 0
        for num, period in enumerate(self.iter_until_period(last)):
            acc_cons += self.get_consume(period)
            for m in maints_rut:
                rut_future = self.rut[m] - acc_cons
                if rut_future < 0:
                    # remove_maint.append(m)
                    # we have reached the first limit on maintenances, we should stop.
                    max_period = num
                    if num == 0:
                        # if its the first period and we're already too late,
                        # we still try.
                        opts_rut[m].add(num)
                    break
                elif rut_future < dif_m[m]:
                    # we still can do the maintenance in this period
                    # but we don't want the same one at 0; duh.
                    if m != self.maint and num != 0:
                        opts_rut[m].add(num)
            # it means we reached the upper limit of at least
            # one of the maintenances
            if max_period is not None:
                break

        intersect = opts_rut.keys() & opts_ret.keys()
        int_dict = \
            opts_ret. \
                filter(intersect). \
                apply(lambda k, v: opts_rut[k] & v). \
                apply(lambda k, v: v if len(v) else opts_ret[k])

        opts_tot = opts_rut.clean(func=lambda v: v)
        opts_tot.update(opts_ret)
        opts_tot.update(int_dict)
        # print(opts_tot)
        # opts_tot = opts_rut._update(opts_ret)._update(int_dict).clean(func=lambda v: v)
        max_opts = min(opts_tot.vapply(lambda l: max(l)).values())
        opts_tot = opts_tot.vapply(lambda v: [vv for vv in v if vv <= max_opts]).vapply(set)
        candidates = [self.create_adjacent(maint, opt) for maint, opts in opts_tot.items() for opt in opts]
        return [c for c in candidates if c.maint is not None]

    def calculate_rut(self, maint, period):
        """
        If next m=maint and is done in t=period, how does rut[m][t] change?
        :param str maint:
        :param str period:
        :return:
        """
        maints_ruts = self.get_maints_ruts()
        m_affects = set()
        if maint is not None:
            m_affects = self.maint_data[maint]['affects'] & maints_ruts
        m_not_affect = maints_ruts - m_affects
        # rut = \
        #     self.rut.\
        #     filter(m_not_affect, check=False).\
        #     clean(func=lambda m: m is not None)
        total_consumption = sum(self.get_consume(p) for p in self.iter_until_period(period))
        # duration_past_maint = self.maint_data.get_m(self.maint, 'duration_periods')
        # if duration_past_maint is None:
        #     duration_past_maint = 0
        # there is an assumption that the next maintenance
        # will always be duration_past_maint periods away.

        # for not affected maintenances, we reduce:
        # check if past maintenance had a duration.
        # and discount the consumption for the first months
        rut = sd.SuperDict()
        for m in m_not_affect:
            rut[m] = self.rut[m] - total_consumption
        # for pos, p in enumerate(self.iter_until_period(period)):
        #     if pos >= duration_past_maint:
        #         rut = rut.vapply(lambda v: v - consumption)

        # for affected maintenances, we set at max:
        for m in m_affects:
            rut[m] = self.maint_data[m]['max_used_time']
            duration = self.maint_data.get_m(maint, 'duration_periods')
            fake_consumption = sum(self.get_consume(p) for p in range(duration))
            rut[m] += fake_consumption

        return rut

    def calculate_ret(self, maint, period):
        """
        If next m=maint and is done in t=period, how does ret change?
        :param str maint:
        :param str period:
        :return:
        """
        # for each maint, except maint, I reduce the ret
        # for maint, I put at max
        maints_rets = self.get_maints_rets()
        m_affects = set()
        maint_data = self.maint_data
        if maint is not None:
            m_affects = set(maint_data[maint]['affects']) & maints_rets
        m_not_affect = maints_rets - m_affects
        time = self.dif_period(period)
        ret = sd.SuperDict()
        for m in m_not_affect:
            ret[m] = self.ret[m] - time
        # ret = self.ret.\
        #     filter(m_not_affect, check=False).\
        #     clean(func=lambda m: m is not None).\
        #     vapply(lambda v: v - time)

        # for affected maintenances, we set at max:
        for m in m_affects:
            ret[m] = maint_data[m]['max_elapsed_time'] + \
                     maint_data[maint]['duration_periods']
        return ret

    def next_period(self, num=1):
        return self.instance.shift_period(self.period, num)

    def dif_period(self, period):
        return self.instance.get_dist_periods(self.period, period)

    def get_consume(self, period):
        return self.instance.get_default_consumption(self.resource, period)

    def iter_until_period(self, period):
        current = self.period
        while current < period:
            yield current
            current = self.instance.get_next_period(current)

    def create_adjacent(self, maint, num_periods):
        period = self.next_period(num_periods)
        last = self.instance.get_param('end')
        if period > last:
            # we link with the last node and we finish
            period = last
            maint = None
        ret = self.calculate_ret(maint, period)
        rut = self.calculate_rut(maint, period)
        defaults = dict(instance=self.instance, resource=self.resource)
        return Node(period=period, maint=maint, rut=rut, ret=ret, **defaults)

    def get_data(self):
        return sd.SuperDict.from_dict({'ret': self.ret, 'rut': self.rut,
                                       'maint': self.maint, 'period': self.period})

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        data1 = self.get_data().to_dictup().to_tuplist().to_set()
        data2 = other.get_data().to_dictup().to_tuplist().to_set()
        dif = data1 ^ data2
        return len(dif) == 0


def walk_over_nodes(node, get_nodes_only=False):
    remaining_nodes = [(node, [])]
    # we store the neighbors of visited nodes, not to recalculate them
    cache_neighbors = {}
    i = 0
    final_paths = []
    # this code gets all combinations
    last_node = get_sink_node(node.instance, node.resource)
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


def get_source_node(instance, resource):
    instance.data = sd.SuperDict.from_dict(instance.data)
    start = instance.get_param('start')
    period = instance.get_prev_period(start)
    resources = instance.get_resources()

    maints = instance.get_maintenances()
    rut = maints.kapply(lambda m: resources[resource]['initial'][m]['used']).clean(func=lambda v: v)
    ret = maints.kapply(lambda m: resources[resource]['initial'][m]['elapsed']).clean(func=lambda v: v)
    return Node(instance, resource, period, ret, rut, None)


def get_sink_node(instance, resource):
    last = instance.get_param('end')
    last_next = instance.get_next_period(last)
    defaults = dict(instance=instance, resource=resource)
    return Node(period=last_next, maint=None, rut=sd.SuperDict(), ret=sd.SuperDict(), **defaults)


def get_create_node(refs, g, n):
    v = refs.get(n)
    if v is None:
        v = g.add_vertex()
        g.vp.period[v] = n.period
        g.vp.maint[v] = n.maint
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
    g.vp.maint = g.new_vp('string')
    g.ep.maint = g.new_ep('string')
    refs = {}
    for n, n2_list in nodes_ady.items():
        v1 = get_create_node(refs, g, n)
        for n2 in n2_list:
            v2 = get_create_node(refs, g, n2)
            e = g.add_edge(v1, v2)
            if n2.maint is None:
                g.ep.maint[e] = ""
            else:
                g.ep.maint[e] = n2.maint

    return g, refs


if __name__ == '__main__':
    import package.params as params
    import data.template_data as td

    path = params.PATHS['data'] + 'template/201903120540/template_in.xlsx'
    data_in = td.import_input_template(path)

    import data.test_data as test_d


    def temp():
        from importlib import reload
        reload(test_d)


    data_in = test_d.dataset1()

    instance = inst.Instance(data_in)
    res = instance.get_resources().keys_l()[0]
    source = get_source_node(instance, res)
    sink = get_sink_node(instance, res)
    # final_paths = walk_over_nodes(source)
    nodes_ady = walk_over_nodes(source, get_nodes_only=True)
    g, refs = adjacency_to_graph(nodes_ady)
    g_source = refs[source]
    g_sink = refs[sink]

    import graph_tool.all as gr
    import numpy.random as rn

    y_ranges = dict(M=lambda: 0,
                    VG=lambda: rn.uniform(10, 15) * (1 - 2 * (rn.random() > 0.5)),
                    VI=lambda: rn.uniform(5, 10) * (1 - 2 * (rn.random() > 0.5)),
                    VS=lambda: rn.uniform(0, 5) * (1 - 2 * (rn.random() > 0.5))
                    )
    # colors = dict(VG='green')
    colors = \
        {'VG': '#4cb33d',
         'VI': '#00c8c3',
         'VS': '#31c9ff',
         'M': '#878787',
         'VG+VI': '#EFCC00',
         'VG+VS': '#EFCC00',
         'VG+VI+VS': '#EFCC00',
         'VI+VS': '#EFCC00'}

    pos = g.new_vp('vector<float>')
    size = g.new_vp('double')
    shape = g.new_vp('string')
    color = g.new_vp('string')
    first = instance.get_param('start')

    for v in g.vertices():
        x = instance.get_dist_periods(first, g.vp.period[v])
        maint = g.vp.maint[v]
        y = y_ranges.get(maint, lambda: 0)()
        # if g.vp.maint[v] == 'M':
        #     y = 0
        pos[v] = (x, y)
        size[v] = 2
        shape[v] = 'circle'
        color[v] = colors.get(maint, 'red')

    gr.graph_draw(g, pos=pos, vertex_text=g.vp.period,
                  edge_text=g.ep.maint, vertex_shape=shape, vertex_fill_color=color)

    final_paths = [p for p in gr.all_paths(g, source=g_source, target=g_sink)]
    len(final_paths)
    len(nodes_ady)
    # print(len(graph))
    # print(len(end_nodes))
    # print(len(final_paths))
    # unique_data = [list(x) for x in set(tuple(x) for x in final_paths)]
    # print(len(unique_data))
    # print(final_paths[0])
    # print(final_paths[-1])
    # print(final_paths)

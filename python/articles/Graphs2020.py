import patterns.create_patterns as cp
import data.test_data as test_d
import package.instance as inst
import network2tikz as nt

def temp():
    from importlib import reload
    reload(test_d)
    reload(cp)

def draw():
    data_in = test_d.dataset3_no_default_5_periods()
    instance = inst.Instance(data_in)
    info = cp.get_graph_of_resource(instance, '1')
    path = '/home/pchtsp/Documents/projects/Graph2020/graph_example.tex'
    first, last = instance.get_start_end()
    def edge_label(tail):
        equiv = {'M': '-1', '': '0'}
        a = tail.assignment
        if a is None:
            return ""
        return equiv.get(a, a)

    def node_label(node):
        pos = instance.get_dist_periods(first, node.period)
        if node.rut is None:
            rut = ret = ''
        else:
            rut = node.rut['M']
            ret = node.ret['M']
        return '{}/{}/{}'.format(pos, rut, ret)
    # node_label = lambda node:
    cp.draw_graph(instance, info['graph'], info['refs_inv'],
                  edge_label=edge_label, node_label=node_label, tikz=True, filename=path)

def draw2():
    nt.plot()
    pass

def table():
    data_in = test_d.dataset3()
    instance = inst.Instance(data_in)
    info = cp.get_graph_of_resource(instance, '0')
    node = [k for k, v in info['refs'].items() if
            k.period == '2018-02' and k.period_end == '2018-06' and k.rut is not None].pop()
    info['node1'] = node
    info['node2'] = info['sink']

    paths = cp.nodes_to_patterns(**info, add_empty=False)
    combos = cp.get_patterns_into_dictup({'1': paths})
    first, last = instance.get_first_last_period()

    assignments = cp.get_assignments_from_patterns(instance, combos, 'M')
    import pandas as pd

    equiv = assignments.take(2).unique2().sorted().kvapply(lambda k, v: (k, v)).to_dict(is_list=False)
    info_pd = \
        pd.DataFrame. \
            from_records(assignments.to_list(), columns=['res', 'pat', 'period', 'assign'] + list(range(4))). \
            filter(['pat', 'period', 'assign']).query('period>="{}" & period<="{}"'.format(first, last))
    info_pd.period = info_pd.period.map(equiv)
    latex_str = info_pd.set_index(['pat', 'period']).unstack('period')['assign']. \
        to_latex(longtable=True, index_names=False, header=False)
    print(latex_str)


def compare_num_paths():
    def num_paths(info, max_paths=10000):
        info['node1'] = info['source']
        info['node2'] = info['sink']
        paths = cp.nodes_to_patterns(**info, add_empty=False, max_paths=max_paths)
        return len(paths)

    data_in = test_d.dataset3()
    instance = inst.Instance(data_in)
    info = cp.get_graph_of_resource(instance, '0')
    a = num_paths(info)
    # 61
    data_in = test_d.dataset4()
    instance = inst.Instance(data_in)
    info = cp.get_graph_of_resource(instance, '0')
    b = num_paths(info, max_paths=1000000)
    # 123011
    data_in = test_d.dataset4()
    instance = inst.Instance(data_in)
    info = cp.get_graph_of_resource(instance, '1')
    b = num_paths(info, max_paths=1000000)
    # 815747
    data_in = test_d.dataset5().get_instance()
    instance = inst.Instance(data_in)
    info = cp.get_graph_of_resource(instance, '0')
    d = num_paths(info, max_paths=1000000)
    # 4208
    data_in = test_d.dataset5().get_instance()
    instance = inst.Instance(data_in)
    info = cp.get_graph_of_resource(instance, '1')
    d = num_paths(info, max_paths=1000000)
    # 1862
    data_in = test_d.dataset6().get_instance()
    instance = inst.Instance(data_in)
    infos = instance.get_resources().\
        kapply(lambda k: cp.get_graph_of_resource(instance, k)).\
        vapply(num_paths, max_paths=1000000)
    # {'0': 59165,
    #  '1': 28050,
    #  '10': 34251,
    #  '11': 693412,
    #  '12': 412632,
    #  '13': 589585,
    #  '14': 1000000,
    #  '15': 1000000,
    #  '16': 204885,
    #  '17': 1119,
    #  '18': 1155,
    #  '19': 4573,
    #  '2': 134184,
    #  '20': 3685,
    #  '21': 1222,
    #  '22': 8635,
    #  '23': 1365,
    #  '24': 5751,
    #  '25': 27697,
    #  '26': 563903,
    #  '27': 166245,
    #  '28': 239134,
    #  '29': 13100,
    #  '3': 2276,
    #  '30': 11665,
    #  '31': 374011,
    #  '32': 268160,
    #  '33': 149156,
    #  '34': 1020,
    #  '35': 9889,
    #  '36': 53490,
    #  '37': 403259,
    #  '38': 7479,
    #  '39': 363896,
    #  '4': 39222,
    #  '40': 7488,
    #  '41': 147110,
    #  '42': 1000000,
    #  '43': 8304,
    #  '44': 160270,
    #  '45': 315480,
    #  '46': 232042,
    #  '47': 66640,
    #  '48': 1464,
    #  '49': 6465,
    #  '5': 157769,
    #  '50': 28407,
    #  '51': 403259,
    #  '52': 1155,
    #  '53': 1000000,
    #  '54': 4200,
    #  '55': 1155,
    #  '56': 1148,
    #  '57': 520651,
    #  '58': 1461,
    #  '59': 2069,
    #  '6': 8920,
    #  '7': 9240,
    #  '8': 17540,
    #  '9': 6204}

if __name__ == '__main__':
    draw()
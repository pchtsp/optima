import pandas as pd

import patterns.graphs as gg

import data.test_data as test_d
import package.instance as inst
import package.params as params

import solvers.heuristic_graph as hg

import reports.compare_experiments as comp

abs_path = params.PATHS['root'] +  'Graph2020/'

def temp():
    from importlib import reload
    reload(test_d)
    reload(gg)

def draw():
    data_in = test_d.dataset3_no_default_5_periods()
    instance = inst.Instance(data_in)
    graph = gg.GraphTool(instance, '1')
    path = abs_path + 'graph_example.tex'
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
    graph.draw(edge_label=edge_label, node_label=node_label, tikz=True, filename=path)

def draw2():
    import network2tikz as nt
    nt.plot()
    pass

def table():
    data_in = test_d.dataset3_no_default_5_periods()
    instance = inst.Instance(data_in)
    info = gg.graph_factory(instance, '1', {})
    path = abs_path + 'table_paths.tex'
    # node = [k for k, v in info['refs'].items() if
    #         k.period == '2018-02' and k.period_end == '2018-06' and k.rut is not None].pop()
    paths = info.nodes_to_patterns(info.source, info.sink, add_empty=False)
    combos = hg.get_patterns_into_dictup({'1': paths})
    first, last = instance.get_first_last_period()
    assignments = hg.get_assignments_from_patterns(instance, combos, 'M')

    equiv = assignments.take(2).unique2().sorted().kvapply(lambda k, v: (k, v)).to_dict(is_list=False)
    equiv2 = {'M': -1, '': 0, '1': 1}
    info_pd = \
        pd.DataFrame. \
            from_records(assignments.to_list(), columns=['res', 'pat', 'period', 'task'] + list(range(4))). \
            filter(['pat', 'period', 'task']).query('period>="{}" & period<="{}"'.format(first, last))
    info_pd.period = info_pd.period.map(equiv)
    info_pd.task = info_pd.task.map(equiv2)
    latex_str = info_pd.set_index(['pat', 'period']).unstack('period')['task']. \
        to_latex(longtable=False, index_names=False, index=False)
    with open(path, 'w') as f:
        f.write(latex_str)


def compare_num_paths():
    def num_paths(graph, max_paths=10000):
        paths = graph.nodes_to_patterns(graph.source, graph.sink, add_empty=False, max_paths=max_paths)
        return len(paths)

    data_in = test_d.dataset3()
    instance = inst.Instance(data_in)
    info = gg.graph_factory(instance, '0')
    a = num_paths(info)
    # 61
    data_in = test_d.dataset4()
    instance = inst.Instance(data_in)
    info = gg.graph_factory(instance, '0')
    b = num_paths(info, max_paths=1000000)
    # 123011
    data_in = test_d.dataset4()
    instance = inst.Instance(data_in)
    info = gg.graph_factory(instance, '1')
    b = num_paths(info, max_paths=1000000)
    # 815747
    data_in = test_d.dataset5().get_instance()
    instance = inst.Instance(data_in)
    info = gg.graph_factory(instance, '0')
    d = num_paths(info, max_paths=1000000)
    # 4208
    data_in = test_d.dataset5().get_instance()
    instance = inst.Instance(data_in)
    info = gg.graph_factory(instance, '1')
    d = num_paths(info, max_paths=1000000)
    # 1862
    data_in = test_d.dataset6().get_instance()
    instance = inst.Instance(data_in)
    infos = instance.get_resources().\
        kapply(lambda k: gg.graph_factory(instance, k)).\
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


def compare_experiments(exp_list, get_progress=False, **kwargs):

    def get_solstats(batch):
        objFunc = batch.\
            get_cases().\
            clean(func=lambda v: v).\
            vapply(hg.GraphOriented.get_objective_function).\
            vapply(lambda v: dict(objective=v))
        return batch.format_df(objFunc).drop('name', axis=1)

    result = comp.get_df_comparison(exp_list, get_log=True, solstats_func=get_solstats, zip=True,
                                    get_progress=get_progress, **kwargs)
    if not get_progress:
        return result

    def expand_df(index):
        row = result.iloc[[index]]
        progress = row.iloc[0].progress.reset_index()[['Time', 'BestInteger']]
        rows = row.loc[row.index.repeat(len(progress))]
        rows = rows.drop('progress', axis=1).reset_index()
        return pd.concat([rows, progress], axis=1)

    tables = [expand_df(i) for i in range(len(result)) if len(result.iloc[i].progress)]
    return pd.concat(tables, axis=0, join='inner')


if __name__ == '__main__':
    # table()
    # t2 = compare_experiments(exp_list=['serv_cluster1_20200701', 'serv_cluster1_20200617_2'],
    #                          solver=dict(serv_cluster1_20200623='HEUR'), get_progress=True,
    #                          scenarios=["numparalleltasks_3", "numparalleltasks_4", "numparalleltasks_5"])
    t2 = compare_experiments(exp_list=['serv_cluster1_20200701'], get_progress=False,
                             solver=dict(serv_cluster1_20200701='HEUR'))

    # t = compare_experiments(exp_list=['prise_srv3_20200603_2'], get_progress=True)
    # [t.iloc[x].progress.tail(10) for x in range(len(t))]
    # [t2.iloc[x].progress.tail(10) for x in range(len(t2))]

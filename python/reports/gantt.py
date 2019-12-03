import os
import plotly as pt
import plotly.figure_factory as ff
import pytups.tuplist as tl
import pytups.superdict as sd
import package.experiment as exp
import re
import arrow

def make_gantt_from_experiment(experiment=None, path='', name='solution.html'):
    if path and not os.path.isdir(path):
        raise ValueError('Path to export image does not exist.')
    if experiment is None:
        if not os.path.isdir(path):
            raise ValueError('Experiment cannot be loaded from path.')
        experiment = exp.Experiment.from_dir(path)
        if experiment is None:
            raise ValueError
    filename = os.path.join(path, name)

    start_end = experiment.get_state_periods() + experiment.get_task_periods()
    start_end = tl.TupList(start_end)

    start_end.sort()
    start_end_2 = start_end.to_dict(result_col=[2]).vapply(lambda x: '+'.join(x)).to_tuplist()
    # try:
    #     pos = start_end.filter(0).apply(lambda x: re.search(string=x, pattern='[0-9]+')[0]).apply(int)
    # except:
    #     pos = [1]*len(start_end)

    last_day = lambda month: arrow.get(month + "-01").shift(months=1).shift(days=-2).format("YYYY-MM-DD")
    transf = lambda item: dict(Task=item[0], Start=item[1]+'-01', Finish=last_day(item[2]), Resource=item[3])

    gantt_data = start_end_2.apply(transf)
    all_names = gantt_data.vapply(lambda v: v['Resource']).unique2()
    # tl.TupList([{**v, **{'pos': pos[k]}} for k, v in enumerate(gantt_data)])

    colors = \
        {'VG': '#4cb33d',
         'VI': '#00c8c3',
         'VS': '#31c9ff',
         'M': '#878787',
         'VG+VI': '#EFCC00',
         'VG+VS': '#EFCC00',
         'VG+VI+VS': '#EFCC00',
         'VI+VS': '#EFCC00'}

    colors = sd.SuperDict.from_dict(colors).\
        fill_with_default(all_names, default='#ff0000')
    # we try to sort according to standard naming.
    # gantt_data.apply(lambda x: re.search(x['Task'], r'\d'))
    try:
        gantt_data.sort(key=lambda x: int(x['Task'][2:]))
    except:
        pass

    # TODO: figure out how to autoadjust width to 100%

    options = dict(show_colorbar=True, group_tasks=True, showgrid_x=True, title="Maintenance planning",
                   bar_width=0.5, width=2000, height=1000)
    fig = ff.create_gantt(gantt_data, colors=colors, index_col='Resource', **options)
    for i in range(len(gantt_data)):
        task = gantt_data[i]['Resource']
        st = gantt_data[i]['Start']
        res = gantt_data[i]['Task']
        text = '{}<br>{}<br>{}'.format(res, st, task)
        fig["data"][i].update(text=text, hoverinfo="text")
    fig['layout'].update(autosize=True, margin=dict(l=150))
    pt.offline.plot(fig, filename=filename, show_link=False, config=dict(responsive=True))

if __name__ == '__main__':
    path = r'C:\Users\pchtsp\Documents\projects\optima_results\experiments\test_remake/'
    make_gantt_from_experiment(path=path)

import os
import plotly as pt
import plotly.figure_factory as ff
import pytups.tuplist as tl
import package.experiment as exp


def make_gantt_from_experiment(experiment=None, path='', name='solution.html'):
    if path and not os.path.isdir(path):
        raise ValueError('Path to export image does not exist.')
    if experiment is None:
        if not os.path.isdir(path):
            raise ValueError('Experiment cannot be loaded from path.')
        experiment = exp.Experiment.from_dir(path)
    filename = os.path.join(path, name)

    start_end = experiment.get_state_periods()
    start_end = tl.TupList(start_end)

    transf = lambda item: dict(Task=item[0], Start=item[1]+'-01', Finish=item[3]+'-30', Resource=item[2])

    df = start_end.apply(transf)

    colors = \
        {'VG': '#4cb33d',
         'VI': '#00c8c3',
         'VS': '#31c9ff',
         'M': '#878787',
         'VG+VI': '#EFCC00',
         'VG+VS': '#EFCC00',
         'VG+VI+VS': '#EFCC00',
         'VI+VS': '#EFCC00'}

    # we try to sort according to standard naming.
    try:
        df.sort(key=lambda x: int(x['Task'][2:]))
    except:
        pass

    # TODO: figure out how to autoadjust width to 100%

    options = dict(show_colorbar=True, group_tasks=True, showgrid_x=True, title="Maintenance planning",
                   bar_width=0.5, width=2000, height=1000)
    fig = ff.create_gantt(df, colors=colors, index_col='Resource', **options)
    for i in range(len(df)):
        text = df[i]['Resource']
        fig["data"][i].update(text=text, hoverinfo="text")
    fig['layout'].update(autosize=True, margin=dict(l=150))
    pt.offline.plot(fig, filename=filename, show_link=False, config=dict(responsive=True))

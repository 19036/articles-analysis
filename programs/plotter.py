import json
import os

import bokeh as bk
from bokeh.io import curdoc
from bokeh.plotting import figure, show
from bokeh.layouts import row, column, grid
from bokeh.palettes import HighContrast3, Turbo256, Category20, linear_palette
import numpy as np
from math import pi
import pandas as pd
from bokeh.palettes import Category20c
from bokeh.plotting import figure, show
from bokeh.transform import cumsum
from bokeh.palettes import HighContrast3
from bokeh.plotting import figure, show, save, output_file
from statistics import mean

curdoc().theme = 'dark_minimal'
calc_inertia = False
calc_abstracts = False
calc_kmeans = False
calc_topics = False
calc_norms = False

################# INERTIA PLOT #################
if calc_inertia:
    output_file(filename="plot_inertia.html", title="HTML1")
    inertia_paths = ['inertia_soball.txt', 'inertia_nsulvl1.txt']

    x = []
    s = []
    v = []

    for i in range(len(inertia_paths)):
        with open(inertia_paths[i], 'r') as f:
            s.append(f.read())
        v.append(json.loads(s[i]))
    for i in range(max(len(j) for j in v)):
        x.append(i + 1)

    p = []
    for i in range(len(inertia_paths)):
        p.append(figure(title=inertia_paths[i], x_axis_label='N', y_axis_label='Inertia', tools="hover", tooltips="y @y"))
        p[i].line(x, v[i], legend_label="Inertia", color="yellow", line_width=2)

    save(row(p))
################# INERTIA PLOT #################


################# ABSTRACT PLOT #################
if calc_abstracts:
    output_file(filename="plot_abstract.html", title="HTML2")

    nsu_all = 547229
    nsu_all_null = 69164
    nsu_lvl1 = 28075
    sob_all = 65000
    sob_all_notnull = 50387
    sob_lvl1 = 7200
    sob_lvl1_notnull = 4149

    levels = ['LVL1', 'LVL2']
    abstracts = ["ABS", "NOABS"]

    data = [{'levels' : levels,
            'ABS'   : [sob_lvl1_notnull, sob_all_notnull-sob_lvl1_notnull],
            'NOABS'   : [sob_lvl1-sob_lvl1_notnull, sob_all-sob_all_notnull-sob_lvl1+sob_lvl1_notnull]},
            {'levels' : levels,
            'ABS'   : [2, 1],
            'NOABS'   : [5, 3]}
            ]

    p = []
    for i in range(len(data)):
        p.append(figure(x_range=levels, height=600, title="Abstract containment", toolbar_location=None, tools="hover", tooltips="$name @levels: @$name"))
        p[i].vbar_stack(abstracts, x='levels', width=0.9, alpha=0.5, color=["cyan", "red"], source=data[i], legend_label=abstracts)
        p[i].y_range.start = 0
        p[i].x_range.range_padding = 0.1
        p[i].xgrid.grid_line_color = None
        p[i].axis.minor_tick_line_color = None
        p[i].outline_line_color = None
        p[i].legend.location = "top_left"
        p[i].legend.orientation = "horizontal"

    save(row(p))
################# ABSTRACT PLOT #################

################# KMEANS PLOT #################
if calc_kmeans:
    output_file(filename="plot_kmeans.html", title="HTML3")
    from bokeh.models import ColumnDataSource

    files = os.listdir('./input_kmeans')

    def first_3chars(x):
        return(x[-11:])

    files = sorted(files, key = first_3chars)
    kmeans_paths = []
    for k in range(len(files)):
        kmeans_paths.append('./input_kmeans/'+files[k])

    s = []
    v = []
    for i in range(len(kmeans_paths)):
        with open(kmeans_paths[i], 'r') as f:
            s.append(f.read())
        v.append(json.loads(s[i]))

    p = []
    tot_av = []
    for i in range(len(v)):
        p.append(figure(title=kmeans_paths[i], x_axis_label='Cluster', y_axis_label='Value', toolbar_location=None, tools="hover", tooltips="y: @y1 , @y"))
        s = 0
        tot_av.append(0)
        for j in range(len(v[i])):
            x = []
            av = 0
            for k in range(len(v[i][j])):
                av += v[i][j][k]
                x.append(s + k + 1)
            tot_av[i]+=av
            av /= len(v[i][j])
            p[i].varea(x, y1=v[i][j], y2=0, fill_alpha=0.6, fill_color=Turbo256[20 + j * (256-20) // len(v[i])])
            s = x[len(x)-1]

            # Average
            p[i].line(x, av, legend_label="Average", color="yellow", line_width=1)
            # Average
        tot_av[i]/=s
        p[i].line(list(range(1, s)), tot_av[i], legend_label="Total Average", color="red", line_width=2)

    p.append(figure(title="TOTAL AVERAGES", x_axis_label='N', y_axis_label='Total Average', tools="hover", tooltips="y @y"))
    p[len(p)-1].line(list(range(2,len(v)+2)), tot_av, legend_label="Average", color="yellow", line_width=3)

    save(grid(p, ncols=3))
################# KMEANS PLOT #################


################# TOPICS PLOT #################
if calc_topics:
    from math import pi

    import pandas as pd

    from bokeh.plotting import figure, show
    from bokeh.transform import cumsum

    output_file(filename="plot_topics.html", title="HTML4")

    files = os.listdir('./input_topics')
    topics_paths = []
    for k in range(len(files)):
        topics_paths.append('./input_topics/' + files[k])

    s = []
    v = []
    for i in range(len(topics_paths)):
        with open(topics_paths[i], 'r') as f:
            s.append(f.read())
        v.append(json.loads(s[i]))

    p = []
    for i in range(len(v)):
        key = "Domains"
        x = dict(v[i][key])

        tot_sum = sum(v[i][key].values())
        x.update({"Other": 0})
        for t in v[i][key]:
            if v[i][key][t]< 0.002 * tot_sum:
                x.pop(t)
                x["Other"]+=v[i][key][t]


        data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'country'})
        data['angle'] = data['value'] / data['value'].sum() * 2 * pi
        data['color'] = linear_palette(Turbo256, len(x))

        p.append(figure(height=450, title=v[i]["Title"], toolbar_location=None, tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0)))
        p[i].title.text_font_size = '14pt'
        p[i].wedge(x=0, y=1, radius=0.4,
                start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
                line_color="white", fill_color='color', legend_field='country', source=data)

        p[i].axis.axis_label = None
        p[i].axis.visible = False
        p[i].grid.grid_line_color = None

    show(grid(p, ncols=3))
################# TOPICS PLOT #################

################# NORMS PLOT #################
if calc_norms:
    output_file(filename="plot_norms.html", title="HTML5")
    from bokeh.models import ColumnDataSource

    files = os.listdir('./input_norms')

    paths = []
    for k in range(len(files)):
        paths.append('./input_norms/'+files[k])

    s = []
    v = []
    for i in range(len(paths)):
        with open(paths[i], 'r') as f:
            s.append(f.read())
        v.append(json.loads(s[i]))

    p = []
    for i in range(len(v)):
        p.append(figure(title=paths[i], x_axis_label='Cluster', y_axis_label='Value', toolbar_location=None, tools="hover", tooltips="y: @y1 , @y"))
        s = 0
        av = 0
        for j in range(len(v[i])):
            av += v[i][j]
            av /= len(v[i])
            p[i].varea(list(range(1, len(v[i])+1)), y1=v[i], y2=0, fill_alpha=0.6, fill_color='red')

            # Average
            # p[i].line(list(range(1, len(v[i])+1)), av, legend_label="Average", color="yellow", line_width=1)
            # Average

    save(grid(p, ncols=3))
################# NORMS PLOT #################

# name = 'nsu_level_1'
name = 'sob_all'
path = f'input_norms/dv_norm_{name}.txt'
with open(path, 'r') as f:
    s = json.loads(f.read())

axis = [i for i in range(len(s))]
# f = figure(title=f'Norms of doc2vec vectors for {name}')
# f.vbar(x=axis, top=s, width=0.7, color='red')
# show(f)















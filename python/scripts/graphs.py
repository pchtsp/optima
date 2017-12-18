import package.aux as aux
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
# import pandas.rpy.common as com
import palettable.colorbrewer as cbrewer
import package.tests as tests

# import rpy2.robjects.lib.timevis as timevis
# we start with some pickle

# directory_path = '/home/pchtsp/Documents/projects/OPTIMA/python/experiments/201711231500/'
directory_path = '/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712181704'

# to make a graph, we need to have something like the following:

"""
id    start      end         content   group     style
1    2015-01     2019-09         <NA>  D601      background-color:;border-color:
5    2019-09     2019-10     [ 0,  4)  D601      background-color:#FFFFB2;border-color:#FFFFB2
"""


indeces = ['UNIT', 'month']
cols = ['state', 'task', 'used']
table = {}
for col in cols:
    tup = [(k[0], k[1], v) for k, v in solution[col].items()]
    table[col] = pd.DataFrame(tup, columns=indeces + [col]).set_index(indeces)

table_n = \
    table['state'].\
    join(table['task']).\
    join(table['used'])

table_n['content'] = np.where(table_n.state != 'V',
                         table_n.state,
                         table_n['task'] + ' (' + table_n['used'].map(str) + ')')

table_nn = table_n[['content', 'task', 'state']].reset_index()
last_period = table_nn.month.values.max()

table_nn.sort_values(indeces, inplace=True)
table_nn['prev'] = table_nn.groupby('UNIT').content.shift(1)
table_nn['nrow'] = table_nn.groupby('UNIT').cumcount() + 1

table_nn = table_nn[np.logical_or(table_nn.prev != table_nn.content, table_nn['nrow']==1)]
table_nn['end'] = table_nn.groupby('UNIT').month.shift(-1)

table_nn.end = np.where(pd.isna(table_nn.end), last_period, table_nn.end)

colors = {'A': "white", 'M': '#FFFFB2', 'V': "#BD0026"}

table_nn['color'] = table_nn.state.replace(colors)
table_nn['style'] = table_nn.color.map("background-color:{0};border-color:{0}".format)

table_nn = table_nn.reset_index().reset_index().\
    rename(columns={'level_0': 'id', 'UNIT': 'group', 'month': 'start'})\
    [['id', 'start', 'end', 'content', 'group', 'style']]

groups = pd.DataFrame({'id': table_nn.group.unique(), 'content': table_nn.group.unique()})

# cbrewer.sequential.YlOrRd_5.hex_colors


# ro.r('''
#        library(timevis)
# ''')

timevis = importr('timevis')
htmlwidgets = importr('htmlwidgets')
# rdf = com.convert_to_r_dataframe(df)
rdf = pandas2ri.py2ri(table_nn)
rdfgroups = pandas2ri.py2ri(groups)

options = ro.ListVector({
    "stack": False,
    "editable": True,
    "align": "center",
    "orientation": "top",
    # "snap": None,
    "margin": 0
    })

graph = timevis.timevis(rdf, groups= rdfgroups, options = options, width="100%")

print(graph)

htmlwidgets.saveWidget(graph, file= "/home/pchtsp/Downloads/calendar_20171124.html", selfcontained=False)

# pandas2ri.ri2py()

 # r_getname = robjects.globalenv['getname']
# table_nn = table_n[table_n.state != 'A'].state.reset_index()
# qual._load_maps_by_type()
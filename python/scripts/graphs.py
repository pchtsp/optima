import package.aux as aux
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
# import pandas.rpy.common as com
from rpy2.robjects import pandas2ri
# import rpy2.robjects.lib.timevis as timevis
# we start with some pickle

directory_path = '/home/pchtsp/Documents/projects/OPTIMA/python/experiments/201711231500/'

solution = aux.load_solution(directory_path+"data_out.pickle")

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

table_n.state = np.where(table_n.state != 'V',
                         table_n.state,
                         table_n['task'] + ' (' + table_n['used'].map(str) + ')')

# table_nn = table_n[table_n.state != 'A'].state.reset_index()
table_nn = table_n.state.reset_index()
    # .\
    # rename(columns={'UNIT':'id', 'state': 'content'})

table_nn.sort_values(indeces, inplace=True)
table_nn['prev'] = table_nn.groupby('UNIT').state.shift(1)
table_nn['nrow'] = table_nn.groupby('UNIT').cumcount() + 1

table_nn[np.logical_or(table_nn.prev==table_nn.state, table_nn['nrow']==1)]
table_nn['end'] = table_nn.groupby('UNIT').month.shift(-1)

ro.r('''
       library(timevis)
''')

timevis = importr('timevis')
# rdf = com.convert_to_r_dataframe(df)
timevis.timevis(pandas2ri.py2ri(table_nn))


# pandas2ri.ri2py()

 # r_getname = robjects.globalenv['getname']
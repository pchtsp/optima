from rpy2 import robjects
from rpy2.robjects.vectors import IntVector, FloatVector
from rpy2.robjects.lib import grid
from rpy2.robjects.packages import importr, data
import rpy2.robjects.lib.ggplot2 as ggplot2
from rpy2.robjects import Formula, Environment, pandas2ri
from rpy2.rinterface import RRuntimeError
import warnings
import math, datetime
import rpy2.robjects as ro
base = importr('base')

# rprint = robjects.globalenv.get("print")
# grdevices = importr('grDevices')
# grid.activate()

def example():
    stats = importr('stats')
    datasets = importr('datasets')
    mtcars = data(datasets).fetch('mtcars')['mtcars']
    rnorm = stats.rnorm
    dataf_rnorm = robjects.DataFrame({'value': rnorm(300, mean=0) + rnorm(100, mean=3),
                                      'other_value': rnorm(300, mean=0) + rnorm(100, mean=3),
                                      'mean': IntVector([0, ]*300 + [3, ] * 100)})

    gp = ggplot2.ggplot(mtcars)

    pp = gp + \
         ggplot2.aes_string(x='wt', y='mpg') + \
         ggplot2.geom_point() +\
         ggplot2.theme_minimal() + \
         ggplot2.theme(**{'axis.text.x': ggplot2.element_text(angle=45)})

def boxplot(table, x, y, xlab=None, ylab=None):
    pandas2ri.activate()
    if xlab is None:
        xlab = x
    if ylab is None:
        ylab = y
    theme_params = {
        'axis.text.x': ggplot2.element_text(xlab, angle = 45),
        'axis.text.y': ggplot2.element_text(ylab),
    }
    return ggplot2.ggplot(table) + \
           ggplot2.aes_string(x=x, y=y) + \
           ggplot2.geom_boxplot() + \
           ggplot2.theme_minimal() + \
           ggplot2.theme(**theme_params)

# + ggplot2.theme(axis.text.x = ,
#           axis.text.y = element_text(face="bold", color="#993333",
#                            size=14, angle=45))
# theme(axis_text_x  = element_text(angle = 90, hjust = 1))

#
# pp.plot()
#
# grdevices.dev_off()
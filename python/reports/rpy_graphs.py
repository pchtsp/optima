from rpy2 import robjects
from rpy2.robjects.vectors import IntVector, FloatVector, StrVector
# from rpy2.robjects.lib import grid
from rpy2.robjects.packages import importr, data
from rpy2.robjects import Formula, Environment, pandas2ri
from rpy2.robjects.packages import STAP

# from rpy2.rinterface import RRuntimeError
# import warnings
# import math, datetime
# import rpy2.robjects as ro
base = importr('base')

# rprint = robjects.globalenv.get("print")
# grdevices = importr('grDevices')
# grid.activate()

def example():
    import rpy2.robjects.lib.ggplot2 as ggplot2
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
    import rpy2.robjects.lib.ggplot2 as ggplot2
    l2p = importr('latex2exp')
    pandas2ri.activate()
    if xlab is None:
        xlab = x
    if ylab is None:
        ylab = y
    theme_params = {
        # angle = 90,
        'axis.text.x': ggplot2.element_text(size=20, hjust=0),
        'axis.text.y': ggplot2.element_text(size=20),
        'axis.title': ggplot2.element_text(size=20, face="bold")
    }
    values_nn = StrVector(table[x])
    plot = ggplot2.ggplot(table) + \
           ggplot2.aes_string(x=x, y=y) + \
           ggplot2.geom_boxplot() + \
           ggplot2.theme_minimal() + \
           ggplot2.theme(**theme_params)+ \
           ggplot2.labs(x=xlab, y=ylab)+ \
           ggplot2.scale_x_discrete(labels=l2p.TeX(values_nn)) + \
           ggplot2.coord_flip()
    # print(plot)
    return plot


# TODO: generalize
def bars(table, x, y, xlab=None, ylab=None):
    import rpy2.robjects.lib.ggplot2 as ggplot2
    pandas2ri.activate()
    if xlab is None:
        xlab = x
    if ylab is None:
        ylab = y
    theme_params = {
        'axis.text.x': ggplot2.element_text(angle = 45, size=14, hjust=1, vjust=1),
        'axis.text.y': ggplot2.element_text(size=14),
        'axis.title': ggplot2.element_text(size=14, face="bold")
    }
    plot = ggplot2.ggplot(table) + \
           ggplot2.aes_string(x=x, y=y) + \
           ggplot2.geom_bar(stat="identity") + \
           ggplot2.theme_minimal() + \
           ggplot2.theme(**theme_params)+ \
           ggplot2.labs(x=xlab, y=ylab)
    # print(plot)
    return plot


def gantt_experiment(path_to_experiment, path_script='./../R/functions/import_results.R'):

    with open(path_script, 'r') as f:
        string = f.read()
    import_results = STAP(string, "import_results")

    import_results.print_solution_and_print(path_to_experiment, width='100%')
    pass

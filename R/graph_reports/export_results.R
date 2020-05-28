source('nps_reports/functions.R')
source('graph_reports/datasets.R')

stop()

data <- get_generic_compare(c('prise_srv3_20200527_2', 'prise_srv3_20200527'),
                    exp_names = list('medium', 'small')
                    )
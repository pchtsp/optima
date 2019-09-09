library(reticulate)
library(tidyverse)
library(knitr)
library(magrittr)
library(ggplot2)

path_docs <- '/Documents'
path_docs <- ''
use_virtualenv('~%s/projects/OPTIMA/python/venv/' %>% sprintf(path_docs), required = TRUE)
# print(current_dir)
py_discover_config()

os = import('os')
sysp = import('sys')
opt_path = ''
# opt_path = '../'
current_dir = getwd()
python_path <- '%s/%s../python/' %>% sprintf(current_dir, opt_path)
scripts_path = paste0(python_path, 'scripts')
stochs_path = paste0(python_path, 'stochastic')

# path = os.path.join(r.current_dir, rel_path_python)
# os$path$join(current_dir, path_python)
# print(path)

sto <- import_from_path('stochastic_analysis', path=scripts_path)
mod <- import_from_path('models', path=stochs_path)
aux <- import_from_path('tools', path=stochs_path)

batch = sto$get_batch()
result_tab = sto$get_treated_table(batch)

table = result_tab %>% filter(gap_abs <100 & errors==0)
x_vars = 
    c('mean_consum', 'mean_consum2', 'mean_consum3', 'mean_consum4', 'init', 'var_consum', 'spec_tasks', 'geomean_cons'
      ) %>% sort


opt = list(q=0.1, bound='lower', y_var='maints', 
           x_vars = x_vars,
           result_tab = table,
           plot_args=list(facet='mean_consum_cut ~ geomean_cons_cut', x='init'),
           plot=FALSE)

response = do.call(mod$test_regression, opt)
clf <- response[1]
mean_std <- response[2]
X_all_norm = aux$normalize_variables(table[x_vars,], mean_std)
y_pred = clf$predict(X_all_norm)
X <- table %>% mutate(pred = y_pred)
# add qcuts

graph_name = '%s_mean_consum_%s_%s_R' %>% sprintf(method, bound, y_var)

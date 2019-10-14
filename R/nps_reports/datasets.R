library(tidyverse)
library(reticulate)
source('nps_reports/functions.R')

get_python_module <- function(rel_path, name){
    use_virtualenv('~/Documents/projects/OPTIMA/python/venv/', required = TRUE)
    # use_condaenv('cvenv', conda='c:/Anaconda3/Scripts/conda.exe', required=TRUE)
    py_discover_config()
    sysp = import('sys')
    opt_path = ''
    python_path <- '%s/%s../python/' %>% sprintf(getwd(), opt_path)
    sysp$path <- c(python_path, sysp$path)
    scripts_path = paste0(python_path, rel_path)
    compare_sto <- import_from_path(name, path=scripts_path)
    compare_sto
}

get_compare_sto <- function() get_python_module('scripts', 'compare_stochastic')

get_result_tab <- function(name){
    sto_funcs <- get_python_module('scripts', 'stochastic_analysis')
    # batches <- get_python_module('package', 'batch')
    # models <- get_python_module('stochastic', 'models')
    # import stochastic.models as models
    result_tab <- sto_funcs$get_table_complete(name=name, use_zip=TRUE)
    result_tab
}

get_quantile_prev <- function(){
    # TODO
}

get_1_tasks <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190725', 'IT000125_20190716')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        mutate(experiment=if_else(experiment==0, 'cuts', 'base'))
}

get_3_tasks <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190730', 'IT000125_20190801')
    equiv <- list('cuts', 'base')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_3') %>% 
        mutate(experiment=equiv[experiment+1] %>% unlist)
}

get_3_tasks_perc_add <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190801', 'IT000125_20190913')
    equiv <- list('base', 'cuts')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_3') %>% 
        mutate(experiment=equiv[experiment+1] %>% unlist) %>% 
        filter_all_exps
}

get_2_tasks <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190730', 'IT000125_20190729')
    equiv <- list('cuts', 'base')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_2') %>% 
        mutate(experiment=equiv[experiment+1] %>% unlist)
}

get_2_tasks_perc_add <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190729', 'IT000125_20190913')
    equiv <- list('base', 'cuts')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_2') %>% 
        mutate(experiment=equiv[experiment+1] %>% unlist) %>% 
        filter_all_exps
}

get_2_tasks_aggresive <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190808', 'IT000125_20190729')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_2') %>% 
        mutate(experiment=if_else(experiment==0, 'cuts', 'base'))
}

get_3_tasks_aggresive <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190808', 'IT000125_20190801')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_3') %>% 
        mutate(experiment=if_else(experiment==0, 'cuts', 'base'))
}

get_3_tasks_aggresive_perc_add <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190915', 'IT000125_20190801')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_3') %>% 
        mutate(experiment=if_else(experiment==0, 'cuts', 'base'))
}

get_4_tasks <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190730', 'IT000125_20190828')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_4') %>% 
        mutate(experiment=if_else(experiment==0, 'cuts', 'base'))
}

get_4_tasks_aggressive <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190812', 'IT000125_20190828')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_4') %>% 
        mutate(experiment=if_else(experiment==0, 'cuts', 'base'))
}

get_4_tasks_perc_add <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190828', 'IT000125_20190913')
    df_original <- compare_sto$get_df_comparison(exp_list)
    equiv <- list('base', 'cuts')
    df_original %>% 
        filter(scenario=='numparalleltasks_4') %>% 
        mutate(experiment=equiv[experiment+1] %>% unlist) %>% 
        filter_all_exps
}

get_1_tasks_CBC <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190815', 'IT000125_20190827')
    equiv <- list('cuts', 'base')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>%
        mutate(experiment=equiv[experiment+1] %>% unlist)
    
}

get_1_tasks_CBC_CPLEX <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190815', 'IT000125_20190827', 'IT000125_20190716')
    equiv <- list('cuts', 'base', 'cplex_base')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_1') %>% 
        mutate(experiment=equiv[experiment+1] %>% unlist) %>% 
        mutate(time=pmin(3600, time)) %>% 
        filter_all_exps
    
}

get_1_tasks_perc_add <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190716', 'IT000125_20190913')
    df_original <- compare_sto$get_df_comparison(exp_list)
    equiv <- list('base', 'cuts')
    df_original %>% 
        filter(scenario=='numparalleltasks_1') %>% 
        mutate(experiment=equiv[experiment+1] %>% unlist) %>% 
        filter_all_exps
}

get_1_tasks_maints <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190829', 'IT000125_20190716')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_1') %>% 
        filter_all_exps %>% 
        mutate(experiment=if_else(experiment==0, 'cuts', 'base'))
}

get_4_tasks_maints <- function(){
    # too many infeasibles
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190829', 'IT000125_20190828')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_4') %>% 
        filter_all_exps %>% 
        mutate(experiment=if_else(experiment==0, 'cuts', 'base'))
}

get_4_tasks_very_aggresive_percadd <- function(){
    # too many infeasibles
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190917', 'IT000125_20190828')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_4') %>% 
        filter_all_exps %>% 
        mutate(experiment=if_else(experiment==0, 'cuts', 'base'))
}

get_all_tasks <- function(){
    compare_sto <- get_compare_sto()
    base <- c('IT000125_20190828', 'IT000125_20190801', 'IT000125_20190729', 'IT000125_20190730')
    equiv <- c(rep('base', 3), "cuts")  %>% lapply(function(x) x)
    dataset <- base %>% lapply(function(x) x)
    df_original <- compare_sto$get_df_comparison(base)
    df_original %>% 
        mutate(dataset=dataset[experiment+1] %>% unlist,
               experiment=equiv[experiment+1] %>% unlist)
}

get_all_tasks_aggresive <- function(){
    compare_sto <- get_compare_sto()
    base <- c('IT000125_20190828', 'IT000125_20190801', 'IT000125_20190729', 'IT000125_20190808', 'IT000125_20190812')
    equiv <- c(rep('base', 3), "cuts", "cuts")  %>% lapply(function(x) x)
    dataset <- base %>% lapply(function(x) x)
    df_original <- compare_sto$get_df_comparison(base)
    df_original %>% 
        mutate(dataset=dataset[experiment+1] %>% unlist,
               experiment=equiv[experiment+1] %>% unlist)
}

get_all_tasks_aggresive_percadd <- function(){
    compare_sto <- get_compare_sto()
    base <- c('IT000125_20190828', 'IT000125_20190801', 'IT000125_20190729', 'IT000125_20190915')
    equiv <- c(rep('base', 3), "cuts")  %>% lapply(function(x) x)
    dataset <- base %>% lapply(function(x) x)
    df_original <- compare_sto$get_df_comparison(base)
    df_original %>% 
        mutate(dataset=dataset[experiment+1] %>% unlist,
               experiment=equiv[experiment+1] %>% unlist)
}

get_all_tasks_very_aggresive_percadd <- function(){
    compare_sto <- get_compare_sto()
    base <- c('IT000125_20190828', 'IT000125_20190801', 'IT000125_20190729', 'IT000125_20190917')
    equiv <- c(rep('base', 3), "cuts")  %>% lapply(function(x) x)
    dataset <- base %>% lapply(function(x) x)
    df_original <- compare_sto$get_df_comparison(base)
    df_original %>% 
        mutate(dataset=dataset[experiment+1] %>% unlist,
               experiment=equiv[experiment+1] %>% unlist)
}
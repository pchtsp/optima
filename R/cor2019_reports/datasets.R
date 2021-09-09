library(reticulate)
source('nps_reports/datasets.R')

path_to_python <- '~/Documents/projects/research/optima/python/venv/'
get_compare_sto <- function() get_python_module('reports', 'compare_experiments', path_to_python=path_to_python)

get_generic_compare <- function(dataset_list, exp_names, scenario_filter=NULL, ...){
    compare_sto <- get_compare_sto()
    compare_sto$params$PATHS[['results']] = '/home/pchtsp/f_gdrive/Nextcloud/HOmeBOX/Documents/optima_results/'
    df_original <- compare_sto$get_df_comparison(exp_list=dataset_list, 
                                                 scenarios=scenario_filter, 
                                                 zip=TRUE,
                                                 get_progress=FALSE,
                                                   ...)
    dataset_names <- dataset_list %>% lapply(function(x) x)
    df_original %<>% 
        mutate(dataset=dataset_names[experiment+1] %>% unlist,
               experiment=exp_names[experiment+1] %>% unlist)
    if (scenario_filter %>% is.null){
        return(df_original)
    }
    tab_filter <- data.frame(scenario=scenario_filter, stringsAsFactors =FALSE)
    df_original %>% semi_join(tab_filter)
}

get_anaylis_sto <- function() get_python_module('scripts', 'stochastic_analysis', path_to_python=path_to_python)

get_result_tab <- function(name){
    sto_funcs <- get_anaylis_sto()
    sto_funcs$params$PATHS[['results']] = '/home/pchtsp/f_gdrive/Nextcloud/HOmeBOX/Documents/optima_results/'
    result_tab <- sto_funcs$get_table_complete(name=name, use_zip=TRUE)
    result_tab
}

if (FALSE){
    
}
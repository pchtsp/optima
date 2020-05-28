library(reticulate)
source('nps_reports/functions.R')

get_python_module <- function(rel_path, name){
    use_virtualenv('~/Documents/projects/OPTIMA/python/venv/', required = TRUE)
    # use_condaenv('cvenv', conda='c:/Anaconda3/Scripts/conda.exe', required=TRUE)
    # c:\Anaconda3\envs\cvenv\Scripts\pip.exe install -r requirements
    py_discover_config()
    sysp = import('sys')
    opt_path = ''
    python_path <- '%s/%s../python/' %>% sprintf(getwd(), opt_path)
    sysp$path <- c(python_path, sysp$path)
    scripts_path = paste0(python_path, rel_path)
    compare_sto <- import_from_path(name, path=scripts_path)
    compare_sto
}

get_compare_sto <- function() get_python_module('articles', 'Graphs2020')

get_generic_compare <- function(dataset_list, scenario_filter=NULL, exp_names=NULL, get_progress=FALSE, zip=TRUE){
    compare_sto <- get_compare_sto()
    if (exp_names %>% is.null){
        # This assumes the dataset_list has only two datasets!
        # (which is the most likely option)
        exp_names <- list('cuts', 'base')
    }
    if (scenario_filter %>% is.null %>% not){
        # just in case the length is 1, we want python to take it as a list.
        # so it has to have length 2
        scenario_filter <- c(scenario_filter, 'WORKAROUND')
    }
    df_original <- compare_sto$compare_experiments(exp_list=dataset_list)
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
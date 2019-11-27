library(reticulate)
source('nps_reports/functions.R')

get_python_module <- function(rel_path, name){
    # use_virtualenv('~/Documents/projects/OPTIMA/python/venv/', required = TRUE)
    use_condaenv('cvenv', conda='c:/Anaconda3/Scripts/conda.exe', required=TRUE)
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

get_compare_sto <- function() get_python_module('scripts', 'compare_stochastic')

get_result_tab <- function(name){
    sto_funcs <- get_python_module('scripts', 'stochastic_analysis')
    # batches <- get_python_module('package', 'batch')
    # models <- get_python_module('stochastic', 'models')
    # import stochastic.models as models
    result_tab <- sto_funcs$get_table_complete(name=name, use_zip=TRUE)
    result_tab
}

get_generic_compare <- function(dataset_list, scenario_filter=NULL, exp_names=NULL, get_progress=FALSE){
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
    df_original <- compare_sto$get_df_comparison(exp_list=dataset_list, 
                                                 scenarios=scenario_filter, 
                                                 get_progress=get_progress)
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

get_1_tasks <- function(){
    get_generic_compare(c('IT000125_20190725', 'IT000125_20190716'), 
                        exp_names = list('cuts', 'base'))
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

get_fixLP_2_tasks <- function(){
    get_generic_compare(c('IT000125_20191023', 'IT000125_20190729'), 
                        exp_names = list('cuts', 'base'), 
                        scenario_filter='numparalleltasks_2')
}

get_old_new_2_tasks <- function(){
    get_generic_compare(c('IT000125_20190729', 'IT000125_20191030'), 
                        exp_names = list('base', 'cuts'), 
                        scenario_filter='numparalleltasks_2', get_progress=FALSE) %>% 
        correct_old_model
    # substract 2* fleet_size * horizon_size from objectives.
}
get_old_new_3_tasks <- function(){
    get_generic_compare(c('IT000125_20190801', 'IT000125_20191030'), 
                        exp_names = list('base', 'cuts'), 
                        scenario_filter='numparalleltasks_3', get_progress=FALSE) %>% 
        correct_old_model
    # substract 2* fleet_size * horizon_size from objectives.
}
get_old_new_1_tasks <- function(){
    get_generic_compare(c('IT000125_20190716', 'IT000125_20191030'), 
                        exp_names = list('base', 'cuts'), 
                        scenario_filter='numparalleltasks_1', get_progress=FALSE) %>% 
        correct_old_model
    # substract 2* fleet_size * horizon_size from objectives.
}
get_old_all <- function(){
    exp_names <- c(rep('base', 3), "cuts")  %>% lapply(function(x) x)
    result <- get_generic_compare(c('IT000125_20190729', 'IT000125_20190801', 'IT000125_20190828', 'IT000125_20191030'), 
                        exp_names = exp_names)
    result %>% correct_old_model

}
get_old_new_4_tasks <- function(){
    get_generic_compare(c('IT000125_20190828', 'IT000125_20190730', 'IT000125_20191030'), 
                        exp_names = list('base', 'cuts', 'old'), 
                        scenario_filter='numparalleltasks_4') %>% 
        correct_old_model
}
get_old_new_4_agg_tasks <- function(){
    get_generic_compare(c('IT000125_20190828', 'IT000125_20190917', 'IT000125_20191030'), 
                        exp_names = list('base', 'cuts', 'old'), 
                        scenario_filter='numparalleltasks_4') %>% 
        correct_old_model
}
get_old_cuts_4_agg_tasks <- function(){
    get_generic_compare(c('IT000125_20190917', 'IT000125_20191030'), 
                        exp_names = list('base', 'old'), 
                        scenario_filter='numparalleltasks_4') %>% 
        correct_old_model
}
get_flexFixLP_2_tasks <- function(){
    get_generic_compare(c('IT000125_20190729', 'IT000125_20191025', 'IT000125_20190915', 'IT000125_20191023'), 
                        exp_names = list('base', 'cuts', 'cuts_ref', 'cuts_relax'), 
                        scenario_filter='numparalleltasks_2')
}

get_all_compare_2 <- function(){
    get_generic_compare(c('IT000125_20190729', 'IT000125_20191017', 'IT000125_20190915', 'IT000125_20191023', 'IT000125_20191025'),
                        exp_names = list('base', 'determ', 'cuts_sto', 'fixLP', 'flexLP'),
                        scenario_filter='numparalleltasks_2')
}

get_all_2 <- function(){
    get_generic_compare(c('IT000125_20190729', 'IT000125_20191017', 'IT000125_20190915', 'IT000125_20191030', 'IT000125_20191025'),
                        exp_names = list('base', 'cuts_determ', 'cuts_sto', 'old', 'cuts_FlexLP'),
                        scenario_filter='numparalleltasks_2') %>% 
        correct_old_model
}

get_determ_3 <- function(){
    get_generic_compare(c('IT000125_20190801', 'IT000125_20191017', 'IT000125_20190917', 'IT000125_20191030', 'IT000125_20191105'),
                        exp_names = list('base', 'determ', 'stoch', 'old', 'stoch_determ'),
                        scenario_filter='numparalleltasks_3') %>% 
        correct_old_model
}

get_all_fixLP <- function(){
    get_generic_compare(c('IT000125_20190729', 'IT000125_20191104', 'IT000125_20191023', 'IT000125_20191025', 'IT000125_20190917'),
                        exp_names = list('base', 'FlexLP_3', 'fixLP', 'FlexLP', 'stoch'),
                        scenario_filter='numparalleltasks_2') %>% 
        correct_old_model
}

get_all_stoch <- function(){
    get_generic_compare(c('IT000125_20190729', 'IT000125_20190808', 'IT000125_20191030', 'IT000125_20191125'),
                        exp_names = list('base', 'stoch_agg', 'old', 'old_stoch'),
                        scenario_filter='numparalleltasks_2') %>% 
        correct_old_model
}

correct_old_model <- function(data){
    # Function that discounts a fix number from the objective function in the old model.
    # So it can be compared with the new one.
    # The fixed number depends on the size of the problem, the time horizon.
    # This assumes that the input dataframe has "dataset=IT000125_20191025_2" for the old model
    correction <- 
        CJ(dataset=c('IT000125_20191025_2', 'IT000125_20191030'), horizon=89, num_tasks=c(1, 2, 3, 4)) %>% 
        mutate(scenario=sprintf("numparalleltasks_%s", num_tasks),
               correction_value = 2*15*num_tasks*horizon) %>% 
        select(dataset, scenario, correction_value)
    
    correct_fun <- function(value, variable) if_else(variable %>% is.na, value, value-variable)
    
    data %>% 
        left_join(correction) %>%
        mutate(best_solution= best_solution %>% correct_fun(correction_value),
               best_bound= best_bound %>% correct_fun(correction_value)) %>% 
        select(-correction_value)
}
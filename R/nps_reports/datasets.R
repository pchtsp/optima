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

get_compare_sto <- function() get_python_module('scripts', 'compare_stochastic')

get_result_tab <- function(name){
    sto_funcs <- get_python_module('scripts', 'stochastic_analysis')
    # batches <- get_python_module('package', 'batch')
    # models <- get_python_module('stochastic', 'models')
    # import stochastic.models as models
    result_tab <- sto_funcs$get_table_complete(name=name, use_zip=TRUE)
    result_tab
}

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
    df_original <- compare_sto$get_df_comparison(exp_list=dataset_list, 
                                                 scenarios=scenario_filter, 
                                                 get_progress=get_progress,
                                                 zip=zip)
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

get_base <- function(num_tasks){
    list(
        'IT000125_20190716',
        'IT000125_20190729',
        'IT000125_20190801',
        'IT000125_20190828'
    )[[num_tasks]]
    'IT000125_20191204'
    
}

get_1_tasks <- function(){
    get_generic_compare(c('IT000125_20190725', 'IT000125_20190716'), 
                        exp_names = list('cuts', 'base'))
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

get_old_all <- function(){
    exp_names <- c(rep('base', 3), "cuts")  %>% lapply(function(x) x)
    result <- get_generic_compare(c('IT000125_20190729', 'IT000125_20190801', 'IT000125_20190828', 'IT000125_20191030'), 
                        exp_names = exp_names)
    result %>% correct_old_model

}

get_determ <- function(num_tasks, ...){
    get_generic_compare(c(get_base(num_tasks), 'IT000125_20191017', 'IT000125_20190915', 'IT000125_20191030', 'IT000125_20191105'),
                        exp_names = list('base', 'determ', 'stoch', 'old', 'stoch_determ'),
                        scenario_filter='numparalleltasks_%s' %>% sprintf(num_tasks)) %>% 
        correct_old_model
}

get_all_fixLP <- function(...){
    get_generic_compare(c(get_base(2), 'IT000125_20191104', 'IT000125_20191023', 'IT000125_20191025', 'IT000125_20190917'),
                        exp_names = list('base', 'base_flp3', 'base_flp', 'base_flp2', 'base_a2r'),
                        scenario_filter='numparalleltasks_2') %>% 
        correct_old_model
}

get_all_stoch <- function(num_tasks, ...){
    get_generic_compare(c(get_base(num_tasks), 'IT000125_20190915', 'IT000125_20190917', 'IT000125_20191130', 
                          'IT000125_20191030', 'IT000125_20191207'),
                        exp_names = list('base', 'base_a1r', 'base_a2r', 'base_a3r', 
                                         'old', 'old_a2r'),
                        scenario_filter='numparalleltasks_%s' %>% sprintf(num_tasks)) %>% 
        correct_old_model
}

get_stoch_a2r <- function(num_tasks, ...){
    get_generic_compare(c(get_base(num_tasks), 'IT000125_20190917', 'IT000125_20191030', 'IT000125_20191207'),
                        exp_names = list('base', 'base_a2r', 'old', 'old_a2r'),
                        scenario_filter='numparalleltasks_%s' %>% sprintf(num_tasks)) %>% 
        correct_old_model
}

get_min_usage_5 <- function(){
    get_generic_compare(c('IT000125_20191122', 'IT000125_20191124'),
                        exp_names = list('base', 'old'),
                        scenario_filter='minusageperiod_5') %>% 
        correct_old_model
}

correct_fun <- function(value, variable) if_else(variable %>% is.na, value, value-variable)

correct_old_model <- function(data, get_progress=FALSE, keep_correction=FALSE){
    # Function that discounts a fix number from the objective function in the old model.
    # So it can be compared with the new one.
    # The fixed number depends on the size of the problem, the time horizon.
    # This assumes that the input dataframe has "dataset=IT000125_20191025_2" for the old model
    manual_tab <- data.table(dataset='IT000125_20191124', horizon=89, num_tasks=1, scenario='minusageperiod_5')
    correction <- 
        CJ(dataset=c('IT000125_20191207', 'IT000125_20191025_2', 
                     'IT000125_20191030', 'IT000125_20191125', 'IT000125_20191207_6'), 
           horizon=89, num_tasks=c(1, 2, 3, 4)) %>% 
        mutate(scenario=sprintf("numparalleltasks_%s", num_tasks)) %>% 
        bind_rows(manual_tab) %>% 
        mutate(correction_value = 2*15*num_tasks*horizon) %>% 
        select(dataset, scenario, correction_value)
    
    data_n <- data %>% 
        left_join(correction) %>%
        mutate(best_solution= best_solution %>% correct_fun(correction_value),
               best_bound= best_bound %>% correct_fun(correction_value))
    
    if (get_progress){
        data_n %<>% 
            mutate(first_relaxed= first_relaxed %>% correct_fun(correction_value))
    }
    if (keep_correction){
        return(data_n)
    }
    data_n %>% select(-correction_value)
}

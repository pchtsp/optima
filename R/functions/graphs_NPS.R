library(reticulate)
library(tidyverse)
library(knitr)
library(magrittr)
library(ggplot2)
library(data.table)

# setup
get_compare_sto <- function(){
    # sink()
    # print('sdfsdfsq')
    # browser()
    use_virtualenv('~/Documents/projects/OPTIMA/python/venv/', required = TRUE)
    py_discover_config()
    sysp = import('sys')
    opt_path = ''
    python_path <- '%s/%s../python/' %>% sprintf(getwd(), opt_path)
    sysp$path <- c(python_path, sysp$path)
    scripts_path = paste0(python_path, 'scripts')
    compare_sto <- import_from_path('compare_stochastic', path=scripts_path)
    compare_sto
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
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_3') %>% 
        mutate(experiment=if_else(experiment==0, 'cuts', 'base'))
}

get_2_tasks <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190730', 'IT000125_20190729')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_2') %>% 
        mutate(experiment=if_else(experiment==0, 'cuts', 'base'))
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

get_1_tasks_CBC <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190815', 'IT000125_20190827')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>%
        mutate(experiment=if_else(experiment==0, 'cuts', 'base'))
    
}

get_1_tasks_CBC_CPLEX <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190815', 'IT000125_20190827', 'IT000125_20190716')
    df_original <- compare_sto$get_df_comparison(exp_list)
    equiv <- list('cbc_cuts', 'cbc_base', 'cplex_base')
    df_original %>% 
        filter(scenario=='numparalleltasks_1' | scenario=='base') %>% 
        mutate(experiment=equiv[experiment+1]) %>% 
        filter_all_exps

}

get_1_tasks_perc_add <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190725', 'IT000125_20190716', 'IT000125_20190913')
    df_original <- compare_sto$get_df_comparison(exp_list)
    equiv <- list('cuts', 'base', 'perc_add')
    df_original %>% 
        filter(scenario=='numparalleltasks_1' | scenario=='base') %>% 
        mutate(experiment=equiv[experiment+1]) %>% 
        filter_all_exps
    
}


get_1_tasks_maints <- function(){
    compare_sto <- get_compare_sto()
    exp_list <- c('IT000125_20190829', 'IT000125_20190716')
    df_original <- compare_sto$get_df_comparison(exp_list)
    df_original %>% 
        filter(scenario=='numparalleltasks_1' | scenario=='base') %>% 
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

aux_compare <- function(raw_df){
    raw_df %>% 
        gather(key='Indicator', value="value", -experiment) %>% 
        spread(experiment, value) %>% 
        mutate(dif_abs = cuts - base,
               dif_perc = (dif_abs/ base*100) %>% round(2))
    
}

times_100_round <- function(value) value %>% multiply_by(100) %>% round(2)

get_summary <- function(raw_df){

    raw_df %>% 
        mutate(sol_code = if_else(is.na(sol_code), 2, sol_code)) %>% 
        group_by(experiment) %>%
        summarise(Infeasible = sum(sol_code==-1),
                  IntegerFeasible=sum(sol_code==2),
                  IntegerInfeasible=sum(sol_code==0),
                  Optimal=sum(sol_code==1),
                  Total = n()) %>% 
        aux_compare
}

get_comparable_sets <- function(raw_df) {
    data.table(
        Indicator=c('Optimal', 'IntegerFeasible $\\cup$ Optimal', 'Infeasible')
        ,'base $\\cap$ cuts'=c(raw_df %>% get_all_optimal %>% distinct(instance) %>% nrow
                         ,raw_df %>% get_all_integer %>% distinct(instance) %>% nrow
                         ,raw_df %>% get_all_infeasible %>% distinct(instance) %>% nrow)
        )
    
}

get_all_same_status <- function(raw_df, code_status){
    raw_df %>% 
        filter(sol_code==code_status) %>% 
        filter_all_exps
}

get_all_optimal <- function(raw_df) get_all_same_status(raw_df, 1)
get_all_infeasible <- function(raw_df) get_all_same_status(raw_df, -1)
get_all_integer_non_optimal <- function(raw_df) get_all_same_status(raw_df, 2)

get_all_integer <- function(raw_df){
    raw_df %>% 
        filter(sol_code>=1) %>%
        filter_all_exps
}


get_quality_perf <- function(raw_df){
    raw_df %>% 
        get_all_integer %>% 
        select(instance, experiment, best_solution) %>% 
        spread(experiment, best_solution) %>% 
        mutate(dif_perc = ((cuts-base)/ abs(base)) %>% times_100_round)
}

filter_all_exps <- function(table){
    num <- table %>% distinct(experiment) %>% nrow
    table %>% 
        group_by(instance) %>% 
        filter(n()==num) %>% 
        ungroup
        
}

# get_quality_perf_stats <- function(raw_df){
    # quality_perf <- get_quality_perf(raw_df)
    # 
    # data.table(
    #     
    # )
    # mean_dif <- quality_perf$dif_perc %>%  mean
    # max_dif <- quality_perf$dif_perc %>% max
    # min_dif <- quality_perf$dif_perc %>% min
    # per_cut <- (sum(quality_perf$dif_perc<0)/length(quality_perf$dif_perc)) %>% times_100_round
    # med_dif <- quality_perf$dif_perc %>%  median
    
# }

get_quality_degr <- function(raw_df){
    raw_df %>% 
        get_all_optimal %>% 
        group_by(instance) %>%
        mutate(min_value = min(best_solution),
               dist_min = best_solution - min_value,
               dist_min_perc = (best_solution - min_value)/abs(min_value)*100
        )
}

get_time_perf_integer <- function(raw_df){
    raw_df %>% 
        get_all_integer %>% 
        select(instance, experiment, time) %>% 
        spread(experiment, time) %>% 
        arrange(base) %>% 
        mutate(instance = row_number()) %>% 
        gather(key = 'experiment',  value='time', -instance)
}

get_time_perf_integer_reorder <- function(raw_df){
    raw_df %>% 
        get_time_perf_integer %>% 
        group_by(experiment) %>% 
        arrange(experiment, time) %>% 
        mutate(percentage = row_number()/n()*100)
}

get_time_perf_optim <- function(raw_df){
    raw_df %>% 
        get_all_optimal %>% 
        select(experiment, instance, time) %>%
        group_by(experiment) %>% 
        summarise(time_mean = mean(time), 
                  time_medi = median(time)) %>% 
        aux_compare
}

get_infeasible_instances <- function(raw_df){
    raw_df %>% 
        filter(sol_code==-1) %>% 
        distinct(instance)
}

get_infeasible_stats <- function(raw_df){
    raw_df %>% 
        get_infeasible_instances %>% 
        inner_join(raw_df) %>% 
        filter(experiment=="base") %>% 
        summarise(Infeasible = sum(sol_code==-1),
                  IntegerFeasible=sum(sol_code==2),
                  IntegerInfeasible=sum(sol_code==0),
                  Optimal=sum(sol_code==1),
                  Total = n()) %>% 
        gather(key='Status', 'number') %>% 
        filter(number>0)
    
}

get_soft_constraints <- function(raw_df, quant_max){
    raw_df %>% 
        get_all_optimal %>% 
        mutate(errors = replace_na(errors, 0)) %>% 
        group_by(experiment) %>% 
        summarise(errors_mean = mean(errors),
                  errors_q95 = quantile(errors, quant_max)) %>% 
        aux_compare
}

if (FALSE){

}
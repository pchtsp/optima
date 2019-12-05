library(tidyverse)
library(knitr)
library(magrittr)
library(ggplot2)
library(data.table)
library(stringr)

value_filt_tails <- function(value, each_tail) value %>% between(., quantile(., c(each_tail[1])), quantile(., c(1-each_tail[2])))

filter_all_exps <- function(table){
    num <- table %>% distinct(experiment) %>% nrow
    table %>% 
        group_by(scenario, instance) %>% 
        filter(n()==num) %>% 
        ungroup
}

aux_compare <- function(raw_df){
    raw_df %>% 
        gather(key='Indicator', value="value", -experiment, -scenario) %>% 
        mutate(value = value %>% round(2)) %>% 
        spread(experiment, value) 
    # %>% 
    #     mutate(dif_abs = cuts - base,
    #            dif_perc = (dif_abs/ base*100) %>% round(2))
}

times_100_round <- function(value) value %>% multiply_by(100) %>% round(2)

get_summary <- function(raw_df, compare=TRUE){
    result <- 
        raw_df %>% 
        filter_all_exps %>% 
        mutate(sol_code = if_else(is.na(sol_code), 2, sol_code)) %>% 
        group_by(scenario, experiment) %>%
        summarise(Infeasible = sum(sol_code==-1),
                  IntegerFeasible=sum(sol_code==2),
                  IntegerInfeasible=sum(sol_code==0),
                  Optimal=sum(sol_code==1),
                  Total = n())
    if (!compare){
        return(result)
    }
    result %>% aux_compare
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

dif_perc_with_base <- function(value, base) ((value-base)/abs(base)) %>% times_100_round

compare_objectives_perc <- function(df, column='best_solution'){
    aux <- 
        df %>% 
        filter(experiment=='base') %>% 
        select(instance, base=column) %>%
        inner_join(df)
    
    column_value <- aux[[column]]
    aux %>% 
        mutate(dif_perc = dif_perc_with_base(column_value, base),
               dif = column_value - base) %>% 
        filter(experiment!='base')
}

get_quality_perf_2 <- function(raw_df){
    raw_df %>% get_all_integer %>% compare_objectives_perc('best_solution')
}

get_quality_degr_2 <- function(raw_df){
    raw_df %>% get_all_optimal %>% compare_objectives_perc('best_solution')
}

get_quality_perf <- function(raw_df){
    raw_df %>% get_quality_perf_2 %>% filter(experiment=='cuts')
}

get_quality_degr <- function(raw_df){
    raw_df %>% get_quality_degr_2 %>% filter(experiment=='cuts')
}

get_time_perf_integer <- function(raw_df){
    raw_df %>% 
        select(scenario, instance, experiment, time) %>% 
        filter_all_exps %>% 
        spread(experiment, time) %>% 
        arrange(base) %>% 
        mutate(instance = row_number()) %>% 
        gather(key = 'experiment',  value='time', -instance, -scenario) %>% 
        filter(time %>% is.na %>% not)
}

get_time_perf_integer_summ <- function(raw_df){
    raw_df %>% 
        get_time_perf_integer %>% 
        group_by(scenario, experiment) %>% 
        summarise(time_mean = mean(time), 
                  time_medi = median(time)) %>% 
        aux_compare
}

get_time_perf_integer_reorder <- function(raw_df){
    raw_df %>% 
        get_time_perf_integer %>% 
        group_by(experiment) %>% 
        arrange(experiment, time) %>% 
        mutate(percentage = row_number()/n()*100)
}

# when comparing times we compare averages, not average relative differences.
get_time_perf_optim <- function(raw_df){
    raw_df %>% 
        get_all_optimal %>% 
        select(scenario, experiment, instance, time) %>%
        group_by(scenario, experiment) %>% 
        summarise(time_mean = mean(time), 
                  time_medi = median(time)) %>% 
        aux_compare
}

get_infeasible_instances <- function(raw_df){
    raw_df %>% 
        filter_all_exps %>% 
        filter(sol_code==-1) %>% 
        distinct(experiment, instance)
}

get_infeasible_stats <- function(raw_df){
    raw_df %>% 
        filter_all_exps %>% 
        filter(experiment=="base") %>% 
        select(-experiment) %>% 
        inner_join(get_infeasible_instances(raw_df)) %>% 
        filter(experiment!="base") %>% 
        group_by(scenario, experiment) %>% 
        summarise(Infeasible = sum(sol_code==-1),
                  IntegerFeasible=sum(sol_code==2),
                  IntegerInfeasible=sum(sol_code==0),
                  Optimal=sum(sol_code==1),
                  Total = n()) %>% 
        aux_compare
}

get_soft_constraints <- function(raw_df, quant_max=0.95, compare=TRUE){
    result <- 
        raw_df %>% 
        get_soft_constraints_2 %>%  
        group_by(scenario, experiment) %>% 
        summarise(errors_mean = mean(dif),
                  errors_q95 = quantile(dif, quant_max))
    if (!compare){
        return(result)
    }
    result %>% aux_compare
}

get_soft_constraints_2 <- function(raw_df){
    raw_df %>% 
        get_all_optimal %>% 
        mutate(errors = replace_na(errors, 0)) %>% 
        compare_objectives_perc('errors')
}

get_infeasible_times <- function(raw_df){
    raw_df %>% 
        get_all_infeasible %>% 
        group_by(scenario, experiment) %>% 
        summarise(time_mean = mean(time), 
                  time_medi = median(time)) %>% 
        aux_compare
}

# TODO: there are some weird things with a few instances that have variance=NA
# it is because the experiment did not load. I have to check the files...
# they are not that many so we will deal with that later.
get_variances <- function(raw_df){
    raw_df %>% 
        get_all_integer %>% 
        compare_objectives_perc('variance') %>% 
        filter(dif_perc %>% is.na %>% not)
}

get_stats_summary <- function(raw_df_progress){
    raw_df_progress %>% 
        get_stats %>% 
        group_by(scenario, experiment) %>% 
        summarise(nodes_mean = mean(nodes), 
                  nodes_medi = median(nodes),
                  frel_medi = median(first_relaxed),
                  crel_medi = median(cuts_best_bound)
                  ) %>% 
        aux_compare
}

get_stats <- function(raw_df_progress){
    raw_df_progress %>% 
        get_all_optimal %>% 
        mutate(cuts_best_bound= cut_info %>% lapply('[[', 'best_bound')
               %>% as.numeric %>% correct_fun(correction_value))    
}

get_mega_summary <- function(df){
    # comparison table.
    # for each experiment
    # feasability:
    # extra infeasible instances as percent of total
    # extra soft constraints violations (avg, 95%)
    # TODO: time to detect infeasible.
    # TODO: sum of variances.
    # performance:
    # extra feasible instances as percent of total
    # time to solve: median, avg
    # optimality:
    # distance from integer: 
    # distance from optimal: 
    
    
    # feasibility
    summary_stats <- get_summary(df, compare=FALSE)
    errors_stats <- get_soft_constraints(df, 0.95, compare=FALSE)
    feasibility <- 
        summary_stats %>% 
        mutate(InfPerc=(Infeasible/Total) %>% times_100_round) %>% 
        select(experiment, InfPerc) %>% 
        inner_join(errors_stats)
    # performance
    times <- get_time_perf_integer(df)
    performance <- 
        summary_stats %>% 
        mutate(Feasible=((IntegerFeasible+Optimal)/Total) %>% times_100_round) %>% 
        select(experiment, Feasible)
    performance <- 
        times %>% 
        group_by(scenario, experiment) %>% 
        summarise(time_mean = mean(time), 
                  time_medi = median(time)) %>% 
        inner_join(performance)
    # optimality
    optim_degr <- get_quality_degr(df)
    optimality <- 
        optim_degr %>% 
        select(scenario, instance, experiment, dist_min_perc) %>% 
        group_by(scenario, experiment) %>% 
        summarise(q_mean= mean(dist_min_perc) %>% round(2),
                  q_medi= median(dist_min_perc) %>% round(2),
                  q_q95 = quantile(dist_min_perc, 0.95) %>% round(2))
    comparison <- 
        feasibility %>% 
        inner_join(performance) %>%
        inner_join(optimality) %>% 
        aux_compare
    
    # dif_abs:
    dif_abs <- data.table(Indicator=c('Feasible', 'InfPerc', 'q_mean', 'q_medi', 'q_q95'))
    comp_dif_abs <- 
        comparison %>% 
        semi_join(dif_abs) %>% 
        select(scenario, Indicator, dif=dif_abs)
    
    comparison %>% 
        anti_join(dif_abs) %>% 
        select(scenario, Indicator, dif=dif_perc) %>% 
        bind_rows(comp_dif_abs) %>% 
        spread(Indicator, dif)
}


if (FALSE){

}
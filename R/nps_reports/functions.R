library(tidyverse)
library(knitr)
library(magrittr)
library(ggplot2)
library(data.table)
library(stringr)
library(latex2exp)
library(RColorBrewer)
library(ggalluvial)

value_filt_tails <- function(value, each_tail) value %>% between(., quantile(., c(each_tail[1])), quantile(., c(1-each_tail[2])))

filter_all_exps <- function(table){
    num <- 
        table %>% 
        distinct(experiment) %>% nrow
    table %>% 
        filter(sol_code %>% is.na %>% not) %>% 
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
        summarise_states
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

compare_with_base <- function(data, column, abs=TRUE){
    column <- enquo(column)
    ref <- data %>% filter(experiment=='base') %>% ungroup %>% select(!!column) %>% extract2(1)
    if (abs){
        func_a <- function(x) x- ref
    } else {
        func_a <- function(x) ((x- ref)/ref) %>% times_100_round()
    }
    data %>% mutate_at(vars(!!column), func_a)
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
        filter_all_exps %>% 
        select(scenario, instance, experiment, time) %>% 
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

get_transitions_stats <- function(raw_df){
    df <- 
        raw_df %>% 
        inner_join(get_status_from_code()) %>% 
        mutate(experiment_parent = experiment %>% str_replace('\\_.*', ''))
        
    df %>% 
        filter_all_exps %>% 
        filter(experiment %in% c("base", "old")) %>% 
        select(scenario, experiment_parent, instance, prev_status=sol_txt) %>% 
        inner_join(df) %>%
        group_by(scenario, experiment, prev_status, post_status=sol_txt) %>% 
        summarise(num= n())
        
}

get_status_from_code <- function(){
    data.table(
        sol_txt = c('Infeasible', 'IntegerInfeasible', 'IntegerFeasible', 'Optimal') %>% 
            factor(., levels=.),
        sol_code = c(-1, 0, 2, 1)
        )
}

summarise_states <- function(grouped_table){
    grouped_table %>% 
        summarise(Infeasible = sum(sol_code==-1),
                  IntegerFeasible=sum(sol_code==2),
                  IntegerInfeasible=sum(sol_code==0),
                  Optimal=sum(sol_code==1),
                  Total = n())
}

get_infeasible_stats <- function(raw_df){
    raw_df %>% 
        filter_all_exps %>% 
        filter(experiment=="base") %>% 
        select(-experiment) %>% 
        inner_join(get_infeasible_instances(raw_df)) %>% 
        filter(experiment!="base") %>% 
        filter(sol_code %>% is.na %>% not) %>% 
        group_by(scenario, experiment) %>% 
        summarise_states %>% 
        aux_compare %>% 
        gather(key="case", 'value', -scenario, -Indicator) %>% 
        filter(value>0) %>% 
        spread(case, value, fill = 0) 
        
}

get_soft_constraints <- function(raw_df, compare=TRUE){
    result <- 
        raw_df %>% 
        get_soft_constraints_2 %>% 
        group_by(scenario, experiment) %>% 
        summarise(errors_mean = mean(dif),
                  errors_new = (sum(dif>0)/n()) %>% times_100_round)
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
        summarise(nodes = mean(nodes), 
                  LP_first = ((best_solution-first_relaxed)/best_solution) %>% abs %>% mean,
                  LP_cuts = ((best_solution-cuts_best_bound)/best_solution) %>% abs %>% mean,
                  time = mean(time)
                  ) %>% 
        mutate_at(vars(LP_first, LP_cuts), times_100_round) %>% 
        aux_compare
    
    # raw_df_progress %>% 
    #     filter(time %>% is.na %>% not) %>% 
    #     group_by(scenario, experiment) %>% 
    #     summarise(
    # 
    #     ) %>% 
    #     aux_compare
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
    # sum of variances.
    # performance:
    # extra feasible instances as percent of total
    # time to solve: median, avg
    # optimality:
    # distance from integer: 
    # distance from optimal: 
    
    # feasibility
    summary_stats <- get_summary(df, compare=FALSE)
    errors_stats <- get_soft_constraints(df, compare=FALSE)
    feasibility <- 
        summary_stats %>% 
        mutate(InfPerc=(Infeasible/Total) %>% times_100_round) %>% 
        select(experiment, InfPerc) %>% 
        compare_with_base(InfPerc) %>% inner_join(errors_stats)

    # performance
    feas_performance <- 
        summary_stats %>% 
        mutate(Feasible=((IntegerFeasible+Optimal)/Total) %>% times_100_round) %>% 
        select(experiment, Feasible) %>% 
        compare_with_base(Feasible)
    
    performance <- 
        df %>% 
        get_time_perf_integer %>% 
        group_by(scenario, experiment) %>% 
        summarise(time_mean = mean(time)) %>% 
        compare_with_base(time_mean, abs=FALSE) %>% 
        inner_join(feas_performance)
    
    # optimality
    optimality <-
        df %>% get_quality_degr_2 %>% 
        group_by(scenario, experiment) %>% 
        summarise(q_mean= mean(dif_perc) %>% round(2),
                  q_medi= median(dif_perc) %>% round(2),
                  q_q95 = quantile(dif_perc, 0.95) %>% round(2))
    
    # variance
    variances_all <- 
        get_variances(df) %>%
        group_by(scenario, experiment) %>%
        summarise(v_mean = mean(dif_perc) %>% round(2))

    # summary
    comparison <- 
        feasibility %>% 
        inner_join(performance) %>%
        inner_join(optimality) %>% 
        inner_join(variances_all) %>% 
        aux_compare
    
    return(comparison)
# 
#         # dif_abs:
#     dif_abs <- data.table(Indicator=c('Feasible', 'InfPerc', 'q_mean', 'q_medi', 'q_q95'))
#     
#     comp_dif_abs <- 
#         comparison %>% 
#         semi_join(dif_abs)
#     
#     comparison %>% 
#         anti_join(dif_abs) %>% 
#         bind_rows(comp_dif_abs) %>% 
#         spread(Indicator, dif)
}


if (FALSE){

}
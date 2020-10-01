# summary -----------------------------------------------------------------

nullToNA <- function(x) {
    x[sapply(x, is.null)] <- NA
    return(x)
}

get_summary_table <- function(exps, exp_names, wider=TRUE, ...){
    #' This is just a wrapper over get_generic_compare
    #' that cleans and treats the table.
    #' and adds the bounds as a column

    data <- get_generic_compare(exps, exp_names = exp_names, ...)
    data_nn <- 
        data %>% 
        filter_all_exps %>%
        mutate(
            scenario = scenario %>% str_extract('\\d+') %>% as.integer %>% multiply_by(15),
            objective = objective %>% round,
            best_bound = best_bound %>% nullToNA %>% unlist,
        ) %>% 
        select(instance, experiment, scenario, objective, errors, best_bound) %>% 
        filter(instance %>% is.na %>% not) %>% 
        arrange(experiment, scenario, instance)
    
    bounds <- data_nn %>% select(instance, scenario, best_bound) %>% filter(best_bound %>% is.na %>% not)
    
    data_nnn <- 
        data_nn %>% 
        select(-best_bound) %>% 
        inner_join(bounds, by=c('instance', 'scenario'))
    
    if (!wider){
        return(data_nnn)
    }
    summary_to_wider(data_nnn)
    
}

summary_to_wider <- function(summary_table, column='objective'){
    summary_table %>% 
        pivot_wider(id_cols = c(instance, scenario), 
                    names_from = experiment, 
                    values_from = column) %>% 
        arrange(scenario, instance)
}

formated_kable <- function(data, escape=TRUE){
    data_n <- 
        data %>% 
        group_by(scenario) %>% 
        mutate(repeated = row_number()==1) %>% 
        ungroup %>% 
        mutate(scenario = ifelse(repeated, scenario, NaN))
    
    rows <- data_n %>% mutate(row=row_number()) %>% filter(scenario %>% is.na %>% not) %>% use_series(row)
    lines <- rows[-1]-1
    
    data_nn <- 
        data_n %>% 
        select(-repeated) %>% 
        select(scenario, instance, everything()) %>% 
        kable(format = 'latex', booktabs = TRUE, linesep='', escape = escape)
    
    for(l in lines){
        data_nn <- kableExtra::row_spec(data_nn, l, hline_after = T )
    }
    return(data_nn)
}


# progress ----------------------------------------------------------------

get_progress <- function(exps, exp_names, solver, scenario_filter=NULL){
    #' This function gets the progress
    #' and then cleans the table, selecting only relevant columns
    #'
    progress <- get_generic_compare(exps, exp_names = exp_names, get_progress=TRUE, 
                                    solver=solver, scenario_filter=scenario_filter)
    
    progress %>% 
        mutate(scenario = scenario %>% str_extract('\\d+') %>% as.integer %>% multiply_by(15)) %>% 
        select(experiment, scenario, instance, Time, BestInteger) %>% 
        mutate(BestInteger=as.numeric(BestInteger),
               Time = as.numeric(Time)) %>% 
        filter(!is.null(BestInteger))
}

draw_progress <- function(progress, count_preprocess=TRUE, log_scale_y=TRUE){
    
    data_graph <- 
        progress %>% 
        mutate(row = sprintf("i=%s", instance))
    
    # "cheating":
    if (!count_preprocess){
        data_graph <- data_graph %>% group_by(experiment, instance) %>% mutate(Time=Time - min(Time))    
    }
    
    filter_df <- data_graph %>% distinct(experiment, row) %>% group_by(row) %>% filter(., n()>1)
    
    data_graph <- data_graph %>% inner_join(filter_df)
    
    p <- 
        ggplot(data_graph, aes(x=Time, y=BestInteger, group=experiment)) + 
        geom_line(aes(color=experiment, linetype=experiment), size=0.7) + 
        facet_grid(rows='row', scales="free_y") +
        labs(color = "Method", linetype = "Method") + 
        ylab("Objective value")
    if (log_scale_y){
        return(p + scale_y_log10())
    }
    return(p)
}

compare_initial <- function(exps, exp_names, solver=solver, scenario_filter=scenario_filter){
    data_nn <- get_summary_table(exps, exp_names, wider=FALSE)
    data_best <- data_nn %>% group_by(scenario, instance) %>% summarise(best=min(objective))
    progress <- get_progress(exps, exp_names, solver=solver, scenario_filter=scenario_filter)
    
    progress %>% 
        arrange(experiment, scenario, instance, Time) %>% 
        group_by(experiment, scenario, instance) %>% 
        summarise(initial=first(BestInteger)) %>% 
        inner_join(data_best) %>% 
        mutate(initial_p = (initial-best)/best *100) %>% 
        mutate(initial_p = initial_p %>% round) %>% 
        pivot_wider(id_cols = c(instance, scenario), 
                    names_from = experiment, values_from = initial_p)
}

to_list_value <- function(vector, value){
    lapply(sapply(vector, function(z) value), function(z) z)
}
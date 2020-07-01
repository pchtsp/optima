source('nps_reports/functions.R')
source('graph_reports/datasets.R')
library(kableExtra)
# library(gginnards)


# header ------------------------------------------------------------------



opts <- options(knitr.kable.NA = "")
path_export_img <- '../../Graph2020/'
path_export_tab <- '../../Graph2020/tables/'
element_text_size <- 10


# summary -----------------------------------------------------------------

nullToNA <- function(x) {
    x[sapply(x, is.null)] <- NA
    return(x)
}

get_summary_table <- function(exps, exp_names, wider=TRUE, ...){
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
        inner_join(bounds, by=c('instance', 'scenario')) %>% 
        mutate(gap=((objective-best_bound)/objective*100) %>% round(2))
    
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

formated_kable <- function(data){
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
        kable(format = 'latex', booktabs = TRUE, linesep='')
    
    for(l in lines){
        data_nn <- kableExtra::row_spec(data_nn, l, hline_after = T )
    }
    return(data_nn)
}


# progress ----------------------------------------------------------------

get_progress <- function(exps, exp_names, solver, scenario_filter=NULL){
    progress <- get_generic_compare(exps, exp_names = exp_names, get_progress=TRUE, solver=solver, scenario_filter=scenario_filter)
    
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
    
    p = ggplot(data_graph, aes(x=Time, y=BestInteger, color=experiment)) + 
        geom_line(size=0.7) + 
        facet_grid(rows='row', scales="free_y")
    if (log_scale_y){
        return(p + scale_y_log10())
    }
    return(p)
}


# exports -----------------------------------------------------------------

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

compare_large <- function(){
    exps <- c( 'serv_cluster1_20200623', 'serv_cluster1_20200615')
    exp_names <- c('short+mip', 'cplex')
    scenario_filter <- c(6, 7, 8) %>% paste0('numparalleltasks_', .)
    progress <- get_progress(exps, exp_names, solver=list(serv_cluster1_20200623='HEUR'), scenario_filter=scenario_filter)
    progress_n <- progress %>% filter(Time>120)
    path <- '%slarge_datasets_progress_120.png' %>% sprintf(path_export_img)
    y_lab <- 'Best solution found'
    draw_progress(progress_n %>% filter(scenario==120), log_scale_y = TRUE, count_preprocess = FALSE) + 
        ylab(y_lab) + theme_minimal() +  labs(color = "Method") + theme(text = element_text(size=element_text_size)) +
        ggsave(path)
    
    path <- '%slarge_datasets_compare.tex' %>% sprintf(path_export_tab)
    data <- get_summary_table(exps, exp_names, scenario_filter=scenario_filter)
    data_nn <- data %>% mutate(`dif (%)`= ((`short+mip`-cplex)/cplex*100) %>% round(2)) %>% formated_kable
    data_nn %>% write_file(path)
    
}

compare_initial_solution <- function(){
    exps <- c('serv_cluster1_20200701', 'serv_cluster1_20200701')
    exp_names <- c('initial', 'NONE')
    # exps <- c('serv_cluster1_20200610', 'serv_cluster1_20200617_2')
    # exp_names <- c('cplex', 'short+mip')
    # scenario_filter <- c(3, 4, 5) %>% paste0('numparalleltasks_', .)
    path <- '%sinitial_solution_compare.tex' %>% sprintf(path_export_tab)
    data <- get_generic_compare(exps, exp_names = exp_names, solver=list(serv_cluster1_20200701='HEUR'))
    data_n <- 
        data %>% 
        filter(experiment=='initial') %>% 
        mutate(
            scenario= scenario %>% str_extract('_\\w+') %>% str_sub(start=2),
            objective = objective %>% round,
            best_bound = best_bound %>% nullToNA %>% unlist,
        ) %>% 
        select(instance, scenario, objective, errors) %>% 
        filter(instance %>% is.na %>% not) %>% 
        arrange(scenario, instance)
    
    data_n %>%
        mutate(scenario= if_else(scenario=='MaintFirst', 'maintFirst', scenario)) %>% 
        pivot_wider(id_cols = instance, 
                names_from = scenario, 
                values_from = objective) %>% 
        arrange(instance) %>% kable(format = 'latex', booktabs = TRUE, linesep='') %>% write_file(path)
    
    
    # data <- compare_initial(exps, exp_names, solver=list(serv_cluster1_20200617_2='HEUR'))
    # data_n <- data %>% mutate(cplex= sprintf('%s %%', cplex), `short+mip`=sprintf('%s %%', `short+mip`)) %>% formated_kable
    # data_n %>% write_file(path)
}

compare_normal <- function(){
    exps <- c('serv_cluster1_20200610', 'serv_cluster1_20200623')
    exp_names <- c('cplex', 'short+mip')
    scenario_filter <- c(3, 4, 5) %>% paste0('numparalleltasks_', .)
    progress <- get_progress(exps, exp_names, solver=list(serv_cluster1_20200623='HEUR'), scenario_filter=scenario_filter)
    progress_n <- progress %>% filter(Time>60)
    path <- '%snormal_datasets_progress_75.png' %>% sprintf(path_export_img)
    y_lab <- 'Best solution found'
    draw_progress(progress_n %>% filter(scenario==75), log_scale_y = TRUE, count_preprocess = FALSE) + 
        ylab(y_lab) + theme_minimal() +  labs(color = "Method") + theme(text = element_text(size=element_text_size)) +
        ggsave(path)
    
    path <- '%snormal_datasets_compare.tex' %>% sprintf(path_export_tab)
    data_summary <- get_summary_table(exps, exp_names, wider=FALSE, scenario_filter=scenario_filter)
    data <- summary_to_wider(data_summary)
    data <- summary_to_wider(data_summary, column='gap')
    data_nn <- data %>% mutate(`dif (%)`= ((`short+mip`-cplex)/cplex*100) %>% round(2)) %>% formated_kable
    data_nn %>% write_file(path)
}

compare_200aircraft <- function(){
    exps <- c('serv_cluster1_20200625', 'serv_cluster1_20200625_3')
    exp_names <- c('cplex', 'graph_cplex')
    data_nn <- get_summary_table(exps, exp_names, wider=FALSE)
    gaps <- summary_to_wider(data_nn, column='gap')
    path <- '%sgaps200.tex' %>% sprintf(path_export_tab)
    gaps %>% rename(`short+mip  (\%)`=graph_cplex, `largeMip (\%)`=cplex) %>% 
        kable(format = 'latex', booktabs = TRUE, linesep='', escape = FALSE) %>% write_file(path)
    summary_to_wider(data_nn, column='objective')

    
    progress <- get_progress(exps, exp_names, solver=list(serv_cluster1_20200625_3='HEUR', serv_cluster1_20200625_2='HEUR'))
    progress_n <- progress %>% filter(Time>100)
    draw_progress(progress_n %>% filter(scenario==195), log_scale_y = TRUE)
}

to_list_value <- function(vector, value){
    lapply(sapply(vector, function(z) value), function(z) z)
}
compare_neighbors <- function(){
    
    exps <- c('serv_cluster1_20200630_1', 'serv_cluster1_20200630_2', 'serv_cluster1_20200630_3')
    exp_names <- c('short', 'mip', 'shortmip')
    progress <- get_generic_compare(exps, exp_names = exp_names, get_progress=TRUE, 
                                    solver=to_list_value(exps, 'HEUR'))
    
    progress_n <- 
        progress %>% 
        mutate(BestInteger=as.numeric(BestInteger),
                                 Time = as.numeric(Time)) %>% 
        filter(!is.null(BestInteger))
    
    path <- '%scompare_neighbors.png' %>% sprintf(path_export_img)
    progress_n %>% 
        filter(scenario=='solveseed_4') %>% 
        mutate(col = scenario %>% str_extract('\\d+') %>% sprintf("seed=%s", .)) %>% 
        draw_progress(log_scale_y = TRUE) + facet_grid(rows=vars(col), scales="free_y", cols=vars(row)) +
        theme_minimal() +  labs(color = "Method") + theme(text = element_text(size=element_text_size)) +
        theme(legend.position = "bottom",
              legend.box = "vertical") +
        ggsave(path)
    
    stopTime <- 
        progress_n %>% 
        mutate(Time_r= round(Time/70)*70) %>% 
        group_by(experiment, scenario, instance, Time_r) %>% 
        summarise(Min = min(BestInteger),
                  Max = max(BestInteger)) %>% 
        filter(Min+100>=Max) %>% 
        group_by(experiment, scenario, instance) %>% 
        summarise(MinNoChange= min(Time_r))
    
    before_local_minim <- 
        progress_n %>% left_join(stopTime) %>% filter((MinNoChange %>% is.na) | (Time < MinNoChange))
    # visualize it
    before_local_minim %>%  
        draw_progress(log_scale_y = TRUE) + facet_grid(rows=vars(row), scales="free_y", cols=vars(scenario))
    
    # export the table:
    base_table <- 
        before_local_minim %>% 
        arrange(experiment, scenario, instance, Time) %>% 
        group_by(experiment, scenario, instance) %>% 
        mutate(Iterations = n()) %>% 
        slice(n()) %>% select(Time, BestInteger, Iterations) %>% 
        group_by(experiment, instance) %>% 
        summarise_at(vars(Time, BestInteger), list(min=min, max=max)) %>% 
        mutate(Time= sprintf("%s-%s", Time_min, Time_max),
               BestInteger= sprintf("%s-%s", BestInteger_min, BestInteger_max))
    
    # time comparison
    path <- '%scompare_neighbors_time.tex' %>% sprintf(path_export_tab)
    base_table %>% 
        select(experiment, instance, Time) %>% 
        pivot_wider(id_cols=instance, names_from=experiment, values_from=Time) %>% 
        rename(`$t_{mip}$`=mip, `$t_{short}$`=short, `$t_{both}$`=shortmip) %>% 
        kable(format = 'latex', booktabs = TRUE, linesep='', escape = FALSE) %>%
        write_file(path)
    
    # quality comparison
    path <- '%scompare_neighbors_obj.tex' %>% sprintf(path_export_tab)
    base_table %>% 
        select(experiment, instance, BestInteger) %>% 
        pivot_wider(id_cols=instance, names_from=experiment, values_from=BestInteger) %>% 
        rename(`$obj_{mip}$`=mip, `$obj_{short}$`=short, `$obj_{both}$`=shortmip) %>% 
        kable(format = 'latex', booktabs = TRUE, linesep='', escape = FALSE) %>%
        write_file(path)
    
            
        # pivot_wider(id_cols=c(instance, scenario), names_from=experiment, values_from=c(Time, BestInteger)) %>% 
        # arrange(instance, scenario) %>% 
        # mutate(scenario = scenario %>% str_extract('\\d+') %>% as.integer) %>% 
        # rename(seed=scenario, `$t_{mip}$`=Time_mip, `$t_{short}$`=Time_short, `$t_{both}$`=Time_shortmip,
        #        `$obj_{mip}$`=BestInteger_mip, `$obj_{short}$`=BestInteger_short, `$obj_{both}$`=BestInteger_shortmip) %>% 
        # kable(format = 'latex', booktabs = TRUE, linesep='', escape = FALSE) %>%
        # write_file(path)
    
    # time_to_initial_solution <-
    #     progress_n %>%
    #     group_by(experiment, scenario, instance) %>%
    #     arrange(Time) %>% slice(1) %>%
    #     select(Time) %>% filter(experiment=='shortmip')
}

# data wrangling ----------------------------------------------------------

if (FALSE){

    exps <- c('prise_srv3_20200528', 'port_peschiera_20200529', 
              'prise_srv3_20200602', 'prise_srv3_20200603_good', 'prise_srv3_20200603_2', 'prise_srv3_20200604', 
              'prise_srv3_20200604_2', 'prise_srv3_20200605_good')
    exp_names <- list('graph', 'cplex', 'graph2', 'graph3', 'cplex2', 'graph4', 'graph_bigmip', 'graph_bigmip_2')
    
    exps <- c('port_peschiera_20200605', 'prise_srv3_20200605_good')
    exp_names <- list('cplex2', 'graph_bigmip_2')
    
    exps <- c('serv_cluster1_20200609_4', 'serv_cluster1_20200610')
    exp_names <- c('graph_bigmip_merged', 'cplex')
    
    # only initial and then cplex
    exps <- c('serv_cluster1_20200610', 'serv_cluster1_20200618')
    exp_names <- c('cplex', 'short+mip')
    
    # cbc
    exps <- c('serv_cluster1_20200609_4', 'serv_cluster1_20200617', 'serv_cluster1_20200617_2', 'serv_cluster1_20200610', 'serv_cluster1_20200617_3')
    exp_names <- c('graph_1', 'graph_2', 'graph_3', 'cplex', 'graph_cbc')
    

    solver <- list(prise_srv3_20200605_good='HEUR', 
                   serv_cluster1_20200609_good='HEUR', 
                   serv_cluster1_20200609_4='HEUR',
                   serv_cluster1_20200616='HEUR',
                   serv_cluster1_20200617='HEUR',
                   serv_cluster1_20200617_2='HEUR', 
                   serv_cluster1_20200617_3='HEUR',
                   serv_cluster1_20200618='HEUR',
                   serv_cluster1_20200622='HEUR')
    progress <- get_progress(exps, exp_names, solver=solver)
    progress_n <- progress %>% filter(Time>200)
    draw_progress(progress_n %>% filter(scenario==120), log_scale_y = TRUE)
    data_nn <- get_summary_table(exps, exp_names)
    data_nn %>% 
        kable(format = 'latex', booktabs = TRUE)
    
    # large
    exps <- c('serv_cluster1_20200616', 'serv_cluster1_20200622', 'serv_cluster1_20200615')
    exp_names <- c('graph', 'graph50', 'cplex')
    
    # only last large
    compare_large()
    
    # ~200 aircraft

    
    
    # compare initial solution 
    compare_initial_solution()
    
    # compare_normal
    compare_normal()
    
}


source('nps_reports/functions.R')
source('graph_reports/datasets.R')

# stop()

get_summary_table <- function(exps, exp_names, wider=TRUE){
    data <- get_generic_compare(exps, exp_names = exp_names)
    data_nn <- data %>% 
        mutate(
            num_aircraft = scenario %>% str_extract('\\d') %>% as.integer %>% multiply_by(15),
            objective = objective %>% round
               ) %>% 
        select(instance, experiment, num_aircraft, objective) %>% 
        filter(instance %>% is.na %>% not) %>% 
        arrange(experiment, num_aircraft, instance)
    
    if (!wider){
        return(data_nn)
    }
    data_nn %>% 
        pivot_wider(id_cols = c(instance, num_aircraft), 
                    names_from = experiment, 
                    values_from = objective) %>% 
        arrange(num_aircraft, instance)
}

# print(n=40)


# PROGRESS:
# stop()

get_progress <- function(exps, exp_names, solver){
    progress <- get_generic_compare(exps, exp_names = exp_names, get_progress=TRUE, solver=solver)
    
    progress %>% 
        mutate(num_aircraft = scenario %>% str_extract('\\d') %>% as.integer %>% multiply_by(15)) %>% 
        select(experiment, num_aircraft, instance, Time, BestInteger) %>% 
        mutate(BestInteger=as.numeric(BestInteger),
               Time = as.numeric(Time)) %>% 
        filter(!is.null(BestInteger))
}

draw_progress <- function(progress, num_aircraft_filt, count_preprocess=TRUE, log_scale_y=TRUE){

    data_graph <- 
        progress %>% 
        filter(num_aircraft==num_aircraft_filt) %>% 
        mutate(row = sprintf("f=%s i=%s", num_aircraft, instance))
    
    # "cheating":
    if (!count_preprocess){
        data_graph <- data_graph %>% group_by(experiment, instance) %>% mutate(Time=Time - min(Time))    
    }
    
    filter_df <- data_graph %>% distinct(experiment, row) %>% group_by(row) %>% filter(., n()>1)
    
    data_graph <- data_graph %>% inner_join(filter_df)
    
    p = ggplot(data_graph, aes(x=Time, y=BestInteger, color=experiment)) + 
        geom_point() + 
        facet_grid(rows='row', scales="free_y")
    if (log_scale_y){
        return(p + scale_y_log10())
    }
    return(p)
}

compare_initial <- function(exps, exp_names, solver=solver){
    data_nn <- get_summary_table(exps, exp_names, wider=FALSE)
    data_best <- data_nn %>% group_by(num_aircraft, instance) %>% summarise(best=min(objective))
    progress <- get_progress(exps, exp_names, solver=solver)
    
    progress %>% 
        arrange(experiment, num_aircraft, instance, Time) %>% 
        group_by(experiment, num_aircraft, instance) %>% 
        summarise(initial=first(BestInteger)) %>% 
        inner_join(data_best) %>% 
        mutate(initial_p = (initial-best)/best *100) %>% 
        mutate(initial_p = initial_p %>% round) %>% 
        pivot_wider(id_cols = c(instance, num_aircraft), 
                    names_from = experiment, values_from = initial_p)
}

if (FALSE){
    exps <- c('prise_srv3_20200528', 'port_peschiera_20200529', 
              'prise_srv3_20200602', 'prise_srv3_20200603_good', 'prise_srv3_20200603_2', 'prise_srv3_20200604', 
              'prise_srv3_20200604_2', 'prise_srv3_20200605_good')
    exp_names <- list('graph', 'cplex', 'graph2', 'graph3', 'cplex2', 'graph4', 'graph_bigmip', 'graph_bigmip_2')
    
    exps <- c('port_peschiera_20200605', 'prise_srv3_20200605_good')
    exp_names <- list('cplex2', 'graph_bigmip_2')
    
    exps <- c('serv_cluster1_20200609_4', 'serv_cluster1_20200610')
    exp_names <- c('graph_bigmip_merged', 'cplex')
    
    exps <- c('serv_cluster1_20200616', 'serv_cluster1_20200615')
    exp_names <- c('graph', 'cplex')
    
    # cbc
    exps <- c('serv_cluster1_20200609_4', 'serv_cluster1_20200617', 'serv_cluster1_20200617_2', 'serv_cluster1_20200610', 'serv_cluster1_20200617_3')
    exp_names <- c('graph_1', 'graph_2', 'graph_3', 'cplex', 'graph_cbc')
    
    # initial only
    exps <- c('serv_cluster1_20200609_4', 'serv_cluster1_20200617_2', 'serv_cluster1_20200610', 'serv_cluster1_20200618')
    exp_names <- c('graph_1', 'graph_3', 'cplex', 'graph_init')
    
    
    solver <- list(prise_srv3_20200605_good='HEUR', 
                   serv_cluster1_20200609_good='HEUR', 
                   serv_cluster1_20200609_4='HEUR',
                   serv_cluster1_20200616='HEUR',
                   serv_cluster1_20200617='HEUR',
                   serv_cluster1_20200617_2='HEUR', 
                   serv_cluster1_20200617_3='HEUR',
                   serv_cluster1_20200618='HEUR')
    progress <- get_progress(exps, exp_names, solver=solver)
    draw_progress(progress, num_aircraft_filt=105, log_scale_y = TRUE)
    data_nn <- get_summary_table(exps, exp_names)
    data_nn %>% 
        kable(format = 'latex', booktabs = TRUE)
    
}


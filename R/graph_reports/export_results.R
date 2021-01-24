source('nps_reports/functions.R')
source('graph_reports/datasets.R')
source('graph_reports/functions.R')
library(kableExtra)
# library(gginnards)


# header ------------------------------------------------------------------

opts <- options(knitr.kable.NA = "")
path_export_img <- '../../Graph2020/'
path_export_tab <- '../../Graph2020/tables/'
element_text_size <- 10


# exports -----------------------------------------------------------------

get_mip_vnd_bb_table <- function(summary, progress, apply_dif=TRUE){
    if (apply_dif){
        difs <-
            progress %>%
            group_by(experiment, scenario, instance) %>%
            arrange(Time) %>%
            slice(n()) %>%
            select(experiment, scenario, instance, last=BestInteger) %>%
            inner_join(summary) %>%
            mutate(dif = last-objective) %>%
            select(instance, experiment, scenario, dif)
    } else {
        difs <- summary %>% distinct(instance, experiment, scenario) %>% mutate(dif=0)
    }
    
    # 
    bounds <-
        summary %>% distinct(instance, scenario, best_bound) %>%
        inner_join(difs %>% ungroup %>% filter(experiment=='MIP') %>% select(-experiment)) %>%
        mutate(best_bound=best_bound+dif) %>% select(-dif)
    
    summary %>% 
        summary_to_wider(column='objective') %>%
        inner_join(bounds) %>% 
        mutate(best_bound = round(best_bound)) %>% 
        mutate(`$\\frac{VND-MIP}{VND}$ (\\%)`= ((`VND`-MIP)/VND*100) %>% round(2)) %>%
        mutate(`$\\frac{VND-BB}{VND}$ (\\%)`= ((`VND`-best_bound)/VND*100) %>% round(2)) %>% 
        rename(BB=best_bound)
}

compare_large <- function(){
    exps <- c( 'serv_cluster1_20200623', 'serv_cluster1_20200615')
    exp_names <- c('VND', 'MIP')
    scenario_filter <- c(6, 7, 8) %>% paste0('numparalleltasks_', .)
    progress <- get_progress(exps, exp_names, solver=list(serv_cluster1_20200623='HEUR', serv_cluster1_20200928='HEUR'), scenario_filter=scenario_filter)
    
    path <- '%slarge_datasets_compare.tex' %>% sprintf(path_export_tab)
    data <- get_summary_table(exps, exp_names, scenario_filter=scenario_filter, wider=FALSE)
    # 
    # we have to make some legal black magic to be sure that the objectives 
    # in the progress are being measured corretly in all cases.
    table_large <- get_mip_vnd_bb_table(data, progress)
    table_large %>% 
        formated_kable(escape = FALSE) %>% write_file(path)
    # data_graph %>% head
    # data_nn <- data %>% mutate(`dif (%)`= ((`VND`-MIP)/MIP*100) %>% round(2)) %>% formated_kable
    # data_nn %>% write_file(path)
    
}

compare_initial_solution <- function(){
    # exps <- c('serv_cluster1_20200701', 'serv_cluster1_20200701')
    # exp_names <- c('initial', 'NONE')
    exps <- c('serv_cluster1_20200701', 'port_peschiera_20200715')
    exp_names <- c('initial', 'bounds')
    
    data <- get_generic_compare(exps, exp_names = exp_names, solver=list(serv_cluster1_20200701='HEUR'))
    bbound <- data %>% filter(experiment=='bounds') %>% select(instance, best_solution) %>% mutate_at(vars(best_solution), unlist)
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
        arrange(scenario, instance) %>% 
        mutate(scenario= if_else(scenario=='MaintFirst', 'maintFirst', scenario)) %>% 
        inner_join(bbound) %>% 
        mutate(gap = (objective-best_solution)/best_solution*100,
               gap = round(gap)) %>% 
        pivot_wider(id_cols = instance, 
                    names_from = scenario, 
                    values_from = gap) %>% 
        arrange(instance)
    
    path <- '%sinitial_solution_compare.tex' %>% sprintf(path_export_tab)
    data_n %>% 
        rename(`maintFirst (\\%)`=maintFirst, `RH (\\%)`=mip, `SPA (\\%)`=short) %>% 
        kable(escape=FALSE, format = 'latex', booktabs = TRUE, linesep='') %>% write_file(path)
    
    
    # data <- compare_initial(exps, exp_names, solver=list(serv_cluster1_20200617_2='HEUR'))
    # data_n <- data %>% mutate(cplex= sprintf('%s %%', cplex), `VND`=sprintf('%s %%', `VND`)) %>% formated_kable
    # data_n %>% write_file(path)
}

compare_normal <- function(){
    exps <- c('serv_cluster1_20200610', 'serv_cluster1_20200623')
    exp_names <- c('MIP', 'VND')
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
    data_nn <- data %>% mutate(`dif (%)`= ((`VND`-MIP)/MIP*100) %>% round(2)) %>% formated_kable
    data_nn %>% write_file(path)
}

compare_200aircraft <- function(){
    exps <- c('serv_cluster1_20200625', 'serv_cluster1_20200702')
    exp_names <- c('MIP', 'VND')
    data_200 <- get_summary_table(exps, exp_names, wider=FALSE)
    progress_200 <- get_progress(exps, exp_names, solver=list(serv_cluster1_20200702='HEUR'))
    table_large <- get_mip_vnd_bb_table(data_200, progress_200, apply_dif = FALSE)
    path <- '%sgaps200.tex' %>% sprintf(path_export_tab)
    table_large %>% formated_kable(escape = FALSE) %>% write_file(path)
    table_large[[7]] %>% mean()
    # gaps <- 
    #     data_200 %>% 
    #     mutate(gap=(objective-best_bound)/objective*100,
    #            gap = gap %>% round(2)) %>% 
    #     summary_to_wider(column='gap')
    # 
    
    # path <- '%sgaps200.tex' %>% sprintf(path_export_tab)
    # gaps %>% rename(`VND (\\%)`=VND, `MIP (\\%)`=MIP) %>%
    #     formated_kable(escape=FALSE) %>% write_file(path)
    # 
    # gaps %>% ungroup %>% summarise_at(vars(VND, MIP), mean)
    # summary_to_wider(data_200, column='objective')
    # errors <- summary_to_wider(data_200, column='errors')
    
    # progress graph
    path <- '%sprogress255.png' %>% sprintf(path_export_img)
    progress_200 %>% filter(Time>100) %>% filter(scenario==255) %>% 
        draw_progress(log_scale_y = TRUE) + 
        ylab('Objective value') + theme_minimal() + theme(text = element_text(size=15)) +
        ggsave(path)
    
    # progress graph with gaps with respect to best bound
    bounds <- data_200 %>% distinct(instance, scenario, best_bound)
    
    data_graph <- 
        progress_200 %>% 
        inner_join(bounds) %>% 
        mutate(BestInteger = (BestInteger-best_bound)/BestInteger*100)
    
    path <- '%sprogress_gaps_very_large.png' %>% sprintf(path_export_img)
    data_graph %>% 
        mutate(instance= as.factor(instance)) %>% 
        ggplot(aes(x=Time, y=BestInteger, colour=instance, linetype=experiment)) + 
        geom_step(size=1.5) + 
        facet_grid(rows='scenario', scales="free_y") +
        labs(linetype = "Method", color = "Instance") + 
        ylab("Percentage gap") + theme_minimal() +
        theme(text = element_text(size=23)) +
        ggsave(path, width = 16, height = 9)
    
    path <- '%sprogress_gaps_very_large_255.png' %>% sprintf(path_export_img)
    data_graph %>% filter(scenario==255) %>% 
        mutate(instance= as.factor(instance)) %>% 
        ggplot(aes(x=Time, y=BestInteger, colour=instance, linetype=experiment)) + 
        geom_step(size=1.5) + 
        # facet_grid(rows='scenario', scales="free_y") +
        labs(linetype = "Method", color = "Instance") + 
        ylab("Percentage gap") + theme_minimal() +
        theme(text = element_text(size=23)) +
        ggsave(path, width = 16, height = 9)
}

compare_neighbors <- function(){
    
    exps <- c('serv_cluster1_20200630_1', 'serv_cluster1_20200630_2', 'serv_cluster1_20200630_3')
    exp_names <- c('SPA', 'RH', 'VND')
    progress <- get_generic_compare(exps, exp_names = exp_names, get_progress=TRUE, 
                                    solver=to_list_value(exps, 'HEUR'))
    
    progress_n <- 
        progress %>% 
        mutate(BestInteger=as.numeric(BestInteger),
               Time = as.numeric(Time)) %>% 
        filter(!is.null(BestInteger))
    
    path <- '%scompare_neighbors.png' %>% sprintf(path_export_img)
    progress_nn <- 
        progress_n %>% 
        filter(instance==81) %>% 
        mutate(col = scenario %>% str_extract('\\d+'))
    
    # labeller_ <- progress_nn$col %>% unique() %>% sapply(function(x) "") %>% labeller(col=.)
    progress_nn %>% 
        ggplot(aes(x=Time, y=BestInteger, group=experiment)) + 
        geom_line(aes(linetype=experiment, color=experiment), size=1) +
        scale_y_log10() +
        facet_grid(cols=vars(col)) +
        theme_minimal() +  labs(color = "Method", linetype="Method") + 
        theme(text = element_text(size=element_text_size)) +
        theme(legend.position = "bottom",
              legend.box = "vertical",
              strip.text = element_blank()) +
        ylab("Objective value") +
        theme(text = element_text(size=23)) +
        ggsave(path, width = 16, height = 9)
    
    
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
    all_points <- 
        before_local_minim %>% 
        arrange(experiment, scenario, instance, Time) %>% 
        group_by(experiment, scenario, instance) %>% 
        mutate(Iterations = n()) %>% 
        slice(n()) %>% 
        group_by(instance) %>% 
        mutate(BestBestInteger=min(BestInteger),
               BestInteger= BestInteger/BestBestInteger) %>% 
        ungroup %>% 
        select(experiment, scenario, instance, Time, BestInteger, Iterations)
    
    # facetd boxpots ?
    boxplot_neighbors <- function(data, y_column){
        data %>% 
            mutate(instance=as.factor(instance)) %>% 
            ggplot(aes(y=data[[y_column]], x=experiment)) + 
            geom_boxplot(size=1) +
            theme_minimal() +
            theme(text = element_text(size=10))
    }
    path <- '%scompare_neighbors_boxplot_objective.png' %>% sprintf(path_export_img) 
    boxplot_neighbors(all_points, 'BestInteger') + scale_y_log10() + 
        ylab("Objective value (normalized)") + xlab('Instance') +
        theme(text = element_text(size=35)) +
        ggsave(path, width = 16, height = 9)
    
    path <- '%scompare_neighbors_boxplot_time.png' %>% sprintf(path_export_img) 
    boxplot_neighbors(all_points, 'Time') + 
        ylab("Time (seconds)") + xlab('Instance') +
        theme(text = element_text(size=35)) +
        ggsave(path, width = 16, height = 9)
}

compare_not_so_large_3600 <- function(){
    exps <- c('serv_cluster1_20200926', 'serv_cluster1_20200926_2')
    exp_names <- c('MIP', 'VND')
    # scenario = 60
    progress <- get_progress(exps, exp_names, solver=list(serv_cluster1_20200926_2='HEUR')) %>% mutate(scenario=60)
    
    last <- progress %>% 
        group_by(experiment, scenario, instance) %>% 
        arrange(Time) %>% 
        slice(n()) %>% 
        select(experiment, scenario, last=BestInteger)
    
    data_nn <- get_summary_table(exps, exp_names, wider=FALSE) %>% mutate(scenario=60)
    
    # we have to make some legal black magic to be sure that the objectives 
    # in the progress are being measured corretly in all cases.
    difs <- data_nn %>% inner_join(last) %>% mutate(dif = last-objective) %>% 
        select(instance, experiment, scenario, dif)
    
    bounds <- 
        data_nn %>% distinct(instance, scenario, best_bound) %>% 
        inner_join(difs %>% filter(experiment=='MIP') %>% select(-experiment)) %>% 
        mutate(best_bound=best_bound-dif) %>% select(-dif)
    
    data_graph <- 
        progress %>% inner_join(bounds) %>% inner_join(difs) %>% 
        mutate(BestInteger = (BestInteger-dif-best_bound)/(BestInteger-dif)*100)
    
    path <- '%sprogress_gaps_not_very_large.png' %>% sprintf(path_export_img)
    data_graph %>% 
        mutate(instance= as.factor(instance)) %>% 
        ggplot(aes(x=Time/60, y=BestInteger, group=experiment)) + 
        geom_step(aes(colour=experiment, linetype=experiment), size=1.5) + 
        facet_grid(cols=vars(instance)) +
        labs(color = "Method", linetype="Method") + 
        ylab("Percentage gap") + theme_minimal() +
        xlab("Time (minutes)") +
        theme(text = element_text(size=23)) +
        theme(
            legend.position = "bottom",
            legend.box = "vertical") +
        scale_x_continuous(breaks=scales::pretty_breaks(n = 2)) +
        ggsave(path, width = 16, height = 9)
}

# data wrangling ----------------------------------------------------------

if (FALSE){
    compare_200aircraft()
    compare_large()
    compare_neighbors()
    compare_initial_solution()
    compare_not_so_large_3600()
    
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
    exp_names <- c('cplex', 'VND')
    
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
    

}


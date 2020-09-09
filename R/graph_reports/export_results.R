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

compare_large <- function(){
    exps <- c( 'serv_cluster1_20200623', 'serv_cluster1_20200615')
    exp_names <- c('VND', 'MIP')
    scenario_filter <- c(6, 7, 8) %>% paste0('numparalleltasks_', .)
    progress <- get_progress(exps, exp_names, solver=list(serv_cluster1_20200623='HEUR'), scenario_filter=scenario_filter)
    progress_n <- progress %>% filter(Time>120)
    path <- '%slarge_datasets_progress_120.png' %>% sprintf(path_export_img)
    y_lab <- 'Best solution found'
    draw_progress(progress_n %>% filter(scenario==120), log_scale_y = TRUE, count_preprocess = FALSE) + 
        ylab(y_lab) + theme_minimal() + theme(text = element_text(size=15)) +
        ggsave(path)
    
    path <- '%slarge_datasets_compare.tex' %>% sprintf(path_export_tab)
    data <- get_summary_table(exps, exp_names, scenario_filter=scenario_filter)
    data_nn <- data %>% mutate(`dif (%)`= ((`VND`-MIP)/MIP*100) %>% round(2)) %>% formated_kable
    data_nn %>% write_file(path)
    
}

compare_initial_solution <- function(){
    # exps <- c('serv_cluster1_20200701', 'serv_cluster1_20200701')
    # exp_names <- c('initial', 'NONE')
    exps <- c('serv_cluster1_20200701', 'port_peschiera_202000715')
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
    exp_names <- c('cplex', 'graph_cplex')
    data_nn <- get_summary_table(exps, exp_names, wider=FALSE)
    gaps <- 
        data_nn %>% 
        mutate(gap=(objective-best_bound)/objective*100,
               gap = gap %>% round(2)) %>% 
        summary_to_wider(column='gap')
    path <- '%sgaps200.tex' %>% sprintf(path_export_tab)
    gaps %>% rename(`VND (\\%)`=graph_cplex, `MIP (\\%)`=cplex) %>%
        formated_kable(escape=FALSE) %>% write_file(path)
    gaps %>% ungroup %>% summarise_at(vars(graph_cplex, cplex), mean)
    summary_to_wider(data_nn, column='objective')
    errors <- summary_to_wider(data_nn, column='errors')

    
    progress <- get_progress(exps, exp_names, solver=list(serv_cluster1_20200625_3='HEUR', 
                                                          serv_cluster1_20200701_2='HEUR',
                                                          serv_cluster1_20200702='HEUR'))
    path <- '%sprogress255.png' %>% sprintf(path_export_img)
    equiv <- data.frame(experiment=exp_names, experiment2=c('MIP', 'VND'))
    progress %>% filter(Time>100) %>% filter(scenario==255) %>% 
        inner_join(equiv) %>% mutate(experiment=experiment2) %>% 
        draw_progress(log_scale_y = TRUE) + 
            ylab('Objective value') + theme_minimal() + theme(text = element_text(size=15)) +
            ggsave(path)
    
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
        geom_line(aes(linetype=experiment, color=experiment)) +
        scale_y_log10() +
        facet_grid(cols=vars(col)) +
        theme_minimal() +  labs(color = "Method", linetype="Method") + 
        theme(text = element_text(size=element_text_size)) +
        theme(legend.position = "bottom",
              legend.box = "vertical",
              strip.text = element_blank()) +
        ylab("Objective value") +
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
    boxplot_neighbors <- function(data, y_column, path){
        data %>% 
            mutate(instance=as.factor(instance)) %>% 
            ggplot(aes(x=instance, y=data[[y_column]], group=experiment)) + 
            geom_jitter(aes(color=experiment, shape=experiment), width=0.4, height=0, size=2) + 
            theme_minimal() +
            theme(text = element_text(size=10)) + 
            labs(color = "Method", shape="Method")
    }
    path <- '%scompare_neighbors_boxplot_objective.png' %>% sprintf(path_export_img) 
    boxplot_neighbors(all_points, 'BestInteger', path) + scale_y_log10() + 
        ylab("Objective value (normalized)") + 
        ggsave(path)

    path <- '%scompare_neighbors_boxplot_time.png' %>% sprintf(path_export_img) 
    boxplot_neighbors(all_points, 'Time', path) + 
        ylab("Time (seconds)") + 
        ggsave(path)

    # base_table <- 
    #     all_points %>% 
    #     group_by(experiment, instance) %>% 
    #     summarise_at(vars(Time, BestInteger), list(min=min, max=max)) %>% 
    #     mutate(Time= sprintf("%s-%s", Time_min, Time_max),
    #            BestInteger= sprintf("%s-%s", BestInteger_min, BestInteger_max))
    # 
    # # time comparison
    # path <- '%scompare_neighbors_time.tex' %>% sprintf(path_export_tab)
    # base_table %>% 
    #     select(experiment, instance, Time) %>% 
    #     pivot_wider(id_cols=instance, names_from=experiment, values_from=Time) %>% 
    #     rename(`$t_{RH}$`=RH, `$t_{SPA}$`=SPA, `$t_{both}$`=VND) %>% 
    #     kable(format = 'latex', booktabs = TRUE, linesep='', escape = FALSE) %>%
    #     write_file(path)
    # 
    # # quality comparison
    # path <- '%scompare_neighbors_obj.tex' %>% sprintf(path_export_tab)
    # base_table %>% 
    #     select(experiment, instance, BestInteger) %>% 
    #     pivot_wider(id_cols=instance, names_from=experiment, values_from=BestInteger) %>% 
    #     rename(`$obj_{RH}$`=RH, `$obj_{SPA}$`=SPA, `$obj_{both}$`=VND) %>% 
    #     kable(format = 'latex', booktabs = TRUE, linesep='', escape = FALSE) %>%
    #     write_file(path)

}

# data wrangling ----------------------------------------------------------

if (FALSE){
    compare_200aircraft()
    compare_large()
    compare_neighbors()
    compare_initial_solution()

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


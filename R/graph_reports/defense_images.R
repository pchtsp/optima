source('nps_reports/functions.R')
source('graph_reports/datasets.R')
source('graph_reports/functions.R')
source('graph_reports/export_results.R')

path_export_img <- '../../phd_defense/images/'
element_text_size <- 23
if (FALSE){

# 255 aircraft ------------------------------------------------------------

    
    exps <- c('serv_cluster1_20200625', 'serv_cluster1_20200702')
    exp_names <- c('MIP', 'VND')
    progress_200 <- get_progress(exps, exp_names, solver=list(serv_cluster1_20200702='HEUR'))
    
    # progress graph with gaps with respect to best bound
    data_200 <- get_summary_table(exps, exp_names, wider=FALSE)
    bounds <- data_200 %>% distinct(instance, scenario, best_bound)
    
    data_graph <- 
        progress_200 %>% 
        filter(scenario==255) %>% 
        inner_join(bounds) %>% 
        mutate(BestInteger = (BestInteger-best_bound)/BestInteger*100)
        
    path <- '%sprogress_gaps_very_large_255.png' %>% sprintf(path_export_img)
    data_graph %>% distinct(instance) %>% mutate(instance2=1:n()) %>% inner_join(data_graph) %>% mutate(instance=instance2) %>% 
        mutate(instance= as.factor(instance)) %>% 
        ggplot(aes(x=Time, y=BestInteger, colour=instance, linetype=experiment)) + 
        geom_step(size=1.5) + 
        # facet_grid(rows='scenario', scales="free_y") +
        labs(linetype = "Method", color = "Instance") + 
        ylab("Percentage gap") + theme_minimal() +
        xlab('Time (seconds)') +
        theme(text = element_text(size=element_text_size)) +
        ggsave(path, width = 16, height = 9)


# gaps large instances ----------------------------------------------------
    # data_200 %>% group_by(experiment, scenario) %>% summarise()
    # 
    # data_200_wider <- get_summary_table(exps, exp_names, wider=TRUE)
    # data_200_wider %>% mutate(dif = (MIP-VND)/MIP*100) %>% summarise(dif=mean(dif))
    # data_200_wider %>% inner_join(bounds) %>% 
    #     mutate(gapMIP=(MIP-best_bound)/MIP*100,
    #            gapVND=(VND-best_bound)/VND*100)
    
    data_200 %>% 
        summary_to_wider(column='objective') %>%
        inner_join(bounds) %>% 
        mutate(gapVND= ((VND-best_bound)/VND*100) %>% round(2)) %>%
        mutate(gapMIP= ((MIP-best_bound)/MIP*100) %>% round(2)) %>% 
        rename(BB=best_bound) %>% summarise_at(vars(gapMIP, gapVND), mean)
    
    print((67.1-23.1)/67.1*100)
    
        
    # data_200

# neighbors ---------------------------------------------------------------

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
    
    progress_nn %>% distinct(col) %>% mutate(col2=1:n()) %>% inner_join(progress_nn) %>% mutate(col=col2) %>% 
        ggplot(aes(x=Time, y=BestInteger, group=experiment)) + 
        geom_step(aes(color=experiment), size=1.5) +
        scale_y_log10() +
        facet_grid(cols=vars(col)) +
        theme_minimal() +  labs(color = "Method", linetype="Method") + 
        theme(text = element_text(size=element_text_size)) +
        # theme(legend.position = "bottom",
        #       legend.box = "vertical",
        #       strip.text = element_blank()) +
        ylab("Objective value") +
        xlab("Time (seconds)") +
        ggsave(path, width = 16, height = 9)
}
source('nps_reports/functions.R')
source('nps_reports/datasets.R')
source('nps_reports/export_results.R')

path_export_img <- '../../phd_defense/images/'


if(FALSE){
    left_tail <- 0.05
    right_tail <- 0.05
    legend_size <- 3

    data_optimisation_results <- get_data_optimisation_results()

    get_stoch_a2r_data <- data_optimisation_results$get_stoch_a2r_data
    raw_df_progress <- data_optimisation_results$raw_df_progress

# quality degradation -----------------------------------------------------
    
    path <- '%squality_degradation_2tasks.png' %>% sprintf(path_export_img)
    ylab <- '% difference in objective'
    
    quality_degr_all <- get_quality_degr_2(get_stoch_a2r_data)
    quality_degr_all_filt <- 
        quality_degr_all %>% 
        filter(experiment=='base_a2r') %>% 
        filter(dif_perc %>% value_filt_tails(c(0, right_tail)))
    
    ggplot(data=quality_degr_all_filt, aes(x=dif_perc)) + 
        theme_minimal() + geom_histogram(binwidth = 0.3) + xlab(ylab) + ylab('Number of instances') + 
        theme(text = element_text(size=element_text_size)) +  
        ggsave(path)

# performance -------------------------------------------------------------
    data <- get_time_perf_integer_reorder(get_stoch_a2r_data)
    path <- '%stime_performance_ordered_2tasks.png' %>% sprintf(path_export_img)
    p <- data %>% 
        inner_join(
            data.table(
            experiment=c('base', 'base_a2r'), 
            experiment2=c('original', 'learnedCuts'))) %>%
        ungroup() %>% 
        mutate(experiment=experiment2) %>% 
        graph_performance(path) 
    p$layers[[1]] <- NULL
    p + theme_minimal() + scale_colour_brewer(palette='Set1') + geom_path(aes(color=experiment, shape=experiment), size=0.7) + ggsave(path)
    
# transitions -------------------------------------------------------------

    clean_instances <- raw_df_progress %>% filter_all_exps %>% distinct(instance)
    transitions <- get_stoch_a2r_data %>% inner_join(clean_instances) %>% get_transitions_stats
    path <- '%stransitions_base_2tasks.png' %>% sprintf(path_export_img)
    transitions %>% 
        filter(experiment=='base_a2r') %>% 
        rename(original=prev_status, learnedCuts=post_status) %>% 
        graph_parent(path=path)
    
}

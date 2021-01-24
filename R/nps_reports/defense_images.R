source('nps_reports/functions.R')
source('nps_reports/datasets.R')
source('nps_reports/export_results.R')

path_export_img <- '../../phd_defense/images/'


if(FALSE){
    left_tail <- 0.05
    right_tail <- 0.05
    legend_size <- 3
    element_text_size <- 23

    data_optimisation_results <- get_data_optimisation_results()

    get_stoch_a2r_data <- data_optimisation_results$get_stoch_a2r_data
    raw_df_progress <- data_optimisation_results$raw_df_progress
    df_fixed <- data_optimisation_results$df_fixed
    
# quality degradation -----------------------------------------------------
    
    path <- '%squality_degradation_2tasks.png' %>% sprintf(path_export_img)
    ylab <- 'Difference in objective function value (%)'
    
    get_stoch_a2r_data_base_a2r <- get_stoch_a2r_data %>% inner_join(data.table(experiment=c('base', 'base_a2r')))
    
    quality_degr_all <- get_quality_degr_2(get_stoch_a2r_data_base_a2r)
    quality_degr_all_filt <- 
        quality_degr_all %>% 
        filter(experiment=='base_a2r') %>% 
        filter(dif_perc %>% value_filt_tails(c(0, right_tail)))
    
    ggplot(data=quality_degr_all_filt, aes(x=dif_perc)) + 
        theme_minimal() + geom_histogram(binwidth = 0.3) + xlab(ylab) + ylab('Number of instances') + 
        theme(text = element_text(size=element_text_size)) +  
        ggsave(path)
    
    # this I say in text
    get_stoch_a2r_data_base_a2r %>% get_dif_times_optimal
    print((1274-228)/1274*100)
    
    # this I say in the text
    get_stoch_a2r_data_base_a2r %>% group_by(experiment) %>% 
        summarise_at(vars(gap, time), mean, na.rm=TRUE)
    # better gaps
    print((20.8-13.4)/20.8*100)
    # better times
    print((2801-1974)/2801*100)

# performance -------------------------------------------------------------
    data <- get_time_perf_integer_reorder(get_stoch_a2r_data_base_a2r)
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
    p + theme_minimal() + scale_colour_brewer(palette='Set1') + 
        geom_path(aes(color=experiment, shape=experiment), size=0.7) + 
        ylab("Time to solve instance (seconds)") +
        theme(text = element_text(size=15)) +  
        ggsave(path)
    
# transitions -------------------------------------------------------------

    clean_instances <- raw_df_progress %>% filter_all_exps %>% distinct(instance)
    transitions <- get_stoch_a2r_data %>% inner_join(clean_instances) %>% get_transitions_stats
    path <- '%stransitions_base_2tasks.png' %>% sprintf(path_export_img)
    transitions %>% 
        filter(experiment=='base_a2r') %>% 
        rename(original=prev_status, learnedCuts=post_status) %>% 
        graph_parent(path=path)
    
}

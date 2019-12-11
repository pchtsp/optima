source('nps_reports/functions.R')
source('nps_reports/datasets.R')

path_export_img <- '../../NPS2019/img/'
path_export_tab <- '../../NPS2019/tab/'
# size <- 'small'
# df_original <- compare_sto$get_instances[[size]]()

# optimization results ----------------------------------------------------

get_data_optimisation_results <- function(){
    df_fixed <- get_all_fixLP()
    raw_df_progress <- get_generic_compare(dataset_list = c(get_base(2), 'IT000125_20191030', 'IT000125_20191017'),
                                           exp_names = list('base', 'old', 'base_determ'),
                                           scenario_filter='numparalleltasks_%s' %>% sprintf(2),
                                           get_progress = TRUE) %>%
        correct_old_model(get_progress = TRUE, keep_correction = TRUE)
    get_stoch_a2r_data <- get_stoch_a2r(2)
    list('df_fixed'=df_fixed, 'raw_df_progress'=raw_df_progress, 'get_stoch_a2r_data'=get_stoch_a2r_data)
}

make_optimisation_results <- function(df_fixed, raw_df_progress, get_stoch_a2r_data, element_text_size=10){
    
    left_tail <- 0.05
    right_tail <- 0.05
    legend_size <- 3

    # compare models
    path <- '%scomparison_models_status_task2.tex' %>% sprintf(path_export_tab)
    get_summary(raw_df_progress) %>% 
        ungroup %>% 
        select(-scenario) %>%
        kable(format='latex', booktabs = TRUE, linesep="") %>% 
        write_file(path)

    path <- '%scomparison_models_optim_task2.tex' %>% sprintf(path_export_tab)
    get_stats_summary(raw_df_progress) %>% 
        ungroup %>% 
        select(-scenario) %>%
        kable(format='latex', booktabs = TRUE, linesep="") %>% 
        write_file(path)

    # summary
    summary_stats <- get_summary(get_stoch_a2r_data)
    path <- '%sstats_2tasks.tex' %>% sprintf(path_export_tab)
    summary_stats %>% 
        rename(Status=Indicator) %>% 
        ungroup %>% 
        select(-scenario) %>% 
        kable(format='latex', booktabs = TRUE, linesep="") %>% 
        write_file(path)
    
    # summary interception:
    
    summary_int <- get_comparable_sets(get_stoch_a2r_data)
    path <- '%sstats_int_2tasks.tex' %>% sprintf(path_export_tab)
    summary_int %>%
        rename(Status=Indicator) %>% 
        kable(format='latex', booktabs = TRUE, linesep="", escape=FALSE) %>% 
        write_file(path)

    # quality degradation
    path <- '%squality_degradation_2tasks.png' %>% sprintf(path_export_img)
    ylab <- '% difference in objective'
    
    quality_degr_all <- get_quality_degr_2(get_stoch_a2r_data)
    quality_degr_all_filt <- 
        quality_degr_all %>% 
        filter(dif_perc %>% value_filt_tails(c(left_tail, right_tail)))
    
    ggplot(data=quality_degr_all_filt, aes(x=experiment, y=dif_perc)) + 
        theme_minimal() + geom_boxplot() + xlab('Experiment') + ylab(ylab) + 
        theme(text = element_text(size=element_text_size)) + coord_flip() + 
        ggsave(path)
    
    
    # quality performance
    # quality_perf <- get_quality_perf(df)
    # filtered_q_perf <- quality_perf %>% filter(between(dif_perc, -7, 7))
    # filtered_quant <- (1 - (filtered_q_perf %>% nrow)/ (quality_perf %>% nrow))*100
    # path <- '%squality_performance_%s_%s.png' %>% sprintf(path_export_img, exp_list[1], exp_list[2])
    # qplot(filtered_q_perf$dif_perc, xlab='Relative gap (in %) among integer solutions.', binwidth=0.3) +
    #     theme(text = element_text(size=20)) + theme_minimal() + ggsave(path)
    
    # time performance
    graph_performance <- function(data, path){
        ggplot(data=data, aes(x=percentage, y=time, color=experiment)) + 
            theme_minimal() + geom_point(size=0.5) + xlab('Instance percentage') + 
            ylab('Time to solve instance') + 
            theme(text = element_text(size=element_text_size)) + 
            guides(color = guide_legend(override.aes = list(size=legend_size))) + ggsave(path)    
    }
    
    data <- get_time_perf_integer_reorder(get_stoch_a2r_data)
    path <- '%stime_performance_ordered_2tasks.png' %>% sprintf(path_export_img)
    graph_performance(data, path)
    
    data <- get_time_perf_integer_reorder(df_fixed)
    path <- '%stime_performance_ordered_fixLP.png' %>% sprintf(path_export_img)
    graph_performance(data, path)

    # infeasible and soft constraints
    infeasible_stats <- get_infeasible_stats(get_stoch_a2r_data)
    errors_stats <- get_soft_constraints(get_stoch_a2r_data)
    path <- '%sinfeas_2tasks.tex' %>% sprintf(path_export_tab)
    
    infeasible_stats %>% 
        filter(!(Indicator %in% c('Total', 'Infeasible'))) %>% 
        mutate(Indicator = '%s_new' %>% sprintf(Indicator)) %>% 
        bind_rows(errors_stats) %>% 
        ungroup %>% select(-scenario) %>% 
        kable(format='latex', booktabs = TRUE, linesep="") %>% 
        write_file(path)
    
    # infeasible: fixLP
    infeasible_stats <- get_infeasible_stats(df_fixed)
    errors_stats <- get_soft_constraints(df_fixed)
    path <- '%sinfeas_fixed_2tasks.tex' %>% sprintf(path_export_tab)
    
    infeasible_stats %>% 
        filter(!(Indicator %in% c('Total', 'Infeasible'))) %>% 
        mutate(Indicator = '%s_new' %>% sprintf(Indicator)) %>% 
        bind_rows(errors_stats) %>% 
        ungroup %>% select(-scenario) %>% 
        kable(format='latex', booktabs = TRUE, linesep="") %>% 
        write_file(path)
    
    # variance
    variances_all <- get_variances(get_stoch_a2r_data)
    path <- '%svariance_2tasks.png' %>% sprintf(path_export_img)
    ylab <- 'Difference in variance (in % of base case)'
    variances_all_filt <- 
        variances_all %>% 
        group_by(experiment) %>%
        filter(dif_perc %>% value_filt_tails(c(0, right_tail)))
    
    ggplot(data=variances_all_filt, aes(x=experiment, y=dif_perc)) + theme_minimal() + 
        geom_boxplot() + xlab('Experiment') + ylab(ylab) + 
        theme(text = element_text(size=element_text_size)) + coord_flip() + ggsave(path)
}

make_optimisation_results(df_fixed, raw_df_progress, get_stoch_a2r_data)

# forecasting results  -------------------------------------------------------
# TODO: finish this function to generate all graphs
make_forecasting_results <- function(result_tab, element_text_size=10){
    
    # this is hardcoded:
    dataset <- 'IT000125_20190716'
    
    # distribution on mean_distances
    path <- '%sdistribution_mean_distances_%s.png' %>% sprintf(path_export_img, dataset)
    ggplot(data=result_tab, aes(x=mean_dist_complete)) + 
        geom_histogram(position="identity", binwidth = 1) + 
        theme_minimal() +
        theme(axis.text.x = element_text(hjust=0),
              strip.text.y = element_text(angle=0),
              text = element_text(size=element_text_size)) +
        xlab('Average distance between checks') + 
        ylab('Number of instances') + 
        ggsave(path)
        
    # maintenances 
    result_tab_n <- result_tab %>% filter(gap_abs<100 & num_errors==0)
    path <- '%smean_consum_vs_maints_nocolor_%s.png' %>% sprintf(path_export_img, dataset)
    ggplot(data=result_tab_n, aes(y=maints, x=mean_consum)) + 
        geom_jitter(alpha=0.4, height=0.2) + theme_minimal() + 
        facet_grid('init_cut ~ .') +
        theme(axis.text.x = element_text(hjust=0),
              strip.text.y = element_text(angle=0),
              text = element_text(size=element_text_size)) +
        xlab('Average consumption in flight hours per period') + 
        ylab('Total number of checks') + 
        ggsave(path)
    
    # quantiles
    # TODO: quantiles

}

make_quantiles <- function(data){
    comp <- get_python_module('scripts', 'compare_stochastic')
    comp$predict_from_table
}

# summary optimization ----------------------------------------------------
get_data_optimisation_summary <- function(){
    get_generic_compare(c('IT000125_20191204', 'IT000125_20190917', 
                          'IT000125_20191030', 'IT000125_20191207',
                          'IT000125_20190808', 'IT000125_20190812', 
                          'IT000125_20191201', 'IT000125_20191130',
                          'IT000125_20191125'),
                        exp_names = list('base', 'base_a2r', 
                                         'old', 'old_a2r',
                                         'base_a1', 'base_a2', 
                                         'old_a3', 'base_a3r',
                                         'old_a1'),
                        scenario_filter='numparalleltasks_%s' %>% sprintf(c(2, 3, 4))) %>%
        correct_old_model
}

make_optimisation_summary <- function(data_summary){
    
    data_per_scenario <- data_summary %>% split(use_series(., scenario))
    treated_data <- data_per_scenario %>% lapply(get_mega_summary)
    col_names <- c("r"="", 
                   'scenario'="$|\\mathcal{I}|$", 
                   'errors_mean'="$E_\\mu$", 
                   'errors_new'="$E_{\\%}$", 
                   'Feasible'="$Feas$",
                   'InfPerc'="$Infeas$",
                   'q_mean'="$Q_\\mu$",
                   'q_medi'="$Q_m$", 
                   'q_q95'="$Q_{95}$", 
                   'time_mean'="$T_\\mu$", 
                   'time_medi'="$T_m$",
                   'v_mean'="$V_\\mu$")
    
    col_names_list <- col_names %>% lapply(FUN=function(x) x)

    # Here we export
    write_func <- function(tab){
        name <- tab$scenario[1]
        col_names_list
        names(tab) <- str_replace(names(tab), '\\_', '\\\\_')
        path <- '%scompare_all_%s.tex' %>% sprintf(path_export_tab, name)
        tab %>% names %>% print
        tab %>% ungroup %>% select(-scenario) %>% 
            mutate(Indicator=col_names_list[Indicator] %>% unlist) %>% 
            rename(Stat=Indicator) %>% 
            kable(format='latex', booktabs = TRUE, linesep="", escape=FALSE) %>% 
            write_file(path)
    }
    treated_data %>% lapply(write_func)
    
}

if (FALSE){
    data_optimisation_results <- get_data_optimisation_results()
    do.call(make_optimisation_results, args=data_optimisation_results)
    
    result_tab <- get_result_tab('IT000125_20190716')
    make_forecasting_results(result_tab, element_text_size=15)
    
    data_summary <- get_data_optimisation_summary()
    make_optimisation_summary(data_summary)
    
}

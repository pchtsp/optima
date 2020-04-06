source('nps_reports/functions.R')
source('nps_reports/datasets.R')



path_export_img <- '../../NPS2019/'
path_export_tab <- '../../NPS2019/'
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
        ggplot(data=data, aes(x=percentage, y=time)) + 
            # theme_minimal() +
            # geom_line(aes(linetype=experiment, color=experiment)) + 
            geom_point(aes(color=experiment, shape=experiment), size=0.7) + xlab('Instance percentage') +
            ylab('Time to solve instance') + 
            scale_colour_brewer(palette='Spectral') +
            theme(text = element_text(size=element_text_size)) + 
            guides(color = guide_legend(override.aes = list(size=legend_size))) +
            labs(shape='Experiment', color='Experiment') +
            ggsave(path)    
    }

    data <- get_time_perf_integer_reorder(get_stoch_a2r_data)
    path <- '%stime_performance_ordered_2tasks.png' %>% sprintf(path_export_img)
    graph_performance(data, path)
    
    data <- get_time_perf_integer_reorder(df_fixed)
    path <- '%stime_performance_ordered_fixLP.png' %>% sprintf(path_export_img)
    graph_performance(data, path)

    # infeasible and soft constraints
    infeasible_stats <- get_infeasible_stats(get_stoch_a2r_data)
    errors_stats <- 
        get_soft_constraints(get_stoch_a2r_data) %>% 
        mutate(Indicator = Indicator %>% str_replace('\\_', '\\\\_'))
    path <- '%sinfeas_2tasks.tex' %>% sprintf(path_export_tab)
    
    infeasible_stats %>% 
        filter(!(Indicator %in% c('Total', 'Infeasible'))) %>% 
        mutate(Indicator = '%s $\\to$ Infeasible' %>% sprintf(Indicator)) %>% 
        bind_rows(errors_stats) %>% 
        ungroup %>% select(-scenario) %>% 
        set_names(., names(.) %>% str_replace('\\_', '\\\\_')) %>% 
        kable(format='latex', booktabs = TRUE, linesep="", escape=FALSE) %>% 
        write_file(path)
    
    # status graph
    transitions <- get_transitions_stats(get_stoch_a2r_data)
    graph_parent <- function(data, path){
        max_num <- 50
        data %>% 
            mutate(label1 = ifelse(num >= max_num, as.character(num), NA)) %>% 
            mutate(label2 = ifelse(num < max_num, as.character(num), NA)) %>% 
            to_lodes_form(axes=3:4, id='alluvium') %>% 
            ggplot(aes(x=x, stratum=stratum, alluvium=alluvium, y=num, fill=stratum, 
                       label = num)) +
            # scale_x_discrete(expand = c(.1, .1)) +
            scale_x_discrete(expand = c(.4, 0)) +
            geom_flow() +
            geom_stratum(alpha = .5) +
            geom_text(stat = "stratum", size=3) +
            theme_minimal() +
            # theme(legend.title = "Status") +
            theme(text = element_text(size=element_text_size)) +
            labs(fill='Status') +
            scale_fill_brewer(palette='Spectral') +
            xlab('Experiment') + ylab('Number of instances') +
            guides(color = guide_legend(override.aes = list(size=legend_size))) +
            # ggrepel::geom_text_repel(
            #     aes(label = label2),
            #     stat = "stratum", size = 4, direction = "y", nudge_x = -0.5
            # ) +
            # ggfittext::geom_fit_text(stat = "stratum", width = 1, min.size = 2) +
            ggsave(path)
    }
    
    path <- '%stransitions_base_2tasks.png' %>% sprintf(path_export_img)
    transitions %>% 
        filter(experiment=='base_a2r') %>% 
        rename(base=prev_status, base_a2r=post_status) %>% 
        graph_parent(path=path)
    path <- '%stransitions_old_2tasks.png' %>% sprintf(path_export_img)
    transitions %>% 
        filter(experiment=='old_a2r') %>%
        rename(old=prev_status, old_a2r=post_status) %>% 
        graph_parent(path=path)    
    
    # infeasible: fixLP
    infeasible_stats <- get_infeasible_stats(df_fixed)
    errors_stats <- get_soft_constraints(df_fixed) %>% 
        mutate(Indicator = Indicator %>% str_replace('\\_', '\\\\_'))
    path <- '%sinfeas_fixed_2tasks.tex' %>% sprintf(path_export_tab)
    
    infeasible_stats %>% 
        filter(!(Indicator %in% c('Total', 'Infeasible'))) %>% 
        mutate(Indicator = '%s $\\to$ Infeasible' %>% sprintf(Indicator)) %>% 
        bind_rows(errors_stats) %>% 
        ungroup %>% select(-scenario) %>% 
        set_names(., names(.) %>% str_replace('\\_', '\\\\_')) %>% 
        kable(format='latex', booktabs = TRUE, linesep="", escape=FALSE) %>% 
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

# forecasting results  -------------------------------------------------------
# TODO: finish this function to generate all graphs
make_forecasting_results <- function(result_tab, element_text_size=10, dataset='IT000125_20190716'){
    
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

make_forecasting_limits <- function(result_tab, element_text_size=10, dataset='IT000125_20190716'){
    options <- 
        list(
            "q"=0.9, 
            "bound"='upper', 
            "y_var"='mean_dist_complete',
            "method"='QuantReg', 
            "max_iter" = 10000, 
            "test_perc"= 0.3
        )
    
    y_predicted <- get_predicted_values(result_tab, options)
    labels1 <- '$\\mu_{WC}$: %s/3' %>% sprintf(c(1, 2, 3)) %>% TeX
    labels2 <- '$\\mu_{C}$: %s/3' %>% sprintf(c(1, 2, 3)) %>% TeX
    cut2 <- function (x, labels) cut(x, 
                                     quantile(x, probs = seq(0, 1, by = 1/3)), 
                                     labels=labels, 
                                     include.lowest=TRUE)
    
    result_tab_n <- 
        result_tab %>% 
        mutate(mean_consum_cut = cut2(mean_consum, labels2),
               geomean_cons_cut = cut2(geomean_cons, labels1))

    xlab <- 'Sum of all remaining flight hours at the beginning of first period'
    ylab <- 'Average distance between maintenances'
    path <- '%sprediction_upper_bounds_%s.png' %>% sprintf(path_export_img, dataset)
    ggplot(data=result_tab_n, aes(x=init, y=mean_dist_complete)) + 
        facet_grid(
            rows=vars(mean_consum_cut),
            cols=vars(geomean_cons_cut),
            labeller=label_parsed
        ) +
        # facet_grid(mean_consum_cut ~ geomean_cons_cut) +
        geom_point(alpha=0.4, height=0.2) +
        geom_point(aes(y=y_predicted), color='blue', shape=1, alpha=0.8, size=0.5) +
        theme_minimal() +
        theme(axis.text.x = element_text(hjust=0),
              strip.text.y = element_text(angle=0),
              text = element_text(size=element_text_size)) +
        xlab(xlab) + 
        ylab(ylab) +
        ggsave(path)
    
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


# run all -----------------------------------------------------------------


if (FALSE){
    data_optimisation_results <- get_data_optimisation_results()
    do.call(make_optimisation_results, args=data_optimisation_results)
    
    result_tab <- get_result_tab('IT000125_20190716')
    make_forecasting_results(result_tab, element_text_size=15)
    make_forecasting_limits(result_tab, element_text_size=15)
    
    data_summary <- get_data_optimisation_summary()
    make_optimisation_summary(data_summary)
 
}

if (FALSE){
    # getting summary of variables and constraints
    # 
    results <- 
        data_optimisation_results$raw_df_progress %>% 
        filter(sol_code > 0) %>% 
        mutate(constraints = lapply(matrix, "[[", 'constraints') %>% unlist,
               variables = lapply(matrix, "[[", 'variables') %>% unlist) %>% 
        select(experiment, constraints, variables) %>% 
        group_by(experiment) %>%
        summarise(cons = mean(constraints),
                  vars = mean(variables))
    
    results[813, ]
    lapply(results$matrix, "[[", 'constraints') %>% 
        results %>% filter((matrix %>% length)>0) %>% nrow
}

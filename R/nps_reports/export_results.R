source('nps_reports/graphs.R')
source('nps_reports/datasets.R')
library(stringr)

path_export_img <- '../../NPS2019/img/'
path_export_tab <- '../../NPS2019/tab/'
# size <- 'small'
# df_original <- compare_sto$get_instances[[size]]()


# optimization results ----------------------------------------------------

# exp_list <- c('IT000125_20190808', 'IT000125_20190812')
exp_list <- c('IT000125_20190725', 'IT000125_20190716') # small
df <- get_1_tasks()
# df_original <- compare_sto$get_df_comparison(exp_list)
# df <- df_original %>% 
#     mutate(experiment=if_else(experiment==0, 'cuts', 'base')) %>% 
#     filter_all_exps

# df <- get_4_tasks_perc_add()

# summary
summary_stats <- get_summary(df)
path <- '%sstats_%s_%s.tex' %>% sprintf(path_export_tab, exp_list[1], exp_list[2])
summary_stats %>% 
    rename(Status=Indicator) %>% 
    ungroup %>% 
    select(-scenario) %>% 
    kable(format='latex', booktabs = TRUE, linesep="") %>% 
    write_file(path)

# summary interception:

summary_int <- get_comparable_sets(df)
path <- '%sstats_int_%s_%s.tex' %>% sprintf(path_export_tab, exp_list[1], exp_list[2])
summary_int %>%
    rename(Status=Indicator) %>% 
    kable(format='latex', booktabs = TRUE, linesep="", escape=FALSE) %>% 
    write_file(path)


# quality degradation
t1 <- get_quality_degr(df)
t1_rel <- t1 %>% filter(experiment=='cuts') %>% use_series(dist_min_perc)
t1_rel_filt <- t1_rel[t1_rel<10]
filtered_quant <- (1 - (t1_rel_filt %>% length) / (t1_rel %>% length))*100
length(t1_rel)
path <- '%squality_degradation_%s_%s.png' %>% sprintf(path_export_img, exp_list[1], exp_list[2])
qplot(t1_rel_filt, xlab='Relative gap (in %) among optimal solutions.', binwidth=0.3) + 
    theme(text = element_text(size=20)) + ggsave(path)

# quality performance
quality_perf <- get_quality_perf(df)
filtered_q_perf <- quality_perf %>% filter(between(dif_perc, -7, 7))
filtered_quant <- (1 - (filtered_q_perf %>% nrow)/ (quality_perf %>% nrow))*100
path <- '%squality_performance_%s_%s.png' %>% sprintf(path_export_img, exp_list[1], exp_list[2])
qplot(filtered_q_perf$dif_perc, xlab='Relative gap (in %) among integer solutions.', binwidth=0.3) +
    theme(text = element_text(size=20)) + ggsave(path)

# time performance
comparison_table <- get_time_perf_integer(df)
comparison_table_reorder <- get_time_perf_integer_reorder(df)
path <- '%stime_performance_%s_%s.png' %>% sprintf(path_export_img, exp_list[1], exp_list[2])
ggplot(data=comparison_table, aes(x=instance, y=time, color=experiment)) + 
    theme_minimal() + geom_point(size=0.5) + theme(text = element_text(size=20)) + 
    ggsave(path)

path <- '%stime_performance_ordered_%s_%s.png' %>% sprintf(path_export_img, exp_list[1], exp_list[2])
ggplot(data=comparison_table_reorder, aes(x=percentage, y=time, color=experiment)) + 
    theme_minimal() + geom_point(size=0.5) + ggplot2::xlab('instance percentage') + 
    theme(text = element_text(size=20)) + ggsave(path)

# infeasible
infeasible_stats <- get_infeasible_stats(df)

path <- '%sinfeas_%s_%s.tex' %>% sprintf(path_export_tab, exp_list[1], exp_list[2])
infeasible_stats %>% 
    kable(format='latex', booktabs = TRUE, linesep="") %>% 
    write_file(path)

# soft constraints
errors_stats <- get_soft_constraints(df, 0.95)
path <- '%ssoft_%s_%s.tex' %>% sprintf(path_export_tab, exp_list[1], exp_list[2])
errors_stats %>% 
    ungroup %>% 
    select(-scenario) %>% 
    kable(format='latex', booktabs = TRUE, linesep="") %>% 
    write_file(path)

# CBC comparison
df <- get_1_tasks_CBC_CPLEX()
comparison_table_reorder <- get_time_perf_integer_reorder(df)

equiv=list(base='base_CBC', cuts='cuts_CBC', cplex_base='base_CPLEX')
comparison_table_reorder_n <- 
    comparison_table_reorder %>% 
    ungroup %>% 
    mutate(experiment=equiv[experiment] %>% unlist)
    
path <- '%s1task_CBC_CPLEX_times.png' %>% sprintf(path_export_img)
ggplot(data=comparison_table_reorder_n, aes(x=percentage, y=time, color=experiment)) + 
    theme_minimal() + geom_point(size=0.5) + ggplot2::xlab('instance percentage') + 
    theme(text = element_text(size=20)) + ggsave(path)
# times_cbc <- get_time_perf_optim(df)
# df %>%  nrow
status_cbc <- 
    get_summary(df) %>% 
    ungroup %>% 
    select(Indicator, base_CBC=base, base_CPLEX=cplex_base, cuts_CBC=cuts)

path <- '%s1task_CBC_CPLEX.tex' %>% sprintf(path_export_tab)
status_cbc %>% 
    kable(format='latex', booktabs = TRUE, linesep="") %>% 
    write_file(path)

comparison_table %>% 
    group_by(experiment) %>% 
    summarise(time_mean = mean(time), 
              time_medi = median(time))



# prediction models -------------------------------------------------------

# maintenances 

dataset <- 'IT000125_20190716'
result_tab <- get_result_tab(dataset)
result_tab_n <- result_tab %>% filter(gap_abs<100 & num_errors==0)

path <- '%smean_consum_vs_maints_nocolor_%s.png' %>% sprintf(path_export_img, dataset)
ggplot(data=result_tab_n, aes(y=maints, x=mean_consum)) + 
    geom_jitter(alpha=0.4, height=0.2) + theme_minimal() + 
    facet_grid('init_cut ~ .') +
    theme(axis.text.x = element_text(hjust=0),
          strip.text.y = element_text(angle=0),
          text = element_text(size=20)) +
    xlab('Average consumption in flight hours per period') + 
    ylab('Total number of maintenances') + 
    ggsave(path)

# quantiles




# summary optimization ----------------------------------------------------

if (FALSE){
    options <- c('get_all_tasks', 
                 'get_all_tasks_aggresive', 
                 'get_all_tasks_aggresive_percadd', 
                 'get_all_tasks_very_aggresive_percadd')
    names(options) <- options
    fun_ <- function(func_name){
        func_name %>% do.call(args=list()) %>% get_mega_summary
    }
    summary <- options %>% lapply(fun_)
    equiv <- list(numparalleltasks_2='30', 
                  numparalleltasks_3='45', 
                  numparalleltasks_4='60')
    equiv2 <- list(get_all_tasks='cuts', 
                   get_all_tasks_aggresive='cuts\\_a', 
                   get_all_tasks_aggresive_percadd='cuts\\_a\\_rec',
                   get_all_tasks_very_aggresive_percadd='cuts\\_aa\\_rec')
    
    col_names <- c("", "$|\\mathcal{I}|$", "$\\mu_e$", "$q95_e$", "Feas", "Infeas", "$\\mu_q$", 
                   "$med_q$", "$q95_q$", "$\\mu_t$", "$med_e$")
    table_out <- 
        summary %>% 
        bind_rows(.id='experiment') %>% 
        ungroup %>% 
        mutate(scenario=equiv[scenario] %>% unlist,
               experiment=equiv2[experiment] %>% unlist) %>% 
        set_names(col_names)
    
    names(table_out) <- str_replace(names(table_out), '\\_', '\\\\_')
    path <- '%scompare_all.tex' %>% sprintf(path_export_tab)
    table_out %>% 
        kable(format='latex', booktabs = TRUE, linesep="", escape=FALSE) %>% 
        write_file(path)
    
}

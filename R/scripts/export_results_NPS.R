source('functions/graphs_NPS.R')

path_export_img <- '../../NPS2019/img/'
path_export_tab <- '../../NPS2019/tab/'
# size <- 'small'
# df_original <- compare_sto$get_instances[[size]]()

exp_list <- c('IT000125_20190808', 'IT000125_20190812')
exp_list <- c('IT000125_20190725', 'IT000125_20190716') # small
df_original <- compare_sto$get_df_comparison(exp_list)
df <- df_original %>% mutate(experiment=if_else(experiment==0, 'cuts', 'base'))

# summary
summary_stats <- get_summary(df)
path <- '%sstats_%s_%s.tex' %>% sprintf(path_export_tab, exp_list[1], exp_list[2])
summary_stats %>% 
    rename(Status=Indicator) %>% 
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
qplot(t1_rel_filt, xlab='Relative gap (in %) among optimal solutions.', binwidth=0.3) + ggsave(path)

# quality performance
quality_perf <- get_quality_perf(df)
filtered_q_perf <- quality_perf %>% filter(between(dif_perc, -7, 7))
filtered_quant <- (1 - (filtered_q_perf %>% nrow)/ (quality_perf %>% nrow))*100
path <- '%squality_performance_%s_%s.png' %>% sprintf(path_export_img, exp_list[1], exp_list[2])
qplot(filtered_q_perf$dif_perc, xlab='Relative gap (in %) among integer solutions.', binwidth=0.3) + ggsave(path)

# quality time
comparison_table <- get_time_perf_integer(df)
comparison_table_reorder <- get_time_perf_integer_reorder(df)
path <- '%stime_performance_%s_%s.png' %>% sprintf(path_export_img, exp_list[1], exp_list[2])
ggplot(data=comparison_table, aes(x=instance, y=time, color=experiment)) + 
    theme_minimal() + geom_point(size=0.5) + ggsave(path)

path <- '%stime_performance_ordered_%s_%s.png' %>% sprintf(path_export_img, exp_list[1], exp_list[2])
ggplot(data=comparison_table_reorder, aes(x=percentage, y=time, color=experiment)) + 
    theme_minimal() + geom_point(size=0.5) + ggplot2::xlab('instance percentage') + ggsave(path)

# infeasible
infeasible_stats <- get_infeasible_instances(df)
infeasible_instances <- get_infeasible_stats(df)

path <- '%sinfeas_%s_%s.tex' %>% sprintf(path_export_tab, exp_list[1], exp_list[2])
infeasible_instances %>% 
    kable(format='latex', booktabs = TRUE, linesep="") %>% 
    write_file(path)

# soft constraints
errors_stats <- get_soft_constraints(df, 0.95)
path <- '%ssoft_%s_%s.tex' %>% sprintf(path_export_tab, exp_list[1], exp_list[2])
errors_stats %>% 
    kable(format='latex', booktabs = TRUE, linesep="") %>% 
    write_file(path)

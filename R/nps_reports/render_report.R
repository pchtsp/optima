library(rmarkdown)
# size <- 'medium'
# options <- c('get_4_tasks_aggressive', 'get_1_tasks_maints', 'get_4_tasks', 'get_1_tasks')
options <- list(list('size'='get_all_stoch', 'num_tasks'=3, 'scale'='identity'), 
             list('size'='get_all_stoch', 'num_tasks'=2, 'scale'='identity'),
             list('size'='get_all_stoch', 'num_tasks'=4, 'scale'='identity'))
# 'scale' = 'identity' # 'identity' or 'log2'
# options <- paste0('get_determ_', c(1, 2, 3, 4))
format <- 'pdf'
format <- 'html'
for (option in options){
    rmarkdown::render("nps_reports/results.Rmd", 
                      output_file = sprintf('nps_reports/NPS_report_%s_%s.%s', 
                                            option$size, option$num_tasks, format),
                      params = option, 
                      output_options = list(self_contained = TRUE),
                      output_format=sprintf('%s_document', format),
                      knit_root_dir=getwd())
}

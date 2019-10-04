library(rmarkdown)
# size <- 'medium'
# options <- c('get_4_tasks_aggressive', 'get_1_tasks_maints', 'get_4_tasks', 'get_1_tasks')
options <- c('get_4_tasks_very_aggresive_percadd')
format <- 'pdf'
format <- 'html'
for (size in options){
    rmarkdown::render("scripts/resultsNPS.Rmd", 
                      output_file = sprintf('reports/NPS_report_%s.%s', size, format),
                      params = list(size = size), 
                      output_options = list(self_contained = TRUE),
                      output_format=sprintf('%s_document', format),
                      knit_root_dir=getwd())
}
library(rmarkdown)
# size <- 'medium'
# paste0(size,"_report_", Sys.Date(), ".html")
options <- c('medium', 'large', 'small')
# options <- c('medium')
format <- 'pdf'
format <- 'html'
for (size in options){
    rmarkdown::render("scripts/resultsNPS.Rmd", 
                      output_file = sprintf('NPS_report_%s.%s', size, format),
                      params = list(size = size), 
                      output_options = list(self_contained = TRUE),
                      output_format=sprintf('%s_document', format))
}

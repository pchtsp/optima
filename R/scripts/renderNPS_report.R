library(rmarkdown)
# size <- 'medium'
# paste0(size,"_report_", Sys.Date(), ".html")
options <- c('medium', 'large', 'small')
# options <- c('medium')
for (size in options){
    rmarkdown::render("scripts/resultsNPS.Rmd", 
                      output_file = sprintf('NPS_report_%s.html', size),
                      params = list(size = size), 
                      output_options = list(self_contained = TRUE))
}

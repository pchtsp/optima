#
# This is the server logic of a Shiny web application. You can run the 
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
# 
#    http://shiny.rstudio.com/
#

library(shiny)
library(timevis)

experiment1 <- PATHS[['experiments']] %>% paste0('201802061201/')
# Define server logic required to draw a histogram
shinyServer(function(input, output) {
    dir_pre <- "../"
    sources <-
        c("functions/import_results.R",
          "functions/params.R",
          "diagrams/exampleDT.R") %>% paste0(dir_pre, .)
    lapply(sources, source)

    set.seed(122)
    histdata <- rnorm(500)
    states <- get_states(experiment1)
    
    output$plot1 <- renderPlot({
        data <- histdata[seq_len(input$slider)]
        hist(data)
    })
    
    output$timevis1 <- renderTimevis(timevis_from_states(states, height = 500))
    
    observe({
        id_selected <- input$timevis1_selected  
        if (id_selected %>% is.null) return()
        id_selected <- id_selected %>% as.integer
        res <- states %>% filter(id==id_selected) %>% use_series(group)
        output$resource <- renderText({sprintf("Aéronef: %s", res)})
        if (input$option_graph == 'tie'){
            output$plotly1 <- renderPlotly({graph_tie(experiment1, res)})
        } else {
            output$plotly1 <- renderPlotly({graph_remaining(experiment1, res)})
        }
    })
})

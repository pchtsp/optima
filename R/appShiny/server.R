library(shiny)
library(timevis)
library(tidyverse)
library(DT)
library(lubridate)
library(data.table)

shinyServer(function(input, output, clientData, session) {
    dir_pre <- "../"
    sources <-
        c("functions/import_results.R",
          "functions/params.R",
          "functions/exampleDT.R") %>% paste0(dir_pre, .)
    lapply(sources, source)

    experiments <- PATHS[['experiments']] %>% completed_experiments()
    default_exp <- '201802061201'
    experiment1 <- experiments[default_exp]
    updateSelectInput(session = session, inputId = "selectExp", choices = experiments, selected = experiment1)
    
    states <- reactive({get_states(input$selectExp, style_config= list(font_size='12px'))})
    missions <- reactive({get_tasks(input$selectExp)})
    parameters <- reactive({get_parameters(input$selectExp)})

    output$missions = DT::renderDataTable({missions() %>% DT::datatable(options = list(dom = 't'))})
    output$parameters = DT::renderDataTable({parameters() %>% DT::datatable(options = list(dom = 't'))})
    
    observe({
        output$timevis1 <- renderTimevis(timevis_from_states(states(), height = 500))
    })
        
    observe({
        # browser()
        id_selected <- input$timevis1_selected
        dates <- input$timevis1_window
        if (id_selected %>% is.null | dates %>% is.null) return()
        id_selected <- id_selected %>% as.integer
        res <- states() %>% filter(id==id_selected) %>% use_series(group)
        output$resource <- renderText({sprintf("Aéronef: %s", res)})
        filter <- months_in_range(dates)
        # browser()
        data_input <- 
            treat_data(experiment1, res) %>% 
            right_join(filter)
            # filter(Mois >= dates[1] & Mois <= dates[2])
        # if (input$option_graph == 'tie'){
        #     output$plotly1 <- renderPlotly({graph_tie_data(data_input)})
        # } else {
            output$plotly1 <- renderPlotly({graph_remaining_data(data_input)})
        # }

            
    })
})

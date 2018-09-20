library(DT)
library(tidyverse)
library(data.table)
library(magrittr)
library(ggplot2)
library(plotly)
library(stringr)

treat_data <- function(exp_directory, aircraft_to_graph){
    # browser()
    solution_path = exp_directory %>% paste0('data_out.json')
    
    solution <- read_json(solution_path)
    
    evolution_rut <- 
        solution %>% 
        extract2("aux") %>% 
        extract2("rut") %>% 
        bind_rows(.id="aircraft") %>% 
        gather(key='Mois', value = 'Temps', -aircraft) %>% 
        filter(aircraft == aircraft_to_graph) %>% 
        mutate(type='heures de vol')
    
    evolution_ret <- 
        solution %>% 
        extract2("aux") %>% 
        extract2("ret") %>% 
        bind_rows(.id="aircraft") %>% 
        gather(key='Mois', value = 'Temps', -aircraft) %>% 
        filter(aircraft == aircraft_to_graph) %>% 
        mutate(type='disponibilité (mois)')
    
    data <- bind_rows(evolution_ret, evolution_rut) %>% 
        mutate(Mois= Mois %>% paste0("-01") %>% as.POSIXct())
    data
}

table_remaining <- function(exp_directory, aircraft_to_graph='A31'){
    data <- treat_data(exp_directory, aircraft_to_graph)
    
    data %<>% mutate(Mois = Mois %>% format("%Y-%m"))
    
    DT::datatable(data, editable = TRUE, filter='none')
    
    }

    
graph_remaining <- function(exp_directory, aircraft_to_graph='A31'){
    data <- treat_data(exp_directory, aircraft_to_graph)
    graph_remaining_data(data)
}

graph_remaining_data <- function(data, x= 'Mois', y= 'Temps', return_object=FALSE){
    p <- ggplot(data, aes_string(x= x, y= y, group='aircraft')) + 
        geom_line() + 
        facet_grid(type~ ., scales = "free_y") +
        theme_minimal() + 
        scale_x_datetime()
    if (return_object){
        return(p)
    }
    ggplotly(p)
}

graph_tie <- function(exp_directory, aircraft_to_graph='A31'){
    data <- treat_data(exp_directory, aircraft_to_graph)
    graph_tie_data(data)
}

graph_tie_data <- function(data){
    data_n <- data %>% spread(key = type, value=Temps)
    
    p <- ggplot(data_n, aes(x= `heures de vol`, y= `disponibilité (mois)`, group='aircraft')) + 
        geom_line() + 
        theme_minimal()
    
    ggplotly(p)
    
}

if (FALSE) {
    dir_pre <- "./"
    sources <-
        c("functions/import_results.R",
          "functions/params.R") %>% paste0(dir_pre, .)
    lapply(sources, source)
    
    experiments <- PATHS[['experiments']] %>% completed_experiments()
    default_exp <- '201802061201'
    experiment1 <- experiments[default_exp]
    
    res <- 'A20'
    # browser()
    data_input <- treat_data(experiment1, res)
    graph_remaining_data(data_input)

    
}

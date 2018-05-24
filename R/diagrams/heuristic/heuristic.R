library(tidyverse)
library(data.table)
library(magrittr)
library(DiagrammeR)

graph_fill_mission <- function(path, ...){
    edges <- read_csv2(path)
    # names_nodes = c(edges$from, edges$to) %>% unique
    
    edges %<>% mutate(to = if_else(link %>% is.na, to, paste0('|', link, '|', to)),
                      line = paste(from, to, sep='-->'))
    text <- edges$line %>% paste(collapse = ';') %>% paste("graph TB", ., sep=';')
    
    DiagrammeR(text, ...)
}

graph_solve <- function(path, ...){
    edges <- read_csv2(path)
    # names_nodes = c(edges$from, edges$to) %>% unique
    
    edges %<>% 
        mutate(to = if_else(link %>% is.na, to, paste0('|', link, '|', to)),
               line = paste(from, to, sep='-->'),
               line = if_else(from == "for_each_task", 
                              paste("subgraph for each task", line, "end", sep="\n"),
                              line)
               )
    text <- edges$line %>% paste(collapse = '\n') %>% paste("graph TB", ., sep='\n')
    
    DiagrammeR(text, ...)
    #     visInteraction(zoomView=FALSE)
}

graph_find_assign_maintenance <- function(path, ...){
    edges <- read_csv2(path)
    nodes <- data.table(label = c(edges$from, edges$to) %>% unique) %>% mutate(id = row_number())
    edges %>% 
        inner_join(nodes, by=c('from'='label')) %>% 
        inner_join(nodes, by=c('to'='label')) %>% 
        mutate(from= id.x, to=id.y) %>% 
        select(-ends_with(".x")) %>% 
        select(-ends_with(".y"))
    
    edges %<>% mutate(to = if_else(link %>% is.na, to, paste0('|', link, '|', to)),
                      line = paste(from, to, sep='-->'))
    text <- edges$line %>% paste(collapse = ';') %>% paste("graph TB", ., sep=';')
    
    DiagrammeR(text, ...)
    
}
library(tidyverse)
library(readxl)
library(magrittr)
library(stringr)
library(zoo)
library(data.table)
library(lubridate)
library(timevis)
library(Hmisc)
library(jsonlite)

print_tasks <- function(exp_directory, max_tasks=9999, ...){
    input_path = exp_directory %>% paste0('data_in.json')
    input <- read_json(input_path)
    tasks <-
        input %>% 
        extract2('tasks') %>% 
        lapply("[", c('consumption', 'num_resource', 'start', 'end')) %>% 
        bind_rows(.id="group") %>% 
        sample_n(., min(max_tasks, nrow(.))) %>% 
        arrange(consumption) %>% 
        mutate(id= c(1:nrow(.)),
               content = sprintf('%s aéronefs (%sh)', num_resource, consumption),
               color = RColorBrewer::brewer.pal(5, "YlOrRd")[rep_len(1:5, n())],
               style= sprintf("background-color:%s;border-color:%s;font-size: 15px", color, color),
        ) %>% 
        slice(sample(1:n()))
        
    make_tasks_gantt(tasks)
}

make_tasks_gantt <- function(data, ...){
    
    config <- list(
        editable = TRUE,
        align = "center",
        orientation = "top",
        snap = NULL,
        margin = 0,
        zoomable= FALSE
    )
    
    groups <- data %>% distinct(group) %>% rename(id= group) %>% mutate(content= id)
    timevis(data, groups= groups, options= config, ...)
}
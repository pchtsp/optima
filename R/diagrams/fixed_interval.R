library(tidyverse)
library(data.table)
library(timevis)
library(lubridate)
library(visNetwork)
library(magrittr)
library(RColorBrewer)

tasks_gantt <- function(...){
    start_month <- as.Date('2017-10-01')
    
    data <- 
        data.table(
            Phase=c(1, 1, 2, 3, 4, 5, 6, 7)
            ,label=LETTERS[1:8]
        ) %>% 
        mutate(
            month_start=sample(1:5, 8, replace=TRUE)
            ,month_end= month_start + sample(1:4, 8, replace=TRUE)
        ) %>% 
        mutate(start= start_month %m+% months(month_start-1),
               end= start_month %m+% months(month_end),
               group= Phase,
               id= c(1:nrow(.)),
               content = label,
               color = RColorBrewer::brewer.pal(8, "YlOrRd")[rep_len(1:8, n())],
               style= sprintf("background-color:%s;border-color:%s;font-size: 15px", color, color),
        )
    config <- list(
        editable = TRUE,
        align = "center",
        orientation = "top",
        snap = NULL,
        margin = 0,
        zoomable= FALSE
    )
    
    groups <- data %>% distinct(group) %>% rename(id= group) %>% mutate(content= id)
    timevis(data, groups= groups, options= config)
}

tasks_employees <- function(){
    tasks <- LETTERS[1:8] %>% set_names(., .)
    employees <- 1:10
    task_colors <- RColorBrewer::brewer.pal(length(tasks), "YlOrRd")
    n_t <- length(tasks)
    
    nodes <- 
        data.table(label=c(tasks, employees)) %>% 
        mutate(
            id = row_number(),
            group= c(
                    rep('T', length(tasks))
                    ,rep('E', length(employees))
            ),
            color = c(
                task_colors
                ,rep("gray",length(employees))
            )
            )

    edges_data <- 
        lapply(tasks, function(x) {
            sample(employees, sample(2:4, 1)) %>% data.table(employee=.) 
            }) %>% 
        bind_rows(.id='task') %>% 
        mutate(employee = employee %>% as.character())
    
    edges <- 
        edges_data %>% 
        inner_join(nodes, by=c('employee'='label')) %>% 
        inner_join(nodes, by=c('task'='label')) %>% 
        mutate(from= id.x, to=id.y) %>% 
        select(-ends_with(".x")) %>% 
        select(-ends_with(".y"))
    
    visNetwork(nodes, edges) %>% 
        visGroups(groupname = "T", value = 20, color='red') %>%
        visGroups(groupname = "E", value = 2, color='gray') %>%
        visLegend(addNodes = list(
            list(label='Tasks', value = 20, shape='icon', 
                  icon = list(code = 'f111', color=task_colors[n_t], size=50))
            ,list(label='Resources', value =2, shape='icon', 
                  icon = list(code = 'f111', color='gray', size=15))
        ), useGroups = FALSE, position='right'
        )
}


library(visNetwork)
library(tidyverse)
library(data.table)
library(magrittr)

maint_fix <- function(){
    resources = paste0("A", 1:4) %>% data.table(label=., color='blue')
    periods = paste0("T", 1:8) %>% data.table(label=., color='green')
    maints = paste0("M", 1:6) %>% data.table(label=., color='yellow')
    
    res_maint <- 
        data.table(resource=resources$label, maint=maints$label) %>% 
        group_by(resource) %>% 
        mutate(pos_maint=row_number(),
               tot_maint=n()) %>% 
        merge(data.table(time=periods$label)) %>% 
        arrange(resource, maint) %>% 
        group_by(resource, maint) %>% 
        mutate(pos_period=row_number(),
               rn = round(runif(1)*2 -1)) %>% 
        filter(pos_period/n() <= pos_maint/tot_maint + rn &
                   pos_period/n() > (pos_maint-1)/tot_maint + rn) %>% data.table
    nodes <- 
        list(resources, periods, maints) %>% 
        bind_rows() %>% 
        mutate(id = row_number())
    
    edges_data <- bind_rows(res_maint %>% distinct(resource, maint) %>% rename(from=resource, to=maint), 
                            res_maint %>% distinct(maint, time) %>% rename(from=maint, to=time)
    )
    
    edges <- 
        edges_data %>% 
        inner_join(nodes, by=c('from'='label')) %>% 
        inner_join(nodes, by=c('to'='label')) %>% 
        mutate(from= id.x, to=id.y) %>% 
        select(-ends_with(".x")) %>% 
        select(-ends_with(".y"))
    
    visNetwork(nodes, edges, height = "800px", width = "800px") %>% 
        visNodes(shapeProperties = list(useBorderWithImage = TRUE))
    
    
    
}
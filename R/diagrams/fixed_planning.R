library(timevis)
library(tidyverse)
library(data.table)
library(magrittr)

config <- list(
    stack = FALSE,
    editable = TRUE,
    align = "center",
    orientation = "top",
    snap = NULL,
    margin = 0,
    zoomable= FALSE
)

data.table(
    id = 'A1'
    ,start= c('2018-01', '2018-10', '2019-05')
    ,end= c('2018-01', '2018-10', '2019-05')
)

states <- 
    solution %>% 
    extract2('state') %>% 
    bind_rows(.id = "UNIT") %>% 
    gather(key = 'month', value = 'state', -UNIT) %>% 
    filter(state %>% is.na %>% not) %>% 
    bind_rows(tasks) %>% 
    collapse_states() %>% 
    inner_join(task_hours) %>% 
    mutate(id= c(1:nrow(.)),
           style= sprintf("background-color:%s;border-color:%s;font-size: 15px", color, color),
           content = if_else(state=='M', 'M', sprintf('%s (%sh)', state, hours))
    ) %>% 
    select(id, start= month, end= post_month, content, group= UNIT, style)

if (max_resources %>% is.null %>% not){
    resources <- states %>% distinct(group) %>% sample_n(max_resources)
    states <- states %>% inner_join(resources)
}

groups_c <- states %>% distinct(group) %>% unlist
groups <- data.table(id= groups_c, content= groups_c)
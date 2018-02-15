library(tidyverse)
library(readxl)
library(magrittr)
library(stringr)
library(zoo)
library(data.table)
library(lubridate)
library(timevis)
library(Hmisc)
library(RColorBrewer)
library(htmlwidgets)
library(jsonlite)

collapse_states <- function(table){
    
    # we complete missing months
    months <- table$month
    resources <- table %>% extract2('UNIT')
    min_date <- months %>% min %>% paste0('-01') %>% as.Date()
    max_date <- months %>% max %>% paste0('-01') %>% as.Date()
    total_rows <- 
        seq(from= min_date, to=max_date, "months") %>% 
        as.character() %>% 
        str_sub(end='7') %>% 
        CJ(month= ., UNIT=resources)
    
    table %>% 
        full_join(total_rows) %>% 
        mutate(state = if_else(is.na(state), "", state),
               temp = state) %>% 
        arrange(UNIT, month) %>% 
        group_by(UNIT) %>%  
        mutate(prev_temp= temp %>% lag,
               first_row= row_number()==1,
               change = temp != prev_temp) %>% 
        filter(first_row | change) %>% 
        mutate(post_month = month %>% lead %>% paste0("-01") %>% ymd() %>% add(0) %>% format("%Y-%m"),
               post_month = post_month %>% if_else(is.na(.), max_date %>% ymd() %>% format("%Y-%m"), .)) %>% 
        select(-temp, -prev_temp, -first_row, -change) %>% 
        ungroup() %>% filter(state != "")
}

print_solution <- function(exp_directory, max_resources=NULL){
    solution_path = exp_directory %>% paste0('data_out.json')
    input_path = exp_directory %>% paste0('data_in.json')
    
    solution <- read_json(solution_path)
    input <- read_json(input_path)
    
    task_hours <-
        input %>% 
        extract2('tasks') %>% 
        lapply("[[", 'consumption') %>% 
        bind_rows() %>% 
        gather(key='state', value = 'hours') %>% 
        bind_rows(data.table(state='M', hours=0)) %>% 
        mutate(bucket= hours %>% as.integer %>% multiply_by(-1) %>% cut2(g= 4),
               color = RColorBrewer::brewer.pal(4, "RdYlGn")[bucket])
    
    tasks <- 
        solution %>% 
        extract2('task') %>% 
        bind_rows(.id = "UNIT") %>% 
        gather(key = 'month', value = 'state', -UNIT) %>% 
        filter(state %>% is.na %>% not)
    
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
        resources <- states %>% distinct(group) %>% slice(1:max_resources)
        states <- states %>% inner_join(resources)
    }
    
    groups_c <- states %>% distinct(group) %>% unlist
    groups <- data.table(id= groups_c, content= groups_c)
    
    config <- list(
        stack = FALSE,
        editable = TRUE,
        align = "center",
        orientation = "top",
        snap = NULL,
        margin = 0
    )
    
    timevis(states, groups= groups, options= config, width="100%")
}

if (FALSE){
    
    path <- '/home/pchtsp/Documents/projects/OPTIMA/data/raw/Planifs M2000.xlsm'
    
    
    sheets <- readxl::excel_sheets(path)
    table <- read_xlsx(path, "M2D")
    months_names <- str_split("Ja  Fe  Ma  Av  Mi  Jn  Jt  Au  Se  Oc  No  De", "\\s+") %>% unlist
    equiv <- data.table(value=months_names, num=c(1:length(months_names)) %>% str_pad(2, "left", "0"))
    
    table_names <- 
        table %>% 
        slice(1) %>% 
        gather() %>% 
        full_join(equiv) %>% 
        # mutate(year= years) %>% 
        mutate(w_year = str_detect(key, "(^\\d{4}\\s)|^2"),
               year = if_else(w_year, key, "") %>% str_extract("^\\d{4}") %>% na.locf(na.rm = FALSE),
               colname= if_else(is.na(year) | is.na(num), make.names(key), paste(year, num, sep="-"))) 
    states <- 
        table %>% 
        set_names(table_names$colname) %>% 
        select(UNIT= X__3, starts_with("20")) %>% 
        filter(str_sub(UNIT, 1, 1)=="D") %>% 
        slice(2:78) %>% 
        gather(key= "month", value="state", starts_with("20")) %>% 
        collapse_states() %>% 
        mutate(id= c(1:nrow(.))) %>% 
        mutate(bucket= state %>% as.integer %>% cut2(g= 5),
               maint = state %>% str_detect('^V\\d+'),
               color= RColorBrewer::brewer.pal(5, "YlOrRd")[bucket],
               color= if_else(maint, "black", color),
               color= if_else(color %>% is.na, '', color),
               style= sprintf("background-color:%s;border-color:%s", color, color)
               ) %>% 
        select(id, start= month, end= post_month, content = bucket, group= UNIT, style) %>% 
        slice(1:1000)
    
    groups_c <- states %>% distinct(group) %>% unlist
    groups <- data.table(id= groups_c, content= groups_c)
    
    config <- list(
        stack = FALSE,
        editable = TRUE,
        align = "center",
        orientation = "top",
        snap = NULL,
        margin = 0
    )
    
    result <- timevis(states, groups= groups, options= config)
    saveWidget(result, file = "/home/pchtsp/Downloads/test.html", selfcontained = FALSE)
}


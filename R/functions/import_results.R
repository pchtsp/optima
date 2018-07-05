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
library(zoo)

collapse_states <- function(table){
    
    # we complete missing months
    months <- table$month
    resources <- table %>% extract2('resource')
    min_date <- months %>% min %>% paste0('-01') %>% as.Date()
    max_date <- months %>% max %>% paste0('-01') %>% as.Date()
    total_rows <- 
        seq(from= min_date, to=max_date, "months") %>% 
        as.character() %>% 
        str_sub(end='7') %>% 
        CJ(month= ., resource=resources)
    
    table %>% 
        full_join(total_rows) %>% 
        mutate(state = if_else(is.na(state), "", state),
               temp = state) %>% 
        arrange(resource, month) %>% 
        group_by(resource) %>%  
        mutate(prev_temp= temp %>% lag,
               first_row= row_number()==1,
               change = temp != prev_temp) %>% 
        filter(first_row | change) %>% 
        mutate(post_month = month %>% lead %>% paste0("-01") %>% ymd() %>% add(0) %>% format("%Y-%m"),
               post_month = post_month %>% if_else(is.na(.), max_date %>% ymd() %>% format("%Y-%m"), .)) %>% 
        select(-temp, -prev_temp, -first_row, -change) %>% 
        ungroup() %>% filter(state != "")
}

get_tasks <- function(exp_directory){
    input_path = exp_directory %>% paste0('data_in.json')
    if (!file.exists(input_path)){
        return(data.frame())
    }
    input <- read_json(input_path)
    input %>% 
        extract2('tasks') %>% 
        lapply(data.frame, stringsAsFactors = F) %>% 
        lapply(select, -starts_with("capacit")) %>% 
        bind_rows(.id="task")
}

get_parameters <- function(exp_directory){
    input_path = exp_directory %>% paste0('data_in.json')
    if (!file.exists(input_path)){
        return(data.frame())
    }
    input <- read_json(input_path)
    input %>% 
        extract2('parameters') %>% 
        data.frame(stringsAsFactors = F) %>% 
        gather(name, value)
}

dicts_to_df <-  function(data){
    data %>% 
        bind_rows(.id='x') %>% 
        gather(key='y', value='z', -x) %>% 
        filter(z %>% is.na %>% not)
}

get_ret <- function(exp_directory, time='used'){
    solution_path = exp_directory %>% paste0('data_out.json')
    input_path = exp_directory %>% paste0('data_in.json')
    if (!file.exists(solution_path)){
        stop(sprintf('No file named %s', solution_path))
    }
    if (!file.exists(input_path)){
        stop(sprintf('No file named %s', input_path))
    }
    solution <- read_json(solution_path)
    input <- read_json(input_path)
    initial <- sapply(input$resources, '[[', 'initial_used')
    max_rem <- input$parameters[['max_used_time']]
    previous_month <- input$parameters$start %>% paste0('-01') %>% as_date %>% subtract(ddays(1)) %>% floor_date('month') %>% format('%Y-%m')
    month_table <- months_in_range(dates=c(previous_month, input$parameters$end) %>% paste0('-1'))
    month_table_n <- 
        month_table %>% rename(period=Mois) %>% mutate(period = period %>% format('%Y-%m')) %>% 
        merge(data.table(resource=initial %>% names))
    
    consumption <- input$tasks %>% lapply('[[', 'consumption') %>% bind_rows() %>% gather(key='task', value='mod')
    tasks <- solution$task %>% dicts_to_df %>% set_names(c('resource', 'period', 'task')) %>% inner_join(consumption)
    maintenances <- solution$state %>% dicts_to_df %>% set_names(c('resource', 'period', 'state'))
    result <- 
        data.table(
            resource= initial %>% names
            ,remaining= initial %>% unlist
            ,period= previous_month
        ) %>% 
        full_join(month_table_n) %>% 
        left_join(maintenances) %>% 
        left_join(tasks) %>% 
        mutate(remaining = if_else(state %>% is.na %>% not, max_rem, remaining)) %>% 
        mutate(mod = -as.numeric(mod)) %>% 
        mutate(mod = if_else(mod %>% is.na, remaining, mod)) %>% 
        mutate(mod = if_else(mod %>% is.na, 0, mod)) %>% 
        mutate(maint_period = ifelse(remaining %>% is.na %>% not, period, NA)) %>% 
        arrange(resource, period) %>% 
        group_by(resource) %>% 
        mutate(maint_period = zoo::na.locf(maint_period)) %>% 
        group_by(resource, maint_period) %>% 
        mutate(mod_c = cumsum(mod))
        
}

months_in_range <- function(dates){
    # browser()
    date1= dates[1] %>% as_date() %>% round_date(unit="month")
    date2= dates[2] %>% as_date() %>% round_date(unit="month")
    seq(date1, by = "month", to=date2) %>% 
        as.POSIXct(tz = Sys.timezone()) %>% 
        round_date(unit="month") %>% 
        data.table(Mois=.)
}

get_states <- function(exp_directory, style_config=list()){
    # browser()
    solution_path = exp_directory %>% paste0('data_out.json')
    input_path = exp_directory %>% paste0('data_in.json')
    
    if (!file.exists(solution_path)){
        stop(sprintf('No file named %s', solution_path))
    }
    if (!file.exists(input_path)){
        stop(sprintf('No file named %s', input_path))
    }
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
        bind_rows(.id = "resource") %>% 
        gather(key = 'month', value = 'state', -resource) %>% 
        filter(state %>% is.na %>% not)
    
    # def_style_config = list(font_size='15px')
    font_size = style_config$font_size
    if (font_size %>% is.null){
        font_size <- '15px'
    }

    states <- 
        solution %>% 
        extract2('state') %>% 
        bind_rows(.id = "resource") %>% 
        gather(key = 'month', value = 'state', -resource) %>% 
        filter(state %>% is.na %>% not) %>% 
        bind_rows(tasks) %>% 
        collapse_states() %>% 
        inner_join(task_hours) %>% 
        mutate(id= c(1:nrow(.)),
               style= sprintf("background-color:%s;border-color:%s;font-size: %s", color, color, font_size),
               content = if_else(state=='M', 'M', sprintf('%s (%sh)', state, hours))
        ) %>% 
        rename(start= month, end= post_month, group= resource)
    states
}

timevis_from_states <- function(states, max_resources=NULL, ...){
    if (max_resources %>% is.null %>% not){
        resources <- states %>% distinct(group) %>% sample_n(max_resources)
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
        margin = 0,
        zoomable= FALSE
    )
    
    timevis(states, groups= groups, options= config, ...)
}

print_solution <- function(exp_directory, max_resources=NULL, ...){

    states <- get_states(exp_directory)
    
    timevis_from_states(states, max_resources=max_resources, ...)
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


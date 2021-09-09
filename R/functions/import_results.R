packs <- installed.packages()
packages <- 
    c('tidyverse', 'readxl', 'magrittr', 'stringr', 'zoo', 'data.table', 'lubridate', 'timevis', 'Hmisc', 'RColorBrewer', 'htmlwidgets', 'jsonlite')
to_install <- !(packages %in% packs)
for (p in packages[to_install]){
    install.packages(p, repos='https://cloud.r-project.org')
}

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

month_to_ymd <- function(x) x %>% paste0("-01") %>% ymd

collapse_states <- function(table){
    
    # we complete missing months
    months <- table$month
    resources <- table %>% extract2('resource')
    min_date <- months %>% min %>% paste0('-01') %>% as.Date()
    # I add one because the end month needs to be one more than the real end month
    max_date <- months %>% max %>% paste0('-01') %>% as.Date() %>% add(months(1))
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
        mutate(post_month = month %>% lead %>% month_to_ymd %>% add(0) %>% format("%Y-%m"),
               post_month = post_month %>% if_else(is.na(.), max_date %>% ymd() %>% format("%Y-%m"), .)) %>% 
        mutate(duration = interval(month_to_ymd(month), month_to_ymd(post_month)) %/% months(1)) %>% 
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

get_states <- function(exp_directory, style_config=list(), state_m='state_m'){
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
    
    # maintenance colors
    maints_tab <- 
        data.table(state=c('VG', 'VI', 'VS', 'M', 'VG+VI', 'VG+VS', 'VG+VI+VS', 'VI+VS'), 
                   color=c('#4cb33d', '#00c8c3', '#31c9ff', '#878787', rep('#EFCC00', 4))) %>% 
        mutate(hours=0)
    # maints_tab <- 
    #     input %>% 
    #     extract2('maintenances') %>% 
    #     lapply("[[", 'priority') %>% 
    #     bind_rows() %>% 
    #     gather(key='state', value = 'hours') %>% 
    #     mutate(hours=0) %>% 
    #     right_join(colors_maint)
    
    # tasks colors
    task_hours <-
        input %>% 
        extract2('tasks') %>% 
        lapply("[[", 'consumption') %>% 
        bind_rows() %>% 
        gather(key='state', value = 'hours')
    if (nrow(task_hours)>0){
        task_hours <-
            task_hours %>% 
            mutate(bucket= hours %>% as.integer %>% multiply_by(-1) %>% cut2(g= 4),
                   color = RColorBrewer::brewer.pal(4, "RdYlGn")[bucket]) %>% 
            select(state, hours, color)
    } else {
        task_hours <- data.table(color=as.character(), state=as.character(), hours=as.integer())
    }
    task_hours <- 
        task_hours %>% 
        # merge with maintenances
        bind_rows(maints_tab)
    
    # get task assignments
    tasks <- 
        solution %>% 
        extract2('task') %>% 
        bind_rows(.id = "resource")
    
    if (nrow(tasks)>0){
        tasks <- 
            tasks %>% 
            gather(key = 'month', value = 'state', -resource) %>% 
            filter(state %>% is.na %>% not)        
    } else {
        tasks <- data.table(resource=as.character(), month=as.character(), state=as.character())
    }

    # def_style_config = list(font_size='15px')
    font_size = style_config$font_size
    if (font_size %>% is.null){
        font_size <- '15px'
    }

    # get maintenance assignments
    treat_maint <- function(res){
        if (state_m=='state'){
            # deprecation function
            res %>% 
                bind_rows(.id = "month") %>% 
                gather(key = 'month', value = 'state')
            
        } else {
            # normal procedure
            res %>% 
                bind_rows(.id = "month") %>% 
                gather(-month, key = 'state', value = 'ind') %>% 
                filter(is.na(ind) %>% not) %>% 
                select(month, state)
        }
    }

    states <- 
        solution %>% 
        extract2(state_m) %>% 
        lapply(treat_maint) %>% 
        bind_rows(.id = "resource") %>% 
        filter(state %>% is.na %>% not) %>%
        arrange(resource, month, state) %>% 
        group_by(resource, month) %>% 
        summarise(state = paste0(state, collapse = '+')) %>% 
        ungroup() %>% 
        # merge tasks
        bind_rows(tasks) %>% 
        # start finish format
        collapse_states() %>% 
        # merge colors and format
        inner_join(task_hours) %>% 
        mutate(id= c(1:nrow(.)),
               style= sprintf("background-color:%s;border-color:%s;font-size: %s", color, color, font_size)
        ) %>% 
        left_join(maints_tab %>%  distinct(state) %>% mutate(maint=TRUE)) %>% 
        rename(start= month, end= post_month, group= resource) %>% 
        mutate(maint= maint %>% is.na %>% not,
               content = if_else(maint, state, sprintf('%sh', hours*duration)))
    
    if (style_config$period_num %>% is.null %>% not){
        first <- 
            states %>% distinct(start) %>% 
            unlist %>% min %>% month_to_ymd
        
        states %<>% mutate(start = (interval(first, month_to_ymd(start)) %/% months(1))+1,
                          end = (interval(first, month_to_ymd(end)) %/% months(1))+1
                          )
    }
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
        zoomable= TRUE
    )
    
    timevis(states, groups= groups, options= config, ...)
}

print_solution <- function(exp_directory, max_resources=NULL, state_m='state_m', ...){

    states <- get_states(exp_directory, state_m=state_m)
    
    timevis_from_states(states, max_resources=max_resources, ...)
}

print_solution_and_print <- function(exp_directory, ...){
    result <- print_solution(exp_directory, ...)
    path_out <- paste0(exp_directory, 'solution.html')
    saveWidget(result, file = path_out, selfcontained = FALSE)
}

completed_experiments <- function(path){
    # browser()
    names <- path %>% list.files()
    with_solution <- path %>% 
        list.files(full.names = T) %>% 
        sapply(function(p){
            c('data_in.json', 'data_out.json') %in% list.files(p) %>% all
        })
    names[with_solution] %>% paste0(path, ., '/') %>% set_names(names[with_solution])
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

args = commandArgs(trailingOnly = TRUE)

if (length(args)==1){
    exp_directory = args[1]
    print_solution_and_print(exp_directory, width='100%')
}

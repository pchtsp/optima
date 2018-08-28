library(tidyverse)
library(readxl)
library(magrittr)
library(stringr)
library(zoo)
library(data.table)
library(lubridate)
library(timevis)
library(RColorBrewer)

source('functions/params.R')

replace_if_default <- function(value, replace_value, default=0){
    if_else(value %>% equals(default), replace_value, value)
}
if (FALSE){
    first_year = 2015
    sheet = 'Visu totale'
    # sheet = 'M2D'
    month_table <- 
        data.table(
            month = c('Ja','Fe','Ma','Av','Mi','Jn','Jt','Au','Se','Oc','No','De'),
            month_num = 1:12 %>% str_pad(width = 2, pad="0")
            )
    
    # Get data
    states_hours <- 
        PATHS$data %>% paste0('raw/Planifs M2000.xlsm') %>% 
        read_xlsx(sheet=sheet, skip=1, n_max=166)
    
    # tidy:
    states_hours_n <- 
        states_hours %>% 
        gather(., key='yearmonth', value='state', 5:(length(.)-1)) %>% 
        mutate(year= yearmonth %>% 
                   str_extract('\\d*$') %>% as.numeric %>% 
                   replace_na(0) %>% add(first_year),
               month= yearmonth %>% str_extract('^\\w{2}')) %>% 
        inner_join(month_table) %>%
        mutate(yearmonth = sprintf('%s-%s', year, month_num)) %>% 
        set_names(names(.) %>% make.names) %>%
        select(Aircraft=N., state, yearmonth) %>% 
        mutate(type = Aircraft %>% str_extract('^\\d*\\w')) %>% 
        data.table
    
    # Split in: states, hours
    states_hours_nn <- 
        states_hours_n %>% 
        mutate(hours = state %>% str_extract('^[\\d\\.]*') %>% as.numeric %>% round,
               state = ifelse(is.na(hours), state, 'flight')) %>% 
        filter(yearmonth >= '2018')
    
    hours_important <- 
        states_hours_nn %>% 
        group_by(hours) %>% 
        summarise(num = n()) %>% 
        arrange(-num) %>% slice(1:15)
    
    states_hours_nn %>% semi_join(hours_important) %>% 
        group_by(yearmonth, hours) %>% 
        summarise(num = n()) %>% arrange(yearmonth, -num) %>% filter(hours==70)

    data_train <-  
        states_hours_nn %>% 
        filter(state != "X") %>% 
        group_by(type, yearmonth) %>% 
        mutate(total = n()) %>% 
        group_by(type, yearmonth, total) %>% 
        filter(hours >= 18 & hours <= 24) %>% 
        summarise(count = n()) %>% 
        mutate(date = yearmonth %>% paste0('-01') %>% as.Date,
               perc = count / total * 100)
        
    ggplot(data_train, aes(x= date, y=perc, colour= type)) + geom_line() + scale_x_date() + theme_minimal()
    
}
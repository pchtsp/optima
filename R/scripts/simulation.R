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
               hours = ifelse(hours %>% is.na, 0, hours),
               state = ifelse(hours==1, 'stopped', state),
               state = ifelse(state=='0', 'stopped', state),
               state = ifelse(hours < 30 & hours > 1, 'train', state),
               state = ifelse(hours >= 30, 'OPEX', state),
               state = ifelse(state %>% str_detect('^V\\d+'), 'maint', state),
               state = ifelse(state %>% str_detect('^[xX]$'), 'obsolete', state),
               date = yearmonth %>% paste0('-01') %>% as.Date
               ) %>% 
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
        filter(hours == 'train') %>% 
        summarise(count = n()) %>% 
        mutate(perc = count / total * 100)
        
    ggplot(data_train, aes(x= date, y=perc, colour= type)) + geom_line() + scale_x_date() + theme_minimal()
    
    states_important <- 
        states_hours_nn %>% 
        group_by(state) %>% 
        summarise(num=n()) %>% 
        arrange(-num) %>% 
        slice(1:9)
    
    data_states <- 
        states_hours_nn %>% 
        filter(type=='D') %>%
        group_by(date, state) %>% 
        semi_join(states_important) %>% 
        summarise(total = n())
    
    ggplot(data_states, aes(x=date, y=total)) + geom_area(aes(fill=state)) + 
        scale_fill_brewer(type='qual', palette=3) + theme_minimal()
}
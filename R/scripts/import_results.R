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

collapse_states <- function(table, column_name){
    
    series <- table[[column_name]]
    
    table %>% 
        mutate(temp = series) %>% 
        arrange(UNIT, month) %>% 
        group_by(UNIT) %>% 
        mutate(prev_temp= temp %>% lag,
               first_row= row_number()==1,
               change = temp != prev_temp) %>% 
        filter(first_row | change) %>% 
        mutate(post_month = month %>% lead %>% paste0("-01") %>% ymd() %>% add(0) %>% format("%Y-%m")) %>% 
        select(-temp, -prev_temp, -first_row, -change) %>% 
        ungroup()
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
        collapse_states("state") %>% 
        mutate(id= c(1:nrow(.))) %>% 
        mutate(bucket= state %>% as.integer %>% cut2(g= 5),
               maint = state %>% str_detect('^V\\d+'),
               color= RColorBrewer::brewer.pal(5, "YlOrRd")[bucket],
               color= if_else(maint, "black", color),
               color= if_else(color %>% is.na, '', color),
               style= sprintf("background-color:%s;border-color:%s", color, color)
               ) %>% 
        collapse_states("bucket") %>% 
        select(id, start= month, end= post_month, content = bucket, group= UNIT, style) %>% 
        slice(1:1000)
    
    groups <- data.table(id= states %>% distinct(group) %>% unlist, content= states %>% distinct(group) %>% unlist)
    
    config <- list(
        stack = FALSE,
        editable = TRUE,
        align = "center",
        orientation = "top",
        snap = NULL,
        margin = 0
    )
    
    timevis(states, groups= groups, options= config)
}

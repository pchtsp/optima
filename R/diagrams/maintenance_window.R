library(tidyverse)
library(data.table)
library(timevis)
library(lubridate)
library(visNetwork)
library(magrittr)
library(RColorBrewer)
source('functions/gantt_tasks.R')
source('functions/pdf_gantt.R')

start_month <- as.Date('2017-10-01')
num_res <- 2

data <- data.table(label=c('M', 'M')) %>% 
    mutate(
        month_start=sample(1:5, num_res, replace=TRUE)
        ,month_end= month_start + 6
    ) %>% 
    mutate(start= start_month %m+% months(month_start-1),
           end= start_month %m+% months(month_end),
           group= 1:num_res,
           id= c(1:nrow(.)),
           content = label,
           color = RColorBrewer::brewer.pal(8, "YlOrRd")[rep_len(1:num_res, n())],
           style= sprintf("background-color:%s;border-color:%s;font-size: 15px", color, color),
    )

min_elapsed <- 40
max_elapsed <- 60

window_start <-
    data %>% 
    mutate(month_start= month_end + 1, 
           month_end= month_start + min_elapsed,
           content = '$T_t^m$',
           start= start_month %m+% months(month_start-1),
           end= start_month %m+% months(month_end),
           id= id + max(id),
           color = '#ff0000',
           style= sprintf("background-color:%s;border-color:%s;font-size: 15px", color, color)
           )

window_end <- 
    window_start %>% 
    mutate(month_start= month_end + 1, 
           month_end= month_end + max_elapsed - min_elapsed,
           content = '$T_t^M$',
           start= start_month %m+% months(month_start-1),
           end= start_month %m+% months(month_end),
           id= id + max(id),
           color = '#00ff00',
           style= sprintf("background-color:%s;border-color:%s;font-size: 15px", color, color)
    )
data_f <- bind_rows(list(data, window_start, window_end))

make_tasks_gantt(data_f)

dir_out <- '/home/pchtsp/Documents/projects/COR2019/gantts/'
data_f %>% states_to_pdfgantt() %>% write(file=dir_out %>% paste0('time_windows.tex'))

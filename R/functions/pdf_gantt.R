library(tidyverse)
library(data.table)
library(timevis)
library(lubridate)
library(magrittr)
library(stringr)

states_to_pdfgantt <- function(data, x_unit=NULL, y_unit=NULL, date_format='isodate'){
  # date_format can also be "simple", "isodate", "isodate-yearmonth"
    t_start <- data$start %>% min()
    if (date_format=='isodate-yearmonth'){
      t_end <- data$end %>% max() %>% ym() %>% subtract(dmonths(1)) %>% format("%Y-%m")
    } else {
      t_end <- data$end %>% max() -1  
    }
    y_string <- ''
    x_string <- ''
    if (x_unit %>% is.null){
      x_string <- 'expand chart=\\textwidth,'
    } else {
      x_string <- sprintf('x unit=%scm,', x_unit) 
    }
    if (y_unit %>% is.null %>% not){
      y_string <- sprintf('\ny unit chart=%scm,', y_unit) 
    }
    if (date_format=='simple'){
      time_slot_unit <- ''
      title_calendar <- '\\gantttitlelist{1,...,%s}{1}' %>% sprintf(t_end)
    } else {
      time_slot_unit <- ',\ntime slot unit=month'
      title_calendar <- '\\gantttitlecalendar{year}'
    }
    
    header <- "\\newcommand\\Dganttbar[5]{
      \\ganttbar[#5]{#1}{#3}{#4}\\ganttbar[inline, bar label font=\\color{white}, #5]{#2}{#3}{#4}
    }
    \\begin{ganttchart}[
    %s%s
    hgrid,
    vgrid,
    time slot format=%s%s
    ]{%s}{%s}
    \\ganttset{bar height=0.7}
    \\ganttset{bar top shift=0.15}
    %s \\\\" %>% sprintf(x_string, y_string, date_format, time_slot_unit, t_start, t_end, title_calendar)
    
    colors <- 
        data %>% 
        distinct(color) %>% 
        mutate(key = row_number() %>% paste0('color', .), color2 = str_sub(color, start=2))
    
    pre_head <- sprintf('\\definecolor{%s}{HTML}{%s}', colors$key, colors$color2) %>% paste0(collapse = '\n')
    footer <- '\\end{ganttchart}'
    
    month_to_ymd <- function(x) x %>% paste0("-01") %>% ymd
    
    if (date_format=="simple"){
      data %<>% 
        mutate(end_m1 = end %>% subtract(1))
    } else {
      data %<>% 
        mutate(end_m1 = end %>% month_to_ymd %>% subtract(months(1)) %>% format('%Y-%m'))
    }

    body <- 
        data %>% 
        inner_join(colors) %>% 
        mutate(style_l = sprintf("bar/.append style={fill=%s,draw=none}", key, key),
               line = sprintf('\\Dganttbar{%s}{%s}{%s}{%s}{%s}', group, content, start, end_m1, style_l)
        ) %>% 
        group_by(group) %>% 
        summarise(line = paste0(line, collapse = '\n')) %>% 
        ungroup() %>% 
        summarise(line = paste0(line, collapse = '\\ganttnewline\n')) %>% 
        extract2(1)
    
    result <- paste(c(pre_head, header, body, footer), collapse = "\n")
    return(result)
    
}

states_expanded <- function(data){
  data %>% 
    mutate(end=start + duration-1) %>% 
    # Need to operate by row, so group by row number
    group_by(group,r=row_number()) %>% 
    # Create nested list column containing the sequence for each pair of Start, End values
    mutate(custom = list(start:end)) %>% 
    # Remove the row-number column, which is no longer needed
    ungroup %>% select(-r) %>% 
    # Unnest the list column to get the desired "long" data frame
    unnest(cols=c(custom)) %>% 
    arrange(group, start) %>%
    mutate(content=if_else(state=='M', '-1', state) %>% as.numeric,
           content=if_else(content>=0, content+1, content) %>% as.character,
           end=custom+1) %>% 
    group_by(group, start) %>% 
    mutate(end_lag=lag(end)) %>% 
    ungroup %>% 
    mutate(start=if_else(end_lag %>% is.na, start, end_lag)) %>% 
    # select(group, start, end, end_lag) %>% head
    select(-custom, -end_lag)
}

states_zeros <- function(data_expanded){
  start <- min(data_expanded$start)
  end <- max(data_expanded$end)
  combo <- CJ(group=data_expanded %>% distinct(group) %>% use_series(group),
              start = start:end)
  combo %>% anti_join(data_expanded) %>% 
    mutate(end = start + 1,
           color = 'none', content = '0')
}

if (FALSE){
    # example taken from thesis_gantt.R
    start_month <- as.Date('2017-10-01')
    
    data <- 
        data.table(
            Phase=c(1, 1, 2, 3, 4, 5, 6, 7)
            ,Libelle=c('État de l’art FMP','Travaux antérieurs','Nouveaux modèles',
                       'Incertitude','Cas d’étude','Prototype logiciel','Mémoire de thèse',
                       'Collaboration Dassault')
            ,month_start=c(1, 1, 9, 17, 25, 25, 31, 9)
            ,month_end= c(8, 8, 16, 24, 30, 30, 36, 30)
        ) %>% 
        mutate(start= start_month %m+% months(month_start-1),
               end= start_month %m+% months(month_end),
               group= Phase,
               id= c(1:nrow(.)),
               content = Libelle,
               color = RColorBrewer::brewer.pal(7, "YlOrRd")[rep_len(1:7, n())],
               style= sprintf("background-color:%s;border-color:%s;font-size: 15px", color, color),
        )
    
}
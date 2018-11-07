library(tidyverse)
library(data.table)
library(timevis)
library(lubridate)
library(magrittr)
library(stringr)

states_to_pgfgantt <- function(data, x_unit=NULL, y_unit=NULL, date_format='isodate'){
    t_start <- data$start %>% min()
    t_end <- data$end %>% max()
    if (x_unit %>% is.null){
      x_string <- 'expand chart=\\textwidth,'
    } else {
      x_string <- sprintf('x unit=%scm,', x_unit) 
    }
    
    if (y_unit %>% is.null %>% not){
      y_string <- sprintf('y unit chart=%scm,', y_unit) 
    }
    
    header <- "\\newcommand\\Dganttbar[5]{
      \\ganttbar[#5]{#1}{#3}{#4}\\ganttbar[inline,bar label font=\\tiny\\color{white}\\bfseries, #5]{#2}{#3}{#4}
    }
    \\begin{ganttchart}[
    %s
    %s
    hgrid,
    vgrid,
    time slot format=%s,
    time slot unit=month
    ]{%s}{%s}
    \\gantttitlecalendar{year} \\\\" %>% sprintf(x_string, y_string, date_format, t_start, t_end)
    
    colors <- 
        data %>% 
        distinct(color) %>% 
        mutate(key = row_number() %>% paste0('color', .), color2 = str_sub(color, start=2))
    
    pre_head <- sprintf('\\definecolor{%s}{HTML}{%s}', colors$key, colors$color2) %>% paste0(collapse = '\n')
    footer <- '\\end{ganttchart}'
    
    body <- 
        data %>% 
        inner_join(colors) %>% 
        mutate(style_l = sprintf("bar/.append style={fill=%s}", key),
               line = sprintf('\\Dganttbar{%s}{%s}{%s}{%s}{%s}', group, content, start, end, style_l)
        ) %>% 
        group_by(group) %>% 
        summarise(line = paste0(line, collapse = '\n')) %>% 
        ungroup() %>% 
        summarise(line = paste0(line, collapse = '\\ganttnewline\n')) %>% 
        extract2(1)
    
    result <- paste(c(pre_head, header, body, footer), collapse = "\n")
    return(result)
    
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
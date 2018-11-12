library(tidyverse)
library(data.table)
library(timevis)
library(lubridate)

gantt_thesis <- function(...){
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
    
    config <- list(
        editable = TRUE,
        align = "center",
        orientation = "top",
        snap = NULL,
        margin = 0,
        zoomable= FALSE
    )
    
    groups <- data %>% distinct(group) %>% rename(id= group) %>% mutate(content= id)
    timevis(data, groups= groups, options= config, ...)
}
gantt_thesis() %>% htmlwidgets::saveWidget('test.html')

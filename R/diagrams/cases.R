library(visNetwork)
library(tidyverse)
library(data.table)
library(magrittr)

base_nodes <- paste('Case', 1:5)
names_nodes = c(base_nodes, 'Cases', 'Publié mars', 'Publié avril')

nodes <- data.table(
    label = names_nodes
) %>% 
    mutate(shape = 'icon'
           ,id = row_number()
           # image= images_name
           ,icon.code = if_else(label=='Cases', 'f187', 'f07b')
           ,icon.color= c(rep("black", 6), rep('red', 2))
           ,icon.size = if_else(label=='Cases', 120, 80)
           ,font.size = if_else(label=='Cases', 30, 20)
    )

edges_data <- data.table(
    from = c('Case 1', 'Case 3', 'Case 5', paste('Case', 1:2), "Case 4")
    ,to = c('Publié mars', 'Publié mars', 'Publié avril', paste('Case', 2:3), "Case 5")
    ,arrows= c('', '', '', rep('to', 3))
) %>% 
    bind_rows(CJ(to=base_nodes, from=c('Cases')))


edges <- 
    edges_data %>% 
    inner_join(nodes, by=c('from'='label')) %>% 
    inner_join(nodes, by=c('to'='label')) %>% 
    mutate(from= id.x, to=id.y) %>% 
    mutate(color='lightblue', 
           font.size = 10,
           value = 0.01,
           length = 300)

visNetwork(nodes, edges, height = "800px", width = "800px") %>% 
    visNodes(shapeProperties = list(useBorderWithImage = TRUE))



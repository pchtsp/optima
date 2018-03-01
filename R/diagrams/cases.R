library(visNetwork)
library(tidyverse)
library(data.table)
library(magrittr)

graph_cases <- function(){
    base_nodes <- paste('Cas', 1:5)
    names_nodes = c(base_nodes, 'Cas dépôt', 'Publié mars', 'Publié avril')
    
    nodes <- data.table(
        label = names_nodes
    ) %>% 
        mutate(shape = 'icon'
               ,id = row_number()
               # image= images_name
               ,icon.code = if_else(label=='Cas dépôt', 'f187', 'f07b')
               ,icon.color= c(rep("black", 6), rep('red', 2))
               ,icon.size = if_else(label=='Cas dépôt', 120, 80)
               ,font.size = if_else(label=='Cas dépôt', 30, 20)
        )
    
    edges_data <- data.table(
        from = c('Cas 1', 'Cas 3', 'Cas 5', paste('Cas', 1:2), "Cas 4")
        ,to = c('Publié mars', 'Publié mars', 'Publié avril', paste('Cas', 2:3), "Cas 5")
        ,arrows= c('', '', '', rep('to', 3))
    ) %>% 
        bind_rows(CJ(to=base_nodes, from=c('Cas dépôt')))
    
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
        visNodes(shapeProperties = list(useBorderWithImage = TRUE)) %>% 
        visInteraction(zoomView=FALSE)
    
    
    
}
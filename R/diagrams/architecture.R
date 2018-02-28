library(visNetwork)
library(tidyverse)
library(data.table)
library(magrittr)

# img_path <- "/home/pchtsp/Documents/projects/OPTIMA"
# img_path <- "./.."
names_nodes = c('planner', 'viewer', 'serveur', 'base de données', 'modèle')
# images_name <- c('', 'serveur.png', 'database.png', 'model.png') %>% sprintf('%s/img/%s',img_path, .)

nodes <- data.table(
    label = names_nodes
    ) %>% 
    mutate(shape = "icon", id = row_number()
           # image= images_name
           ,icon.code = c('f007', 'f007', 'f233', 'f1c0', 'f013')
           ,icon.color= c('orange', 'blue', rep("black", 3))
           ,icon.size = 80
           ,font.size = 25
           )
    
edges_data <- data.table(
    from= c('planner', 'serveur', 'serveur', 'serveur', 'serveur', 'serveur', 'serveur')
    ,to= c( 'serveur', 'base de données', 'modèle', 'planner', 'viewer', 'planner', 'serveur')
    ,arrows= c(rep('to', 1), '', '', rep('to', 3), '')
    ,label= c('Excel', 'postgreSQL', 'python', 'R+html5', 'R+html5', 'csv+json', 'python')
)

edges <- 
    edges_data %>% 
    inner_join(nodes, by=c('from'='label')) %>% 
    inner_join(nodes, by=c('to'='label')) %>% 
    mutate(from= id.x, to=id.y) %>% 
    select(-ends_with(".x")) %>% 
    select(-ends_with(".y")) %>% 
    mutate(color='lightblue', 
           font.size = 5,
           value = 0.01,
           length = 300)

visNetwork(nodes, edges, height = "800px", width = "800px") %>% 
    visNodes(shapeProperties = list(useBorderWithImage = TRUE))
    


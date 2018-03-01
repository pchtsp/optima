library(visNetwork)
library(tidyverse)
library(data.table)
library(magrittr)

graph_functionalities <- function(){
    data = 
        list(
            data.table(
                from = 'Gestion de cas'
                ,to = c('Créer cas', 'Copier cas', 'Eliminer cas', 'Publier cas')
                ,color="#abd9e9"
                ,value=1
            ),
            data.table(
                from = 'Rapports'
                ,to = c('Visualisation', 'Comparaison des cas', 'Autres rapports')
                ,color="#abd9e9"
                ,value=1
            ),
            data.table(
                from = "Données d'entrée"
                ,to = c('Editer données', 'Mettre à jour historiques')
                ,color="#abd9e9"
                ,value=1
            ),
            data.table(
                from = "Planification"
                ,to = c('Editer solution', 'Vérificateur automatique')
                ,color="#abd9e9"
                ,value=1
            ),
            data.table(
                from = "Optimisateur"
                ,to = c('Configurer résolution', 'Génerer solution')
                ,color="#abd9e9"
                ,value=1
            ),
            data.table(
                from = "Intégration"
                ,to = c('Importer Excel', 'Exporter fichiers')
                ,color="#abd9e9"
                ,value=1
            ),
            data.table(
                from = 'Application'
                ,to = c('Rapports', 'Gestion de cas', "Données d'entrée", 
                        'Planification', 'Optimisateur', 'Intégration')
                ,shape = "circle"
                ,value=100
            )
        ) 
    
    data %<>% bind_rows()
    # data %<>% extract2(1)
    
    nodes <- 
        data %>% 
        full_join(., select(., from), by=c("to"= 'from')) %>% 
        select(-from) %>% 
        distinct %>% 
        mutate(id= row_number(),
               # shadow = TRUE
        ) %>% 
        rename(label=to)
    
    edges <- 
        data %>% 
        inner_join(nodes, by=c('from'='label')) %>% 
        inner_join(nodes, by=c('to'='label')) %>% 
        select(from= id.x, to=id.y)
    
    # head(nodes)
    
    # edges <- data
    # 
    visNetwork(nodes, edges, height = "800px", width = "800px") %>% 
        visInteraction(zoomView=FALSE)
    # visNetwork(nodes, edges, width = "100%") %>% visHierarchicalLayout
    
    
}
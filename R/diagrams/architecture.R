library(visNetwork)
library(tidyverse)
library(data.table)
library(magrittr)

img_path <- "/home/pchtsp/Documents/projects/OPTIMA"
# img_path <- "./.."
names_nodes = c('pc', 'server', 'db', 'model')
images_name <- c('', 'server.png', 'database.png', 'model.png') %>% sprintf('%s/img/%s',img_path, .)

nodes <- data.table(
    label = names_nodes
    ) %>% 
    mutate(shape = "icon", id = row_number()
           # image= images_name
           ,icon.code = c('f108', 'f233', 'f1c0', 'f013')
           ,icon.color= "black"
           )
    
edges_data <- data.table(
    from= c('pc', 'server', 'server', 'server', 'model'),
    to= c('server', 'db', 'model', 'pc', 'server')
)

    # c('postgreSQL', 'model', 'Excel file', 'Excel template', 'csv / json', 'html')


# data %<>% bind_rows()
# data %<>% extract2(1)

# nodes <- 
#     data %>% 
#     full_join(., select(., from), by=c("to"= 'from')) %>% 
#     select(-from) %>% 
#     distinct %>% 
#     mutate(id= row_number(),
#            # shadow = TRUE
#     ) %>% 
#     rename(label=to)

edges <- 
    edges_data %>% 
    inner_join(nodes, by=c('from'='label')) %>% 
    inner_join(nodes, by=c('to'='label')) %>% 
    select(from= id.x, to=id.y)

# head(nodes)

# edges <- data
# 
visNetwork(nodes, edges, height = "800px", width = "800px") %>% 
    visNodes(shapeProperties = list(useBorderWithImage = TRUE))
    
# visNetwork(nodes, edges, width = "100%") %>% visHierarchicalLayout


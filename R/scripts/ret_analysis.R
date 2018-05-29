library(ggplot2)
library(magrittr)
source('functions/params.R')
source('functions/import_results.R')

exp_directory = PATHS[['experiments']] %>% paste0('201805241334/')
tasks <- get_tasks(exp_directory)
params <- get_parameters(exp_directory)
params <- params$value %>% magrittr::set_names(params$name)
month_range <- seq.Date(from=params['start'] %>% paste0('-01') %>% as_date, 
                        to=params['end'] %>% paste0('-01') %>% as_date,
                        by="months")
task_period <- 
    tasks %>% 
    mutate_at(c('start', 'end'), function(x) x %>% paste0('-01') %>% as_date) %>% 
    merge(data.table(period=month_range)) %>% 
    filter(start<=period & end>= period) %>% 
    group_by(period, type_resource) %>% 
    summarise_at(c('num_resource', 'consumption'), sum)

input_path = exp_directory %>% paste0('data_in.json')
input <- read_json(input_path)
rut_init <- sapply(input$resources, '[[', 'initial_used')
ret_init <- sapply(input$resources, '[[', 'initial_elapsed') %>% data.frame(ret_init=., resource=names(.), stringsAsFactors = F)
type_resource <- sapply(input$resources, '[[', 'capacities') %>% sapply('[[', 1) %>% data.frame(type_resource=., resource=names(.), stringsAsFactors = F)

tot_resource <- type_resource %>% group_by(type_resource) %>% summarise(tot_resource=n())

maintenances <- 
    lapply(
        0:4,
        function(x) {
            data.table(period=month_range) %>% mutate(period_maint = period %m+% months(x))
        }
    ) %>% bind_rows() %>% arrange(period)

ret_info <- 
    ret_init %>% 
    inner_join(type_resource) %>% 
    mutate(start=params['start'] %>% paste0('-01') %>% as_date,
           period = start %m+% months(ret_init - 1)) %>% 
    group_by(type_resource, period) %>% 
    summarise(maint_ret=n()) %>% 
    inner_join(maintenances) %>% 
    mutate(period = period_maint) %>% 
    group_by(type_resource, period) %>% 
    summarise_at(c('maint_ret'), sum)

resource_avail <- 
    task_period %>% 
    left_join(ret_info) %>% 
    inner_join(tot_resource) %>% 
    mutate(maint_ret = replace_na(maint_ret, 0)) %>% 
    mutate(available=tot_resource - maint_ret - num_resource)

ggplot(task_period, aes(x=period, y=num_resource, fill=type_resource)) +
    geom_col() +
    scale_x_date() +
    theme_minimal()

ggplot(resource_avail, aes(x=period, y=available, fill=type_resource)) +
    geom_col() +
    scale_x_date() +
    theme_minimal()

ggplot(resource_avail, aes(x=period, y=maint_ret, fill=type_resource)) +
    geom_col() +
    theme_minimal()

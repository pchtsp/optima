source('functions/params.R')
source('functions/import_results.R')
source('functions/status_resources.R')
source('functions/gantt_tasks.R')
source('functions/pdf_gantt.R')

# main functions ----------------------------------------------------------
exp_directory <- 'C:/Users/pchtsp/Documents/borrar/dell_20190515_all/base/201905151456_2/'
exp_directory = PATHS[['experiments']] %>% paste0('201801131817/')
# PATHS[['experiments']] %>% paste0('201802061201/')
exp_directory = PATHS[['experiments']] %>% paste0('../hp_20181104/base/201811061411/')
exp_directory = PATHS['results'] %>% paste0('hp_20181104/base/201811061411/')
exp_directory = PATHS['results'] %>% paste0('clust1_20181121/tminassign_[1]/201811252202/')
# exp_directory <- PATHS[['results']] %>% paste0('simulated_data/2_task_types/201810051400/')
print_solution(exp_directory, width='100%')
print_tasks(exp_directory)
graph_remaining(exp_directory)

# graph renaming ----------------------------------------------------------

res <- 'A20'
# res <- '85'
res <- '30'

data_input <- treat_data(exp_directory, res)
data_input_n <- 
    data_input %>% 
    rename(Periods=Mois, Time=Temps) %>% 
    mutate(type = if_else(str_detect(type, '^dispo'), 'availability (periods)', type),
           type = if_else(str_detect(type, '^heur'), 'flight hours (hours)', type))

p <- graph_remaining_data(data_input_n, x='Periods', y='Time', return_object=TRUE)
p


# graph window of maintenances --------------------------------------------

m_period <- 
    data_input_n %>% 
    filter(Time==60) %>% 
    arrange(Periods) %>% 
    select(resource=aircraft, month=Periods) %>% 
    mutate(month = month %>% str_sub(end=7), state="M") %>% 
    collapse_states() %>% 
    slice(1) %>% 
    extract2('post_month') %>% 
    paste0("-01") %>% 
    as.POSIXct()
 
x_limits <- m_period %m+% c(months(30-6), months(60-6))

rut <- get_ret(exp_directory, res)

# p + ggplot2::geom_vline(xintercept = ), linetype=2)
p + geom_rect(aes(xmin=x_limits[1], xmax=x_limits[2], ymin=0, ymax=Inf), alpha = .005)
# data_states <- get_states(experiment1)
# periods <- 
# data_states %>% 
#     group_by()

# graph with remaining ret/ rut ----------------------------------------------------

exp_directory <- PATHS[['experiments']] %>% paste0('201810081437/')
exp_directory <- PATHS[['results']] %>% paste0('simulated_data/3_task_types_capa_slack/201810091058/')
resources <- get_states(exp_directory) %>% distinct(group) %>% extract2(1)

data_input <- lapply(resources, function(res){
    treat_data(exp_directory, res, rut_name='rut', ret_name='ret')
}) %>% bind_rows()

data_input_n <-
    data_input %>% 
    select(group=aircraft, start=Mois, value=Temps, type) %>% 
    spread(key='type', value='value') %>% 
    mutate(start= start %>% format('%Y-%m')) %>% 
    arrange(group, start) %>% 
    group_by(group) %>% 
    mutate(ret = lag(ret), rut= lag(rut))
states <- 
    get_states(exp_directory) %>% 
    left_join(data_input_n) %>% 
    mutate(content= ifelse(state=="M",
                           sprintf('%s (%s h /%s p)', state, rut, ret),
                           content))

timevis_from_states(states, max_resources=NULL, width='100%')


# pdf graph ---------------------------------------------------------------
exp_directory = PATHS['results'] %>% paste0('hp_20181104/base/201811061411/')
states <- get_states(exp_directory)

# when we pass period_num we convert periods to integers:
states <- get_states(exp_directory, style_config = list(period_num=TRUE))
resources <- states %>% distinct(group) %>% sample_n(10)
states <- states %>% inner_join(resources) %>% mutate_at(vars(group), as.numeric) %>% arrange(group)
dir_out <- '/home/pchtsp/Documents/projects/COR2019/gantts/'

# -----------graph solution
text <- states_to_pdfgantt(states, y_unit = 0.5, date_format = 'isodate-yearmonth')
text <- states_to_pdfgantt(states, y_unit = 1, date_format = 'isodate-yearmonth')
text <- states_to_pdfgantt(states, y_unit = 1, date_format = 'simple')
write(text, file=dir_out %>% paste0('gantt_5aircraft.tex'))

# we expand the states to bring something close to a matrix:
expanded <- states %>% states_expanded
expanded %>% 
    states_to_pdfgantt(y_unit=1, date_format='simple') %>% 
    write(file=dir_out %>% paste0('gantt_5aircraft_exp.tex'))
expanded %>% states_zeros %>%
    states_to_pdfgantt(y_unit=1, date_format='simple') %>% 
    write(file=dir_out %>% paste0('gantt_5aircraft_exp_0.tex'))


    # -----------other options...
data <- tasks_gantt_data()
data %>% states_to_pdfgantt(x_unit=1, y_unit=0.6) %>% write(file=dir_out %>% paste0('gantt.tex'))

data %>% states_to_pdfgantt(y_unit=0.6) %>% write(file=dir_out %>% paste0('gantt.tex'))


# example of states -------------------------------------------------------

#       id   start     end   content group                                         style
#    <int>   <chr>   <chr>    <fctr> <chr>                                         <chr>
#  1     1 2015-01 2016-01      <NA>  D601               background-color:;border-color:
#  2     4 2016-01 2016-02 [ 0,   4)  D601 background-color:#FFFFB2;border-color:#FFFFB2
#  3     5 2016-02    <NA> [48,1200]  D601 background-color:#BD0026;border-color:#BD0026
#  4    49 2015-01 2016-02 [ 0,   4)  D602 background-color:#FFFFB2;border-color:#FFFFB2
#  5    50 2016-02 2016-03 [ 4,  23)  D602 background-color:#FECC5C;border-color:#FECC5C
#  6    51 2016-03 2016-04 [48,1200]  D602 background-color:#BD0026;border-color:#BD0026
#  7    52 2016-04 2016-05 [24,  48)  D602 background-color:#F03B20;border-color:#F03B20
#  8    53 2016-05 2016-07 [ 4,  23)  D602 background-color:#FECC5C;border-color:#FECC5C
#  9    55 2016-07 2016-08 [24,  48)  D602 background-color:#F03B20;border-color:#F03B20
# 10    56 2016-08 2016-09 [ 4,  23)  D602 background-color:#FECC5C;border-color:#FECC5C

source('functions/params.R')
source('functions/import_results.R')
source('functions/exampleDT.R')
source('functions/gantt_tasks.R')

exp_directory = PATHS[['experiments']] %>% paste0('201809121748/')
print_solution(exp_directory, width='100%')
print_tasks(exp_directory, max_tasks = 8)

PATHS[['experiments']] %>% paste0('201802061201/')

res <- 'A20'

data_input <- treat_data(experiment1, res)
data_input_n <- 
    data_input %>% 
    rename(Periods=Mois, Time=Temps) %>% 
    mutate(type = if_else(str_detect(type, '^dispo'), 'availability (periods)', type),
           type = if_else(str_detect(type, '^heur'), 'flight hours (hours)', type))

graph_remaining_data(data_input_n, x='Periods', y='Time')

# data_states <- get_states(experiment1)
# periods <- 
# data_states %>% 
#     group_by()


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

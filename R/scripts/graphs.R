source('functions/params.R')
source('functions/import_results.R')

exp_directory = PATHS[['experiments']] %>% paste0('201805241334/')
print_solution(exp_directory)

 
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

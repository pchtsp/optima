scenario_get_gaps <- function(all_exp_dir){
    get_gap <- function(path){
        path %>% 
            read_file %>% 
            str_match('gap = [\\d\\.]+\\, ([\\d\\.]+)\\%') %>% 
            extract2(2) %>% 
            as.numeric()
    }
    
    
    scenarios <- all_exp_dir %>% paste0(c('1_task','2_task'))
    all_scenarios <- lapply(scenarios, list.dirs)
    
    lapply(all_scenarios, function(scenario){
        scenario %>% 
            paste0('/results.log') %>% 
            magrittr::extract(., file.exists(.)) %>% 
            sapply(get_gap, USE.NAMES = FALSE)
    })
}

scenario_mean <- function(all_exp_dir){
    scenario_get_gaps(all_exp_dir) %>% 
        lapply(function(scenario){
            scenario %>% 
                mean(na.rm=TRUE) %>% 
                round(2)
        })
    
}
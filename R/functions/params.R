library(magrittr)

path_root = '/home/pchtsp/Documents/projects/'
path_project = path_root %>% paste0("OPTIMA/") 
path_results = '/home/pchtsp/Dropbox/OPTIMA_results/'

PATHS = list(
    root= path_root
    ,results = path_results
    ,experiments = path_results %>% paste0("experiments/")
    ,img = path_root %>% paste0("img/")
    ,latex = path_project %>% paste0("latex/")
    ,r_project = path_project %>% paste0("R/")
    ,data = path_project %>% paste0("data/")
)

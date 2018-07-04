library(magrittr)

path_root = '/home/pchtsp/Documents/projects/'
path_results = '/home/pchtsp/Dropbox/OPTIMA_results/'

PATHS = list(
    root= path_root
    ,results = path_results
    ,experiments = path_results %>% paste0("experiments/")
    ,img = path_root %>% paste0("OPTIMA/img/")
    ,latex = path_root %>% paste0("OPTIMA/latex/")
    ,r_project = path_root %>% paste0("OPTIMA/R/")
)

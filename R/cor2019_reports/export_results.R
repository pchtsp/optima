source('nps_reports/functions.R')
source('cor2019_reports/datasets.R')
source('functions/import_results.R')


# header ------------------------------------------------------------------

opts <- options(knitr.kable.NA = "")
path_export_img <- './../../COR2019/img/'
path_export_img <- '/home/pchtsp/Documents/projects/research/COR2019/img/'
path_export_tab <- './../../COR2019/tables/'
path_export_tab <- '/home/pchtsp/Documents/projects/research/COR2019/tables/'
element_text_size <- 10

# exports -----------------------------------------------------------------
equiv1 <- data.table(scenario=c(paste0("numparalleltasks_", c(2, 3, 4))), 
                     scenario_n=c(sprintf("$|I|=%s$", c(30, 45, 60))))
equiv2 <- data.table(scenario=c(paste0("numperiod_", c(120, 140))), 
                     scenario_n=c(sprintf("$|T|=%s$", c(120, 140))))
equiv <- bind_rows(equiv1, equiv2, data.table(scenario=c('pricerutend_1', 'base'), scenario_n=c('max{rft}=1', 'base')))
equiv <- bind_rows(equiv, data.table(
    scenario=sprintf("minusageperiod_%s", c(5, 15, 20)), 
    scenario_n=sprintf("$U^{min}=%s$", c(5, 15, 20))
    ))




directory <- '/home/pchtsp/f_gdrive/Nextcloud/OPTIMA_documents/results/clust1_20190322/'

get_scenario <- function(scenario='base'){
    directory_n <- paste0(directory, scenario, '/')
    has_data <- function(name) file.exists(paste0(name, 'data_out.json'))
    files <- list.dirs(directory_n) %>% paste0('/') 
    files <- files[files %>%  has_data]
    names(files) <- (files %>% str_match('\\/([\\d_]+)\\/$'))[,2]
    data_states <- files  %>% lapply(get_states, state_m='state') %>% bind_rows(.id='file')
}


summary_table <- function(){
    exps <- c('clust1_20190322', 'clust1_20190322')
    exp_names <- c('MIP', 'MIP2')
    data <- get_generic_compare(exps, exp_names)
    data %>% names
    data_n <- 
        data %>% 
        filter(experiment=='MIP') %>% 
        group_by(scenario) %>% 
        mutate(sizet=n(),
               inf = sol_code==-1,
               no_int= sol_code==0) %>% 
        filter(!inf) %>% 
        summarise(
            `g^avg`=mean(gap),
            `t^min`=min(time),
            `t^avg`=mean(time),
            nonzero=mean(matrix_nonzeros),
            no_int = sum(no_int),
            inf = sizet-n(),
            vars = mean(matrix_variables),
            cons = mean(matrix_constraints)
        ) %>% slice(1)
}

Umin <- function(){
    dataset <- scenarios <- c('base', 'minusageperiod_5', 'minusageperiod_15', 'minusageperiod_20') %>% set_names(., .) %>% lapply(get_scenario)
    dataset_all <- 
        dataset %>% bind_rows(.id='scenario') %>% inner_join(equiv)
    
    num_files <- dataset_all %>% distinct(scenario_n, file) %>% group_by(scenario_n) %>% summarise(num=n())
    data <- dataset_all %>% filter(state=='M') %>% group_by(scenario_n) %>% 
        summarise(maints=n()) %>% 
        inner_join(num_files) %>% 
        mutate(perc_maints=(maints/num) %>% round(2)) %>% arrange(perc_maints) %>% select(Scenario=scenario_n, `$m_{avg}$`=perc_maints)
    data %>% kable(format = 'latex', booktabs = TRUE, escape = FALSE) %>% 
        write_file(path_export_tab %>% paste0('freq_maints_usage.tex'))
}

price_rut_end <- function(){

    dataset <- scenarios <- c('base', 'pricerutend_1') %>% set_names(., .) %>% lapply(get_scenario)
    dataset_n <- dataset %>% bind_rows(.id='scenario')
    
    min_date <- min(dataset_n$start) %>% paste0('-01') %>% as.Date()
    max_date <- max(dataset_n$end) %>% paste0('-01') %>% as.Date()
    months <- 
        seq(from=min_date , to=max_date, "months") %>% 
        as.character() %>% 
        str_sub(end='7')
    
    equiv_dates <- data.table(start=months, start_pos=1:length(months))
    
    dataset_all <- dataset_n %>% inner_join(equiv_dates) %>% inner_join(equiv)

    # histogram of maintenance starts    
    plot <- dataset_all %>% filter(state=='M') %>% 
        ggplot(aes(x=start_pos, fill=scenario_n)) + 
        geom_histogram(position = 'identity', alpha=0.4, bins=15) + ylab('Frequency') + xlab("Starting period of check") +
        theme_minimal() + labs(fill = "Configuration")+ theme(text = element_text(size=15))
    plot + theme(aspect.ratio=0.5) + ggsave(path_export_img %>% paste0('distribution_maints.png'))
    
    dataset_all %>% filter(state=='M') %>%  group_by(scenario) %>% summarise(num_maints=n())

    remaining_rut <- 
        dataset_all %>% filter(state=='M') %>% group_by(scenario, file, group) %>% summarise(lastM=max(start_pos)) %>% 
        right_join(dataset_all) %>% filter(start_pos>=lastM) %>% group_by(scenario_n, file) %>% 
        summarise(total=1000-sum(duration*hours)/15)
    
    # remaining_rut %>% group_by(scenario) %>% summarise(n())
    
    # histogram of final states
    plot <- remaining_rut %>% 
        ggplot(aes(x=total, fill=scenario_n)) + 
        geom_histogram(position = 'identity', alpha=0.4, bins=10) + ylab('Frequency') + 
        xlab("Average remaining flight time for the fleet at the end of planning horizon") + labs(fill="Configuration") + 
        theme_minimal() + theme(text = element_text(size=15))
    plot
    plot + ggsave(path_export_img %>% paste0('hist_final_states.png'))
}

# data wrangling ----------------------------------------------------------

if (FALSE){
    exps <- c('clust1_20190322', 'clust1_20190322')
    exp_names <- c('MIP', 'MIP2')
    scenarios <- c('base', 'numparalleltasks_2', 'numparalleltasks_3', 'numparalleltasks_4',
                   'numperiod_120', 'numperiod_140', 'pricerutend_1')
    data <- get_generic_compare(exps, exp_names, scenario_filter = scenarios)
    data_n <- 
        data %>% 
        filter(experiment=='MIP') %>% 
        full_join(equiv) %>% 
        mutate(scenario = ifelse(scenario_n %>% is.na, scenario, scenario_n))
    data_n %>% names
    data_n %>% 
        ggplot(aes(y=scenario, x=gap)) + geom_boxplot() + theme_minimal()+
        xlab('Relative gap') + ylab('Scenario') + scale_y_discrete(labels=TeX(data_n$scenario)) +
        theme(text = element_text(size=20))
}


python3 python/scripts/exec.py -d "{\"solver\": \"GUROBI\", \"path\": \"/home/disc/f.peschiera/Documents/projects/optima/results/test/\"}"

python3 python/scripts/exec_iteratively.py -p "{\"results\": \"/home/disc/f.peschiera/Documents/projects/optima/results/clust1_20181024/\"}" -d "{\"solver\": \"CPLEX\"}" > log_20181024.txt &

python3 python/scripts/exec_iteratively.py -p "{\"results\": \"/home/pchtsp/Dropbox/OPTIMA_results/hp_2018110777/\"}" -d "{\"solver\": \"CPLEX\", \"num_period\": 10}" -s "{\"seed\": 42}" > log.txt &


# clust1_20181112
python3 python/scripts/exec_iteratively.py -p "{\"results\": \"/home/disc/f.peschiera/Documents/projects/optima/results/clust1_20181112/\"}" -d "{\"solver\": \"GUROBI\", \"num_period\": 90}" -s "{\"seed\": 42}" > log.txt &

# hp_20181114:
python3 python/scripts/exec_iteratively.py -p "{\"results\": \"/home/pchtsp/Dropbox/OPTIMA_results/hp_20181114/\"}" -d "{\"solver\": \"CBC\", \"num_period\": 90, \"integer\": true}" -s "{\"seed\": 42}" -c "{\"maint_duration\": [4, 8], \"perc_capacity\": [0.1, 0.2], \"max_used_time\": [800, 1200], \"max_elapsed_time\": [40, 80], \"elapsed_time_size\": [20, 40]}"  -q 10 

# hp_20181119
python3 python/scripts/exec_iteratively.py -p "{\"results\": \"/home/pchtsp/Dropbox/OPTIMA_results/hp_20181119/\"}" -d "{\"solver\": \"CHOCO\", \"num_period\": 90, \"integer\": true}" -s "{\"seed\": 42}" -c "{\"maint_duration\": [4, 8], \"perc_capacity\": [0.1, 0.2], \"max_used_time\": [800, 1200], \"max_elapsed_time\": [40, 80], \"elapsed_time_size\": [20, 40]}"  -q 10 

# clust1_20181121: 
python3 python/scripts/exec_iteratively.py -p "{\"results\": \"results/clust1_20181121/\"}" -d "{\"solver\": \"CPLEX\", \"num_period\": 90}" -s "{\"seed\": 42}" -c "{\"num_period\": [120, 140] , \"num_parallel_tasks\": [2, 3, 4] , \"min_usage_period\": [5, 15, 20] , \"min_avail_percent\": [0.05, 0.2] , \"min_avail_value\": [2, 3] , \"min_hours_perc\": [0.3, 0.7] , \"t_min_assign\": [[1], [3], [6]], \"price_rut_end\": [1]}"  -q 10 > log.txt &

# clust1_20181128: 
python3 python/scripts/exec_iteratively.py -p "{\"results\": \"results/clust1_20181128/\"}" -d "{\"solver\": \"CPLEX\", \"num_period\": 90}" -s "{\"seed\": 52}" -c "{\"num_period\": [120, 140] , \"num_parallel_tasks\": [2, 3, 4] , \"min_usage_period\": [5, 15, 20] , \"min_avail_percent\": [0.05, 0.2] , \"min_avail_value\": [2, 3] , \"min_hours_perc\": [0.3, 0.7] , \"t_min_assign\": [[1], [3], [6]], \"price_rut_end\": [1]}"  -q 40 > log.txt &

# clust1_20181128 (2): 
python3 python/scripts/exec_iteratively.py -p {"results": "./../../optima_results/clust1_20181128/"} -d {"solver": "CPLEX", "num_period": 90} -s {"seed": 52} -c {"num_parallel_tasks": [3, 4] , "min_usage_period": [5, 15, 20] , "min_avail_percent": [0.05, 0.2] , "min_avail_value": [2, 3] , "min_hours_perc": [0.3, 0.7] , "t_min_assign": [[1], [3], [6]], "price_rut_end": [1]} -q 40 > log.txt &

# clust1_20181128 (3): 
nohup python3 python/scripts/exec_iteratively.py -p "{\"results\": \"./../../optima_results/clust1_20181128/\"}" -d "{\"solver\": \"CPLEX\", \"num_period\": 90}" -s "{\"seed\": 52}" -c "{\"min_avail_value\": [3] , \"min_hours_perc\": [0.3, 0.7] , \"t_min_assign\": [[1], [3], [6]], \"price_rut_end\": [1]}"  -q 40 > log.txt &

# clust1_20181128 (4): 
nohup python3 python/scripts/exec_iteratively.py -p "{\"results\": \"/tmp/f.peschiera/optima_results/clust1_20181128/\"}" -d "{\"solver\": \"CPLEX\", \"num_period\": 90}" -s "{\"seed\": 52}" -c "{\"min_avail_value\": [3] , \"min_hours_perc\": [0.3, 0.7] , \"t_min_assign\": [[1], [3], [6]], \"price_rut_end\": [1]}"  -q 40 > /tmp/f.peschiera/log.txt &

# clust1_20190108
nohup python3 python/scripts/exec_iteratively.py -p "{\"results\": \"/tmp/f.peschiera/optima_results/clust1_20190108/\"}" -d "{\"solver\": \"CBC\", \"num_period\": 90}" -s "{\"seed\": 42}" -q 10 > /tmp/f.peschiera/log.txt &

# clust1_20190114
nohup python3 python/scripts/exec_iteratively.py -p "{\"results\": \"/tmp/f.peschiera/optima_results/clust1_20190114/\"}" -d "{\"solver\": \"CPLEX\", \"num_period\": 90, \"timeLimit\": 0, \"writeMPS\": \"True\"}" -s "{\"seed\": 42}" -q 10 > /tmp/f.peschiera/log.txt &

# clust1_20190115
nohup python3 python/scripts/exec_iteratively.py -p "{\"results\": \"/tmp/f.peschiera/optima_results/clust1_20190115/\"}" -d "{\"solver\": \"CPLEX\", \"num_period\": 90, \"timeLimit\": 0, \"writeMPS\": \"True\"}" -s "{\"seed\": 42}" -c "{\"min_usage_period\": [5, 15, 20]}" -q 10 > /tmp/f.peschiera/log.txt &

# get files back:

rsync -rav -e ssh --include '*/' --exclude='formulation.lp' f.peschiera@serv-cluster1:/home/disc/f.peschiera/Documents/projects/optima/results/* ./

# cplex tuner

ls -d -1 /tmp/f.peschiera/optima_results/clust1_20190114/base/*/formulation.mps > modelfile
ls -d -1 /tmp/f.peschiera/optima_results/clust1_20190115/minusageperiod_5/*/formulation.mps > modelfile
nohup python3 ./python/scripts/cplex_tuner.py > /tmp/f.peschiera/log_20190115.txt &

# deploy src

mkdir C:\Users\pchtsp\Downloads\src
git archive --format=tar HEAD python | (cd C:/Users/pchtsp/Downloads/src/ && tar xf -)
git archive --format=tar HEAD R/functions/import_results.R | (cd C:/Users/pchtsp/Downloads/src/ && tar xf -)
git archive --format=tar HEAD data/template | (cd C:/Users/pchtsp/Downloads/src/ && tar xf -)
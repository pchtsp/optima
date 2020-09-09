import pstats
p = pstats.Stats('scripts/restats')
p.strip_dirs().sort_stats(-1)
p.sort_stats('cumulative').print_stats(20)  # tottime, cumulative


# check_swap_size
# check_sequence
# try_change_node
# python -m cProfile -o scripts/restats execution/exec.py -pr C:/Users/Documents/projects/ROADEF2018/ -rr C:/Users/pchtsp/Dropbox/ROADEF2018/
# python -m cProfile -o scripts/restats execution/exec.py
# python -m cProfile -o scripts/restats scripts/main.py
# python -m cProfile -o scripts/restats execution/stochastic_analysis.py
# python -m cProfile -o scripts/restats solvers/heuristic_graph.py

# for line profiling:
# kernprof -l execution/exec.py
# kernprof -l scripts/main.py
# python -m line_profiler exec.py.lprof
# python -m line_profiler main.py.lprof
# kernprof -l scripts/main.py -p "{\"experiments\": \"/home/pchtsp/Documents/projects/optima_results_old/experiments/\"}" -mm 0.8


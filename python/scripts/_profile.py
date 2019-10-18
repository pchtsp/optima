import pstats
p = pstats.Stats('scripts/restats')
p.strip_dirs().sort_stats(-1)
p.sort_stats('tottime').print_stats(20)  # tottime, cumulative


# check_swap_size
# check_sequence
# try_change_node
# python -m cProfile -o scripts/restats scripts/exec.py -p "{\"experiments\": \"C:/Users/pchtsp/Documents/borrar/experiments/\", \"results\": \"C:/Users/pchtsp/Documents/borrar/\"}"
# python -m cProfile -o scripts/restats scripts/exec.py

# for line profiling:
# kernprof -l scripts/exec.py
# kernprof -l scripts/exec.py -p "{\"experiments\": \"C:/Users/pchtsp/Documents/borrar/experiments/\", \"results\": \"C:/Users/pchtsp/Documents/borrar/\"}"
# python -m line_profiler exec.py.lprof
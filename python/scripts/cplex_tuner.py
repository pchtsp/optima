import subprocess

# modelfile is a file where each line has the path to a mps file.
# defaults.prm is where the result is written.

cplex = subprocess.Popen('cplex', stdin = subprocess.PIPE)
cplex_cmds = "set timeLimit 18000\n"
cplex_cmds += "set tune timeLimit 3600\n"
cplex_cmds += "tools tune modelfile defaults.prm\n"
cplex_cmds += "quit\n"
cplex_cmds = cplex_cmds.encode('UTF-8')
cplex.communicate(cplex_cmds)


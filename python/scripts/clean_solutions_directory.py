import package.tests as exp

path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments"
exps = exp.clean_experiments(path, regex=r'^201', clean=False)

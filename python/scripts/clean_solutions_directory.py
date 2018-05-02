import package.tests as exp
import package.params as pm
import pprint as pp

exps = exp.clean_experiments(pm.PATHS['experiments'], regex=r'^201804', clean=False)
pp.pprint(exps)
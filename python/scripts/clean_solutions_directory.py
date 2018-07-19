import package.experiment as exp
import package.params as pm
import pprint as pp
import shutil
import os
import package.params as pm

exps = exp.clean_experiments(pm.PATHS['experiments'], regex=r'^201804', clean=False)
pp.pprint(exps)


# paths_exps = [os.path.join(pm.PATHS['experiments'], k) for k, v in options_e.items() if v['solver']=='CPO' and v['timeLimit']==600]
# for ed in paths_exps:
#     try:
#         shutil.rmtree(ed)
#     except:
#         print(ed)
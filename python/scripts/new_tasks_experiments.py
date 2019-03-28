import scripts.COR2019 as cor
import package.reports as rep
import pandas as pd

results = {}

for name, exp in [('original', 'dell_20190326'), ('new', 'dell_20190327')]:
    table = rep.get_simulation_results(exp)
    results[name] = rep.summary_table(table)

results = pd.concat(results)
html = results.to_html()
file_path = r'\\luq\franco.peschiera.fr$\MyDocs\graphs\result.html'
with open(file_path, 'w') as f:
    f.write(html)

# cor.statistics_experiment(experiment)

# scenarios = \
#     dict(
#         MIN_HOUR_5="dell_20190327_2/numperiod_90",
#         MIN_HOUR_15="dell_20190327_2/numperiod_90",
#         MIN_HOUR_20="dell_20190327_2/numperiod_90",
#         BASE="clust_params2_cplex/base",
#     )
#
# cor.get_scenarios_to_compare(scenarios)


import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import scripts.exec as exec
import json
import argparse

def re_make_paths(path_to_experiment, scenario_instances):
    scenario_paths = {s: os.path.join(path_to_experiment, s) for s in scenario_instances}
    instances_paths = {s: {i: os.path.join(scenario_paths[s], i) for i in instances}
                       for s, instances in scenario_instances.items()}
    return scenario_paths, instances_paths


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='(re)solve an instance MFMP.')
    parser.add_argument('-d', '--options', dest='config_dict', type=json.loads)
    parser.add_argument('-e', '--experiment', dest='experiment', required=True)

    args = parser.parse_args()

    path_to_experiment_in = args.experiment

    scenarios = os.listdir(path_to_experiment_in)
    scenario_paths = {s: os.path.join(path_to_experiment_in, s) for s in scenarios}
    scenario_instances = {s: os.listdir(v) for s, v in scenario_paths.items()}

    # path structure for instances
    path_to_experiment_out = path_to_experiment_in + '_remake'
    if not os.path.exists(path_to_experiment_out):
        os.mkdir(path_to_experiment_out)
    scenario_paths_in, instances_paths_in = re_make_paths(path_to_experiment_in, scenario_instances)
    scenario_paths_out, instances_paths_out = re_make_paths(path_to_experiment_out, scenario_instances)

    for scenario, instances_dict in instances_paths_in.items():
        if not os.path.exists(scenario_paths_out[scenario]):
            os.mkdir(scenario_paths_out[scenario])
        for instance, path in instances_dict.items():
            new_options = {}
            if args.config_dict:
                new_options = args.config_dict
            new_options['path'] = instances_paths_out[scenario][instance]
            exec.re_execute_instance(path, new_options)

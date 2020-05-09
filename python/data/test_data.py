import os

this_file = os.path.realpath(__file__)
parent_dir = os.path.dirname(this_file)

class DataSet(object):

    def __init__(self, input_data, solution_data=None, options=None):
        self.input_data = input_data
        self.solution_data = solution_data
        self.options = options
        pass

    def get_instance(self):
        return self.input_data

    def get_solution(self):
        return self.solution_data

    def get_options(self):
        return self.options

    @classmethod
    def from_directory(cls, rel_path):
        abs_path = os.path.join(parent_dir, rel_path)
        input_data = get_file_path(os.path.join(abs_path, 'data_in.json'))
        solution_data = get_file_path(os.path.join(abs_path, 'data_out.json'))
        options = get_file_path(os.path.join(abs_path, 'options.json'))
        options['path'] = abs_path
        return cls(input_data, solution_data, options)

def dataset1():
    maints = {
        'M': {
            'duration_periods': 4
            , 'capacity_usage': 1
            , 'max_used_time': 1000
            , 'max_elapsed_time': 60
            , 'elapsed_time_size': 4
            , 'used_time_size': 50
            , 'type': '1'
            , 'depends_on': []
        }
        , 'VG': {
            'duration_periods': 1
            , 'capacity_usage': 3
            , 'max_used_time': None
            , 'max_elapsed_time': 8
            , 'elapsed_time_size': 2
            , 'used_time_size': None
            , 'type': '2'
            , 'depends_on': ['M']
        }
        , 'VI': {
            'duration_periods': 1
            , 'capacity_usage': 6
            , 'max_used_time': None
            , 'max_elapsed_time': 16
            , 'elapsed_time_size': 2
            , 'used_time_size': None
            , 'type': '2'
            , 'depends_on': ['M']
        }
        , 'VS': {
            'duration_periods': 1
            , 'capacity_usage': 4
            , 'max_used_time': 600
            , 'max_elapsed_time': None
            , 'elapsed_time_size': None
            , 'used_time_size': 200
            , 'type': '2'
            , 'depends_on': ['M']
        }}
    model_data = {
        'parameters': {
            'start': '2018-01'
            , 'num_period': 60
            , 'min_usage_period': 15
        },
        'resources': {
            '1': {
                'initial': {
                    'M': {'used': 500, 'elapsed': 30}
                    , 'VG': {'elapsed': 4, 'used': None}
                    , 'VI': {'elapsed': 8, 'used': None}
                    , 'VS': {'used': 300, 'elapsed': None}
                }
                , 'min_usage_period': {'default': 20, '2018-05': 4}
            }
        },
        'maintenances': maints,
        'tasks': {},
        'maint_types': {'1': {'capacity': {'2018-05': 0}}, '2': {'capacity': {'2018-05': 0}}}
    }
    return model_data

def dataset2():
    maints = {
        'M': {
            'duration_periods': 4
            , 'capacity_usage': 1
            , 'max_used_time': 1000
            , 'max_elapsed_time': 60
            , 'elapsed_time_size': 4
            , 'used_time_size': 50
            , 'type': '1'
            , 'depends_on': []
            , 'priority': 1
        }
        , 'VG': {
            'duration_periods': 1
            , 'capacity_usage': 3
            , 'max_used_time': None
            , 'max_elapsed_time': 8
            , 'elapsed_time_size': 2
            , 'used_time_size': None
            , 'type': '2'
            , 'depends_on': ['M']
            , 'priority': 20
        }
        , 'VI': {
            'duration_periods': 1
            , 'capacity_usage': 6
            , 'max_used_time': None
            , 'max_elapsed_time': 16
            , 'elapsed_time_size': 2
            , 'used_time_size': None
            , 'type': '2'
            , 'depends_on': ['M']
            , 'priority': 5
        }
        , 'VS': {
            'duration_periods': 1
            , 'capacity_usage': 4
            , 'max_used_time': 600
            , 'max_elapsed_time': None
            , 'elapsed_time_size': None
            , 'used_time_size': 200
            , 'type': '2'
            , 'depends_on': ['M']
            , 'priority': 2
        }}
    model_data = {
        'parameters': {
            'start': '2018-01'
            , 'num_period': 10
            , 'min_usage_period': 15
            , 'maint_capacity': 4
            , 'default_type2_capacity': 66
        },
        'resources': {
            '1': {
                'initial': {
                    'M': {'used': 500, 'elapsed': 30}
                    , 'VG': {'elapsed': 4, 'used': None}
                    , 'VI': {'elapsed': 8, 'used': None}
                    , 'VS': {'used': 300, 'elapsed': None}
                }
                , 'min_usage_period': {'default': 20, '2018-05': 4}
                , 'states': {}
            }
        },
        'maintenances': maints,
        'tasks': {},
        'maint_types': {'1': {'capacity': {'2018-05': 0}}, '2': {'capacity': {'2018-05': 0}}}
    }
    return model_data

def solution2():
    return {
        'state_m':
            {'1': {'2018-07': {'VI': 1}, '2018-03': {'VG': 1}}}
    }

def dataset3():
    maints = {
        'M': {
            'duration_periods': 6
            , 'capacity_usage': 1
            , 'max_used_time': 1000
            , 'max_elapsed_time': 60
            , 'elapsed_time_size': 30
            , 'used_time_size': 500
            , 'type': '1'
            , 'depends_on': []
            , 'priority': 1
        }}
    missions = {
            '1': {
                'start': '2018-02'
                , 'end': '2018-07'
                , 'consumption': 30
                , 'num_resource': 1
                , 'type_resource': 0
                , 'min_assign': 3
                , 'capacities': [0]
            }
        }
    model_data = {
        'parameters': {
            'start': '2018-01'
            , 'num_period': 10
            , 'min_usage_period': 5
            , 'maint_capacity': 4
        },
        'resources': {
            '1': {
                'initial': {
                    'M': {'used': 500, 'elapsed': 30}
                }
                , 'min_usage_period': {'default': 5, '2018-05': 4}
                , 'states': {}
                , 'type': 0
                , 'capacities': [0]
            }
        },
        'maintenances': maints,
        'tasks': missions,
        'maint_types': {'1': {'capacity': {'2018-05': 0}}}
    }
    return model_data


def dataset3_no_default():
    data = dataset3()
    data['parameters']['min_usage_period'] = 0
    data['resources']['1'].pop('min_usage_period')
    return data


def dataset3_no_default_5_periods():
    data = dataset3_no_default()
    data['parameters']['num_period'] = 5
    data['tasks']['1']['end'] = '2018-05'
    return data


def dataset4():
    maints = {
        'M': {
            'duration_periods': 6
            , 'capacity_usage': 1
            , 'max_used_time': 1000
            , 'max_elapsed_time': 60
            , 'elapsed_time_size': 10
            , 'used_time_size': 500
            , 'type': '1'
            , 'depends_on': []
            , 'priority': 1
        }}
    resources = {
            '1': {
                'initial': {
                    'M': {'used': 500, 'elapsed': 30}
                }
                , 'states': {}
                , 'type': 0
                , 'capacities': [0]
            },
            '2': {
                'initial': {
                    'M': {'used': 700, 'elapsed': 40}
                }
                , 'states': {}
                , 'type': 0
                , 'capacities': [0]
            }
        }
    missions = {
            'O1': {
                'start': '2018-02'
                , 'end': '2018-07'
                , 'consumption': 30
                , 'num_resource': 1
                , 'type_resource': 0
                , 'min_assign': 3
                , 'capacities': [0]
            },
            'O2': {
                'start': '2018-08'
                , 'end': '2019-06'
                , 'consumption': 70
                , 'num_resource': 1
                , 'type_resource': 0
                , 'min_assign': 4
                , 'capacities': [0]
            },
            'O3': {
                'start': '2019-07'
                , 'end': '2020-12'
                , 'consumption': 30
                , 'num_resource': 1
                , 'type_resource': 0
                , 'min_assign': 3
                , 'capacities': [0]
            }
        }
    model_data = {
        'parameters': {
            'start': '2018-01'
            , 'num_period': 36
            , 'min_usage_period': 0
            , 'maint_capacity': 4
        },
        'resources': resources,
        'maintenances': maints,
        'tasks': missions,
        'maint_types': {}
    }
    return model_data

def dataset5():
    return DataSet.from_directory('cases/202003121542')

def dataset6():
    return DataSet.from_directory('cases/202003231502')

def get_file_path(abs_path):
    from .data_input import load_data
    data_dir = abs_path
    return load_data(data_dir)




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
    model_data = {
        'parameters': {
            'start': '2018-01'
            , 'num_period': 10
            , 'min_usage_period': 5
            , 'maint_capacity': 4
            , 'default_type2_capacity': 66
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
        'tasks': {
            'O1': {
                'start': '2018-02'
                , 'end': '2018-07'
                , 'consumption': 30
                , 'num_resource': 1
                , 'type_resource': 0
                , 'min_assign': 3
                , 'capacities': [0]
            }
        },
        'maint_types': {'1': {'capacity': {'2018-05': 0}}}
    }
    return model_data


def solution2():
    return {
        'state_m':
            {'1': {'2018-07': {'VI': 1}, '2018-03': {'VG': 1}}}
    }
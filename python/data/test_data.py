

def dataset1():
    maints = {
        'M': {
            'duration_periods': 4
            , 'capacity_usage': 1
            , 'max_used_time': 1000
            , 'max_elapsed_time': 60
            , 'elapsed_time_size': 4
            , 'used_time_size': 1000
            , 'type': '1'
            , 'depends_on': []
        }
        , 'VG': {
            'duration_periods': 1
            , 'capacity_usage': 3
            , 'max_used_time': None
            , 'max_elapsed_time': 8
            , 'elapsed_time_size': 3
            , 'used_time_size': None
            , 'type': '2'
            , 'depends_on': ['M']
        }
        , 'VI': {
            'duration_periods': 1
            , 'capacity_usage': 6
            , 'max_used_time': None
            , 'max_elapsed_time': 16
            , 'elapsed_time_size': 3
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
            , 'used_time_size': 50
            , 'type': '2'
            , 'depends_on': ['M']
        }}
    model_data = {
        'parameters': {
            'start': '2018-01'
            , 'num_period': 30
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
                , 'min_usage_period': {'default': 4, '2018-05': 4}
            }
        },
        'maintenances': maints,
        'tasks': {},
        'maint_types': {'1': {'capacity': {'2018-05': 0}}, '2': {'capacity': {'2018-05': 0}}}
    }
    return model_data
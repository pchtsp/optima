import pandas as pd
import data.dates as aux
import numpy as np
import pytups.superdict as sd
import pytups.tuplist as tl


def get_parameters(tables):
    data = tables['params'].set_index('Parametre')['Valeur'].to_dict()
    data['end'] = aux.shift_month(data['start'], data['num_period'] - 1)
    return data


def get_equiv_names():
    return {
        'duree': 'duration_periods'
        , 'BC': 'max_elapsed_time'
        , 'BC_tol': 'elapsed_time_size'
        , 'BH': 'max_used_time'
        , 'BH_tol': 'used_time_size'
        , 'capacit_utili': 'capacity_usage'
        , 'avion': 'resource'
        , 'mois_derniere': 'elapsed'
        , 'heures_derniere': 'used'
        , 'heures': 'hours'
        , 'capacite': 'capacity'
        , 'mois': 'period'
        , 'mission': 'task'
        , 'date_debut': 'start'
        , 'date_fin': 'end'
        , 'type_avion': 'type_resource'
        , 'nb_avions': 'num_resource'
        , 'nb_heures': 'consumption'
    }


def get_maintenance(tables):
    equiv = get_equiv_names()
    maint_tab  =\
        tables['maintenances']. \
        rename(columns=equiv). \
        replace({np.nan: None})

    maint_tab.type = maint_tab.type.astype(str)

    data = \
        maint_tab. \
            set_index('maint'). \
            to_dict(orient='index')
    for k, v in data.items():
        if v['depends_on']:
            v['depends_on'] = [v['depends_on']]
        else:
            v['depends_on'] = []
    return data

def get_tasks(tables):
    """
    :param tables:
    :return: dictionary in mission format
    """
    if 'missions' not in tables:
        return sd.SuperDict()
    tasks = \
        tables['missions'].\
        rename(columns=get_equiv_names()).\
        set_index('task')
    tasks['matricule'] = ''
    task_dict = sd.SuperDict.from_dict(tasks.to_dict(orient='index'))
    capacities = \
        task_dict.\
        vapply(lambda v: sd.SuperDict(capacities=[v['type_resource']]))
    _dif= lambda x, y: aux.get_months(x, y)
    min_assign = \
        task_dict.\
        vapply(lambda x: _dif(x['start'], x['end'])).\
        vapply(lambda v: sd.SuperDict(min_assign=len(v)+1))
    return task_dict.update(capacities).update(min_assign)

def get_resources(tables):

    params = get_parameters(tables)
    start = params['start']

    equiv = get_equiv_names()

    states = tables['etats_initiaux'].rename(columns=equiv)
    maint_tab = tables['maintenances']

    def elapsed_time_between_dates(value, series2):
        return pd.Series(len(aux.get_months(p2, value)) - 1 if not pd.isna(p2) else np.nan
                         for p2 in series2)

    resources = tables['avions'].rename(columns=equiv)
    resources.resource = resources.resource.astype(str)
    states.resource = states.resource.astype(str)

    # Flight hour consumption
    flight_hours_dict = {}
    min_usage_period = {}
    if 'heures_vol' in tables:
        flight_hours = tables['heures_vol'].rename(columns=equiv)
        flight_hours.resource = flight_hours.resource.astype(str)
        flight_hours_dict = flight_hours.set_index(['resource', 'period'])['min_usage_period'].to_dict()
        flight_hours_dict = sd.SuperDict.from_dict(flight_hours_dict).to_dictdict()

    if 'min_usage_period' in resources.columns:
        min_usage_period = resources.set_index('resource')['min_usage_period'].to_dict()
        min_usage_period = \
            sd.SuperDict.from_dict(min_usage_period).\
                clean(func=lambda v: not np.isnan(v)).\
                vapply(lambda v: {'default': v})
        min_usage_period.update(flight_hours_dict)
        min_usage_period = min_usage_period.vapply(lambda v: {'min_usage_period': v})

    # Initial conditions
    resources_initial = \
        pd.merge(resources, states, on='resource'). \
        merge(maint_tab[['maint', 'BC', 'BH']], on='maint'). \
        assign(used = lambda x: x.hours - x.used). \
        assign(elapsed=lambda x: x.elapsed.str.slice(stop=7)). \
        assign(elapsed=lambda x: elapsed_time_between_dates(start, x.elapsed)). \
        assign(elapsed=lambda x: x.BC - x.elapsed). \
        assign(used=lambda x: x.BH - x.used). \
        filter(['resource', 'maint', 'used', 'elapsed']). \
        assign(initial='initial'). \
        replace({np.nan: None}).\
        set_index(['resource', 'initial', 'maint']). \
        to_dict(orient='index')

    resources_tot = sd.SuperDict.from_dict(resources_initial).to_dictdict()

    # Aircraft type
    if 'type' in resources:
        resources.type = resources.type.astype(str)
        aircraft_type = resources.set_index('resource')['type'].to_dict()
        aircraft_type = \
            sd.SuperDict.from_dict(aircraft_type).\
            vapply(lambda v: sd.SuperDict(type=v, capacities=[v]))
    else:
        _default = sd.SuperDict(type=1, capacities=[1])
        aircraft_type = \
            resources_tot.\
                vapply(lambda v: _default)

    resources_tot.update(min_usage_period)
    resources_tot.update(aircraft_type)

    def_resources = {'states': {}, 'code': ''}
    for r in resources_tot:
        resources_tot[r].update(def_resources)

    return resources_tot

def get_maint_types(tables):

    equiv = get_equiv_names()
    maint_type_tab = tables['maint_capacite'].rename(columns=equiv)
    maint_type_tab.type = maint_type_tab.type.astype(str)
    maint_type = maint_type_tab.set_index(['type', 'period']).to_dict(orient='index')
    maint_type = sd.SuperDict.from_dict(maint_type).to_dictdict()
    maint_type = maint_type.to_dictup().to_tuplist().\
        to_dict(result_col=3, indices=[0, 2, 1], is_list=False).to_dictdict()
    return maint_type

def import_input_template(path):
    """
    :param path:
    :return: the data needed to create an instance
    """

    sheets = ['maintenances', 'etats_initiaux', 'avions', 'params',
              'heures_vol', 'maint_capacite', 'missions']

    xl = pd.ExcelFile(path)
    present_sheets = set(sheets) & set(xl.sheet_names)

    tables = {sh: xl.parse(sh) for sh in present_sheets}
    data = dict(
        parameters = get_parameters(tables)
        , tasks = get_tasks(tables)
        , maintenances = get_maintenance(tables)
        , resources = get_resources(tables)
    )

    if 'maint_capacite' in present_sheets:
        data['maint_types'] = get_maint_types(tables)

    return data

def export_input_template(path, data):
    """
    :param path:
    :param data: the instance.data part, dictionary
    :return: nothing
    """
    equiv = sd.SuperDict(get_equiv_names()).reverse()
    equiv['index'] = 'maint'

    for k, v in data['maintenances'].items():
        v['depends_on'].remove(k)
        if len(v['depends_on']):
            v['depends_on'] = v['depends_on'][0]
        else:
            v['depends_on'] = ""

    # maintenances
    maint_tab = \
        pd.DataFrame.from_dict(data['maintenances'], orient='index').\
            drop(['affects'], axis=1).\
            reset_index().\
            rename(columns=equiv)

    # parameters
    param_tab = \
        sd.SuperDict(data['parameters']).\
        clean(func=lambda x: type(x) in [int, str]).\
        to_df(orient='index').\
        reset_index(). \
        rename(columns={'index': 'Parametre', 0: 'Valeur'})

    maints = sd.SuperDict.from_dict(data['maintenances'])
    m_elapsed = maints.get_property('max_elapsed_time')
    m_used = maints.get_property('max_used_time')
    start = data['parameters']['start']

    # resources
    cols_res = ['resource', 'maint', 'status', 'val']
    resources_tups = \
        sd.SuperDict.from_dict(data['resources']).\
        get_property('initial').to_dictup().to_tuplist().to_list()

    def date_add_value(date, values):
        return pd.Series(aux.shift_month(date, v) if not np.isnan(v) else np.nan for v in values)

    res_t = \
        pd.DataFrame(resources_tups, columns=cols_res).\
            set_index(['resource', 'maint', 'status']).\
            unstack('status')

    res_t.columns = res_t.columns.droplevel(0)
    res_t = res_t.rename_axis(None, axis=1).reset_index()
    res_t['elapsed'] = res_t['elapsed'] - res_t['maint'].map(m_elapsed)
    res_t['used'] = res_t['used'] - res_t['maint'].map(m_used)
    res_t['elapsed'] = date_add_value(start, res_t['elapsed'])

    equiv = {v: k for k, v in get_equiv_names().items()}

    resources_tab = res_t.rename(columns=equiv)
    avions_tab = resources_tab[['avion']].drop_duplicates().assign(heures=0)

    to_write = dict(
        maintenances = maint_tab,
        etats_initiaux = resources_tab,
        params = param_tab,
        avions = avions_tab
    )

    with pd.ExcelWriter(path) as writer:
        for sheet, table in to_write.items():
            table.to_excel(writer, sheet_name=sheet, index=False)
        writer.save()

    return True


def export_output_template(path, input_data, output_data):
    """

    :param path:
    :param input_data: instance.data
    :param output_data: solution.data
    :return:
    """
    columns = ['avion', 'mois', 'maint', 'aux']
    sol_maints = \
        sd.SuperDict.from_dict(output_data['state_m']).\
        to_dictup().\
        to_tuplist().\
        to_df(columns=columns).\
        drop('aux', axis=1)

    input_data = sd.SuperDict.from_dict(input_data)
    maint_data = input_data['maintenances']
    max_elapsed_time = maint_data.get_property('max_elapsed_time')
    max_used_time = maint_data.get_property('max_used_time')
    columns = ['state', 'maint', 'avion', 'mois', 'rem']

    remaining = \
        sd.SuperDict.from_dict(output_data['aux']).\
            to_dictup().\
            to_tuplist().\
            to_df(columns=columns). \
            assign(mois= lambda x: x.mois.map(aux.get_next_month)).\
            set_index(columns[:-1])[
            'rem'].unstack('state').reset_index().\
            assign(ret= lambda x: x.maint.map(max_elapsed_time) - x.ret + 1). \
            assign(rut=lambda x: x.maint.map(max_used_time) - x.rut)

    # Here, we calculate the consumption for the specific day
    min_usage_period = input_data['resources'].get_property('min_usage_period')
    _func = lambda a, b: min_usage_period[a].get(b, min_usage_period[a]['default'])
    remaining['cons'] = remaining[['avion', 'mois']].apply(lambda x: _func(*x), axis=1)
    remaining['rut'] += remaining['cons']
    remaining.drop(['cons'], axis=1, inplace=True)

    equiv = {'rut': 'reste_BH', 'ret': 'reste_BC'}
    result = \
        sol_maints.\
            merge(remaining, on=['maint', 'avion', 'mois'], how='left'). \
            rename(columns=equiv). \
            sort_values(['avion', 'mois', 'maint'])

    capacities_names = ['default_type2_capacity', 'maint_capacity']
    capacities = input_data['parameters'].filter(capacities_names)
    equiv = sd.SuperDict({2: 'default_type2_capacity', 1: 'maint_capacity'})
    m_cap = \
        equiv.\
        vapply(lambda v: capacities[v]).\
        to_df(columns=['cap'], orient='index').\
        rename_axis('type').reset_index()

    m_usage = maint_data.get_property('capacity_usage')
    m_type = maint_data.get_property('type')
    m_info = \
        pd.DataFrame.\
        from_dict({'usage': m_usage, 'type': m_type})\
        .rename_axis('maint').reset_index()
    m_info.type = m_info.type.astype('int64')

    num_maints = \
        tl.TupList(sol_maints.to_records()).\
        to_dict(result_col=1, indices=[2, 3]).\
        to_lendict().\
        to_tuplist().\
        to_df(columns=['period', 'maint', 'number']).\
        merge(m_info).\
        assign(total= lambda x: x.number * x.usage).\
        groupby(['period', 'type'])[['total']].\
        agg(sum).reset_index().merge(m_cap)

    to_write = {'sol_maints': result, 'sol_stats': num_maints}

    with pd.ExcelWriter(path) as writer:
        for sheet, table in to_write.items():
            table.to_excel(writer, sheet_name=sheet, index=False)
        writer.save()

    return True


def import_output_template(path):

    sheets = ['sol_maints']
    # xl = pd.ExcelFile(path)
    # missing = set(sheets) - set(xl.sheet_names)
    # if len(missing):
    #     raise KeyError('The following sheets were not found: {}'.format(missing))

    tables = {sh: pd.read_excel(path, sheet_name=sh) for sh in sheets}
    equiv = {'avion': 'resource', 'mois': 'period', 'maint':'maint'}
    columns = list(equiv.values())
    tables['sol_maints'].avion = tables['sol_maints'].avion.astype(str)

    states_table = \
        tables['sol_maints'].\
        rename(columns=equiv).\
        filter(columns).\
        assign(value=1).\
        set_index(columns)

    states_m = \
        sd.SuperDict(states_table['value'].to_dict()).\
        to_dictdict()

    states_table_n =\
        states_table.\
            reset_index().\
            set_index(["resource", 'period'])

    states = \
        sd.SuperDict(states_table_n['maint'].to_dict()).\
        to_dictdict()
    states_table_n['maint'].to_dict()


    data = dict(
        state_m = states_m
        ,state = states
        , task = {}
    )
    return data


if __name__ == '__main__':

    import package.experiment as exp
    import package.params as pm
    import package.instance as inst

    # path = r'C:\Users\pchtsp\Documents\borrar\experiments\201903101307/'
    # self = exp.Experiment.from_dir(path)
    # data = self.instance.data
    # export_input_template(path + 'template_in.xlsx', data)
    # stop

    path = pm.PATHS['data'] + 'template/official/'
    data = import_input_template(path + 'template_in.xlsx')
    instance = inst.Instance(data)
    instance.get_capacity_calendar()
    export_input_template(path, data)

    from package.params import PATHS, OPTIONS

    path = PATHS['data'] + 'examples/201903041426/'
    experiment = exp.Experiment.from_dir(path)

    data = experiment.instance.data
    export_input_template(path, data)

    experiment.set_remaining_usage_time_all('rut')
    experiment.set_remaining_usage_time_all('ret')

    path_out = '/home/pchtsp/Documents/projects/optima_dassault/data/template_out_out.xlsx'
    data = experiment.solution.data
    data_in = experiment.instance.data
    export_output_template(path_out, data_in, data)

    path_out = '/home/pchtsp/Documents/projects/optima_dassault/data/template_out_out.xlsx'
    data = import_output_template(path_out)

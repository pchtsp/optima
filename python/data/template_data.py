import pandas as pd
import package.auxiliar as aux
import numpy as np
import pytups.superdict as sd


def get_parameters(tables):
    data = tables['params'].set_index('Parametre')['Valeur'].to_dict()
    data['end'] = aux.shift_month(data['start'], data['num_period'] - 1)
    return data


def get_equiv_maint():
    return {
        'duree': 'duration_periods'
        , 'BC': 'max_elapsed_time'
        , 'BC_tol': 'elapsed_time_size'
        , 'BH': 'max_used_time'
        , 'BH_tol': 'used_time_size'
        , 'capacite': 'capacity'
        , 'capacit_utili': 'capacity_usage'
    }

def get_maintenance(tables):
    equiv = get_equiv_maint()
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


def get_resources(tables):

    params = get_parameters(tables)
    start = params['start']

    equiv = {
        'avion': 'resource'
        , 'mois_derniere': 'elapsed'
        , 'heures_derniere': 'used'
    }

    states = \
        tables['etats_initiaux'].\
        rename(columns=equiv)

    maint_tab = tables['maintenances']

    def elapsed_time_between_dates(value, series2):
        return pd.Series(len(aux.get_months(p2, value)) - 1 if not pd.isna(p2) else np.nan
                         for p2 in series2)

    equiv = {
        'avion': 'resource'
        , 'heures': 'hours'
    }

    resources = tables['avions'].rename(columns=equiv)

    resources.resource = resources.resource.astype(str)
    states.resource = states.resource.astype(str)

    resources = \
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

    resources = sd.SuperDict.from_dict(resources).to_dictdict()
    def_resources = {'type': '1', 'capacities': [], 'states': {}, 'code': ''}

    for r in resources:
        resources[r].update(def_resources)

    return resources

def import_input_template(path):
    """
    :param path:
    :return: the data needed to create an instance
    """

    sheets = ['maintenances', 'etats_initiaux', 'avions', 'params']
    tables = {sh: pd.read_excel(path, sheet_name=sh) for sh in sheets}
    data = dict(
        parameters = get_parameters(tables)
        , tasks = {}
        , maintenances = get_maintenance(tables)
        , resources = get_resources(tables)
    )

    return data

def export_input_template(path, data):
    """
    :param path:
    :param data: the instance.data part, dictionary
    :return: nothing
    """
    equiv = sd.SuperDict(get_equiv_maint()).reverse()
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

    equiv = {'elapsed': 'mois_derniere', 'used': 'heures_derniere', 'resource': 'avion'}

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


def export_output_template(path, data):
    """

    :param path:
    :param data: solution.data
    :return:
    """
    columns = ['avion', 'mois', 'maint', 'aux']
    sol_maints = \
        sd.SuperDict.from_dict(data['state_m']).\
        to_dictup().\
        to_tuplist().\
        to_df(columns=columns).\
        drop('aux', axis=1)

    columns = ['state', 'maint', 'avion', 'mois', 'rem']
    rem_m = \
        sd.SuperDict.from_dict(data['aux']).\
            to_dictup().\
            to_tuplist().\
            to_df(columns=columns).\
            set_index(columns[:-1]).\
            unstack('state')

    rem_m.columns = rem_m.columns.droplevel(0)
    rem_m = rem_m.rename_axis(None, axis=1).reset_index()

    equiv = {'rut': 'reste_BH', 'ret': 'reste_BC'}
    result = \
        sol_maints.\
            merge(rem_m, on=['maint', 'avion', 'mois'], how='left'). \
            rename(columns=equiv). \
            sort_values(['avion', 'mois', 'maint'])

    to_write = {'sol_maints': result}

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

    path = r'C:\Users\pchtsp\Documents\borrar\experiments\201903101307/'
    self = exp.Experiment.from_dir(path)
    data = self.instance.data
    export_input_template(path + 'template_in.xlsx', data)
    stop

    path = '/home/pchtsp/Documents/projects/optima_dassault/data/template_in.xlsx'
    path = '/home/pchtsp/Documents/projects/optima_dassault/data/template_in_out.xlsx'
    data = import_input_template(path)
    export_input_template(path, data)

    from package.params import PATHS, OPTIONS
    import package.experiment as exp

    path = PATHS['data'] + 'examples/201903041426/'
    experiment = exp.Experiment.from_dir(path)

    data = experiment.instance.data
    export_input_template(path, data)

    experiment.set_remaining_usage_time_all('rut')
    experiment.set_remaining_usage_time_all('ret')

    path_out = '/home/pchtsp/Documents/projects/optima_dassault/data/template_out_out.xlsx'
    data = experiment.solution.data
    export_output_template(path_out, data)

    path_out = '/home/pchtsp/Documents/projects/optima_dassault/data/template_out_out.xlsx'
    data = import_output_template(path_out)

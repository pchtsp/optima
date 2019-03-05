import os
import pandas as pd
import package.auxiliar as aux
import numpy as np
import package.superdict as sd


def get_parameters(tables):
    return tables['params'].set_index('Parametre')['Valeur'].to_dict()


def get_equiv_maint():
    return {
        'duree': 'duration_periods'
        , 'BC': 'max_elapsed_time'
        , 'BC_tol': 'elapsed_time_size'
        , 'BH': 'max_used_time'
        , 'BH_tol': 'used_time_size'
        , 'capacite': 'capacity'
    }


def get_equiv_res():
    pass


def get_maintenance(tables):
    equiv = get_equiv_maint()
    maint_tab  =\
        tables['maintenances']. \
        rename(columns=equiv). \
        replace({np.nan: None})

    return \
        maint_tab.\
        set_index('maint').\
        to_dict(orient='index')


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
        return pd.Series(len(aux.get_months(p2, value)) - 1 for p2 in series2)

    equiv = {
        'avion': 'resource'
        , 'heures': 'hours'
    }

    resources = \
        tables['avions'].\
        rename(columns=equiv). \
        merge(states, on='resource').\
        assign(used = lambda x: x.hours - x.used). \
        assign(elapsed=lambda x: x.elapsed.str.slice(stop=7)). \
        assign(elapsed=lambda x: elapsed_time_between_dates(start, x.elapsed)). \
        merge(maint_tab[['maint', 'BC', 'BH']], on='maint'). \
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

    maint_tab = \
        pd.DataFrame.from_dict(data['maintenances'], orient='index'). \
            reset_index().\
            rename(columns=equiv)

    param_tab = \
        pd.DataFrame.from_dict(data['parameters'], orient='index').\
        reset_index().\
        rename(columns={'index': 'Parametre', 0: 'Valeur'})

    maints = sd.SuperDict.from_dict(data['maintenances'])
    m_elapsed = maints.get_property('max_elapsed_time')
    m_used = maints.get_property('max_used_time')
    start = data['parameters']['start']

    cols_res = ['resource', 'maint', 'status', 'val']
    resources_tups = \
        sd.SuperDict.from_dict(data['resources']).\
        get_property('initial').to_dictup().to_tuplist().to_list()

    def date_add_value(values, date):
        return pd.Series(aux.shift_month(date, v) if not np.isnan(v) else np.nan for v in values)

    res_t = \
        pd.DataFrame(resources_tups, columns=cols_res).\
            set_index(['resource', 'maint', 'status']).\
            unstack('status')

    res_t.columns = res_t.columns.droplevel(0)
    res_t = res_t.rename_axis(None, axis=1).reset_index()
    res_t['elapsed'] = res_t['elapsed'] - res_t['maint'].map(m_elapsed)
    res_t['used'] = res_t['used'] - res_t['maint'].map(m_used)
    res_t['elapsed'] = date_add_value(res_t['elapsed'], start)

    equiv = {'elapsed': 'mois_derniere', 'used': 'heures_derniere', 'resource': 'avion'}

    resources_tab = res_t.rename(columns=equiv)
    avions_tab = resources_tab[['avion']].assign(heures=0)


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


if __name__ == '__main__':
    path = '/home/pchtsp/Documents/projects/optima_dassault/data/template_in.xlsx'
    data = import_input_template(path)

    path_out = '/home/pchtsp/Documents/projects/optima_dassault/data/template_in_out.xlsx'
    export_input_template(path_out, data)

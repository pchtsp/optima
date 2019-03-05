import os
import pandas as pd
import re
# import unidecode
import unicodedata
import package.auxiliar as aux
import numpy as np
import copy
import json
import pickle
import package.superdict as sd


def make_name(name):
    # we take out spaces and later weird accents
    # we replace parenthesis with an underscore
    name = re.sub(pattern=r'\(', string=name, repl='_')
    name = re.sub("\s[a-z]", lambda m: m.group(0)[1].upper(), name)
    name = re.sub(pattern=r'[\s\n\):\+\?]', string=name, repl='')
    s2 = unicodedata.normalize('NFD', name).encode('ascii', 'ignore')
    return str(s2, 'utf-8')
    # return unidecode.unidecode(name)


def make_names(names):
    return [make_name(name) for name in names]


def generate_data_from_source(source=r'../data/raw/parametres_DGA_final.xlsm'):
    if not os.path.exists(source):
        print('Following path does not exist: {}'.format(source))
        return None
    excel_file = pd.ExcelFile(source)
    sheets = excel_file.sheet_names
    # print(sheets)
    excel_info = {make_name(sheet): excel_file.parse(sheet) for sheet in sheets}
    # print(excel_info.keys())

    for sheet in excel_info:
        excel_info[sheet].columns = make_names(excel_info[sheet].columns)

    return excel_info


def export_data_csv(excel_info, destination="../data/csv"):
    for sheet in excel_info:
        excel_info[sheet].to_csv(destination + r'/{}.csv'.format(sheet), index=False)
    return


def generate_data_from_csv(directory=r'../data/csv/'):
    csvfiles = os.listdir(directory)
    csvfiles_dict = {path: os.path.splitext(path)[0] for path in csvfiles}
    return {csvfiles_dict[csv]: pd.read_csv(directory + csv) for csv in csvfiles}


def get_model_data(source=r'../data/raw/parametres_DGA_final.xlsm'):
    # we import the data set.
    table = generate_data_from_source(source=source)
    # print(table)

    params = table['Parametres']

    planning_cols = [col for col in params if re.findall(string=col, pattern=r'\d+$') and
                     int(re.findall(string=col, pattern=r'\d+$')[0]) in range(2, 5)]

    horizon = params[planning_cols]
    horizon = horizon[~horizon.iloc[:, 1].isna()].rename(columns=lambda x: "c" + x[-1])
    horizon = horizon.assign(date=horizon.c4.apply(str) + "-" +
                                  horizon.c3.apply(lambda x: str(x).zfill(2)))
    horizon = horizon[~horizon.iloc[:, 0].isna()].set_index("c2")["date"].to_dict()

    params_gen = params[~params.Unnamed9.isna()].rename(
        columns={'Unnamed9': 'name', 'Unnamed10': 'value'})[['name', 'value']]

    params_gen = params_gen.set_index('name').to_dict()['value']

    # TASKS AND RESOURCES
    # if a task has 5 required capacities
    # and it has only 4 capacities in common with a resource
    # then the resource is not fitted to do the task
    # we assume all resources have the capacity=99
    # we use the resource type as a capacity to match them to tasks

    tasks_data = table['Missions']
    tasks_data = \
        tasks_data.assign(start=tasks_data.AnneeDeDebut.apply(str) + '-' +
                                tasks_data.MoisDeDebut.apply(lambda x: str(x).zfill(2)),
                          end=tasks_data.AnneeDeFin.apply(str) + '-' +
                              tasks_data.MoisDeFin.apply(lambda x: str(x).zfill(2)))

    tasks_data.set_index('IdMission', inplace=True)

    capacites_col = ['Type'] + [col for col in tasks_data if col.startswith("Capacite")]
    capacites_mission = tasks_data.reset_index(). \
        melt(id_vars=["IdMission"], value_vars=capacites_col) \
        [['IdMission', "value"]]
    capacites_mission = capacites_mission[~capacites_mission.value.isna()].set_index('value')

    avions = table['Avions_Capacite']

    capacites_col = ['TypeAvion', 'Capacites'] + [col for col in avions if col.startswith("Unnamed")]
    capacites_avion = avions.melt(id_vars=["IdAvion"], value_vars=capacites_col)[['IdAvion', "value"]]

    capacites_avion_extra = capacites_avion.IdAvion.drop_duplicates().to_frame().assign(value=99)
    capacites_avion = pd.concat([capacites_avion[~capacites_avion.value.isna()],
                                 capacites_avion_extra]).set_index('value')

    mission_capacities = aux.tup_to_dict(
        capacites_mission.to_records().tolist(), result_col=0)
    aircraft_capacities = aux.tup_to_dict(
        capacites_avion.to_records().tolist(), result_col=0)

    maint = table['DefinitionMaintenances']
    avions_state = table['Avions_Potentiels']

    model_data = {}
    model_data['parameters'] = {
        'maint_weight': 1,
        'unavail_weight': 1,
        'max_used_time': maint.GainPotentielHoraire_heures.values.min().__float__()
        ,'max_elapsed_time': maint.GainPotentielCalendaire_mois.values.min().__int__()
        ,'maint_duration': maint.DureeMaintenance_mois.values.max().__int__()
        ,'maint_capacity': params_gen['Maintenance max par mois'].__int__()
        ,'start': horizon["Début"]
        ,'end': horizon["Fin"]
    }

    av_tasks = tasks_data.index
    # [task]
    model_data['tasks'] = {
        task: {
            'start': tasks_data.start.to_dict()[task]
            , 'end': tasks_data.end.to_dict()[task]
            , 'consumption': tasks_data['MaxPu/avion/mois'].to_dict()[task]
            , 'num_resource': tasks_data.nombreRequisA1.to_dict()[task]
            , 'type_resource': tasks_data['Type'].to_dict()[task]
            , 'matricule': tasks_data['MatriculeMission'].to_dict()[task]
            , 'min_assign': tasks_data['MinAffectation'].to_dict()[task]
            , 'capacities': mission_capacities[task]
        } for task in av_tasks
    }

    av_resource = np.intersect1d(avions_state.IdAvion, avions.IdAvion)

    model_data['resources'] = {
        resource: {
            'initial_used': avions_state.set_index("IdAvion")['PotentielHoraire_HdV'].to_dict()[resource]
            , 'initial_elapsed': avions_state.set_index("IdAvion")['PotentielCalendaire'].to_dict()[resource]
            , 'code': avions.set_index("IdAvion")['MatriculeAvion'].to_dict()[resource]
            , 'capacities': aircraft_capacities[resource]
        } for resource in av_resource
    }

    return model_data


def generate_solution_from_source(source=r'../data/raw/Planifs M2000.xlsm'):
    excel_file = pd.ExcelFile(source)

    sheets = excel_file.sheet_names
    table = pd.read_excel(source, sheet_name='Visu totale', header=None)
    year = table.loc[0, 4:]
    year = np.where(np.char.startswith(np.array(year, dtype="U4"), '20'),
             year,
             np.nan)
    year = pd.Series(year).fillna(method='ffill').apply(str)
    months_names = ("Ja  Fe  Ma  Av  Mi  Jn  Jt  Au  Se  Oc  No  De").split(r'  ')
    month_pos = {months_names[pos]: str(pos+1).zfill(2) for pos in range(len(months_names))}
    # lines = table.loc[1, 4:].isin(month_pos).reset_index(drop=True)
    months = table.loc[1, 4:].apply(lambda x: month_pos.get(x, "00")).reset_index(drop=True)
    colnames = ['code'] + list(year + '-' + months)
    table_n = table.loc[2:, 3:].copy()
    table_n.columns = colnames
    state = pd.melt(table_n, id_vars="code", var_name="month", value_name="state").dropna()
    state = state[~state.month.str.endswith("00")]
    return state.set_index(["code", "month"])['state'].to_dict()


def combine_data_states(model_data, historic_data):
    codes = aux.get_property_from_dic(model_data['resources'], 'code')
    codes_inv = {value: key for key, value in codes.items()}
    historic_data_n = {
        (codes_inv[code], month): value for (code, month), value in historic_data.items()\
        if code in codes_inv
    }
    previous_states = {key: 'M' for key, value in historic_data_n.items()
                       if int(str(value).startswith('V'))
                       }
    model_data_n = copy.deepcopy(model_data)
    for key, value in aux.dicttup_to_dictdict(previous_states).items():
        model_data_n['resources'][key]['states'] = value
    return model_data_n


def load_data(path, file_type=None):
    if file_type is None:
        splitext = os.path.splitext(path)
        if len(splitext) == 0:
            raise ImportError("file type not given")
        else:
            file_type = splitext[1][1:]
    if file_type not in ['json', 'pickle']:
        raise ImportError("file type not known: {}".format(file_type))
    if not os.path.exists(path):
        return False
    if file_type == 'pickle':
        with open(path, 'rb') as f:
            return pickle.load(f)
    if file_type == 'json':
        with open(path, 'r') as f:
            return json.load(f)


def export_data(path, dictionary, name=None, file_type="json", exclude_aux=False):
    # I'm gonna add the option to exclude the 'aux' section of the object, if exists
    # we're assuming obj is a dictionary
    dictionary = copy.deepcopy(dictionary)
    if exclude_aux and 'aux' in dictionary:
        dictionary.pop('aux', None)
    if not os.path.exists(path):
        os.mkdir(path)
    if name is None:
        name = aux.get_timestamp()
    path = os.path.join(path, name + "." + file_type)
    if file_type == "pickle":
        with open(path, 'wb') as f:
            pickle.dump(dictionary, f)
    if file_type == 'json':
        with open(path, 'w') as f:
            json.dump(dictionary, f, cls=MyEncoder)
    return True


def import_pie_solution(path_solution, path_input):
    # path_solution = "/home/pchtsp/Documents/projects/PIE/glouton/solution.csv"
    # path_input = "/home/pchtsp/Documents/projects/PIE/glouton/parametres_DGA_final.xlsm"
    model_data = get_model_data(path_input)
    table = pd.read_csv(path_solution, sep=';')
    periods = [i for i in range(1, len(table.columns))]
    table.columns = ['resource'] + periods
    table = pd.melt(table, value_vars=periods, id_vars=['resource'])
    table = table[~pd.isna(table.value)]
    table = table[~(table.value == '-')]

    resources_equiv = aux.get_property_from_dic(model_data["resources"], 'code')
    r_e_i = {v: k for k, v in resources_equiv.items()}
    tasks_equiv = aux.get_property_from_dic(model_data["tasks"], 'matricule')
    t_e_i = {v: k for k, v in tasks_equiv.items()}
    start, end = model_data['parameters']['start'], model_data['parameters']['end']
    p_e_i = {k: aux.get_months(start, end)[k-1] for k in periods}

    elements = table.value.str.split('$')
    ismission = elements.apply(lambda x: len(x) == 2)
    state = table[~ismission].reset_index(drop=True)
    state.value = "M"

    state_dict = state.set_index(['resource', 'variable'])['value'].to_dict()
    state_dict_e = {(r_e_i[k[0]], p_e_i[k[1]]): v for k, v in state_dict.items()}
    state_dict_f = aux.dicttup_to_dictdict(state_dict_e)

    table.value = elements.apply(lambda x: x[0])

    missions = table[ismission]
    missions_dict = missions.set_index(['resource', 'variable'])['value'].to_dict()
    missions_dict_e = {(r_e_i[k[0]], p_e_i[k[1]]): t_e_i[v] for k, v in missions_dict.items()}
    task_dict_f = aux.dicttup_to_dictdict(missions_dict_e)

    return {
        'state': state_dict_f,
        'task': task_dict_f
    }

def import_excel_template(path):
    path = '/home/pchtsp/Documents/projects/optima_dassault/data/template_in.xlsx'
    sheets = ['maintenances', 'etats_initiaux', 'avions', 'params']
    tables = {sh: pd.read_excel(path, sheet_name=sh) for sh in sheets}
    today = '2018-11'
    equiv = {
        'duree': 'duration_periods'
        , 'BC': 'max_elapsed_time'
        , 'BC_tol': 'elapsed_time_size'
        , 'BH': 'max_used_time'
        , 'BH_tol': 'used_time_size'
        , 'capacite': 'capacity'
    }
    data = {}

    params = tables['params'].set_index('Parametre').to_dict(orient='index')

    tasks = data['tasks'] = {   }

    maint_tab  =\
        tables['maintenances']. \
        rename(columns=equiv). \
        replace({pd.np.nan: None})

    maints = data['maintenances'] =\
        maint_tab.\
        set_index('maint').\
        to_dict(orient='index')

    equiv = {
        'avion': 'resource'
        , 'mois_derniere': 'elapsed'
        , 'heures_derniere': 'used'
    }

    states = \
        tables['etats_initiaux'].\
        rename(columns=equiv)

    equiv = {
        'avion': 'resource'
        , 'heures_derniere': 'hours'
        , 'mois_derniere': 'period'
    }

    def elapsed_time_between_dates(value, series2):
        return pd.Series(len(aux.get_months(p2, value)) for p2 in series2)

    resources = \
        tables['avions'].\
        rename(columns=equiv). \
        merge(states, on='resource').\
        assign(used = lambda x: x.hours - x.used). \
        assign(period=lambda x: x.period.str.slice(stop=7)). \
        assign(elapsed=lambda x: x.elapsed.str.slice(stop=7)). \
        assign(elapsed=lambda x: elapsed_time_between_dates(today, x.elapsed)). \
        merge(maint_tab[['maint', 'max_elapsed_time', 'max_used_time']], on='maint'). \
        assign(elapsed=lambda x: x.max_elapsed_time - x.elapsed). \
        assign(used=lambda x: x.max_used_time - x.used). \
        filter(['resource', 'maint', 'used', 'elapsed']). \
        assign(initial='initial'). \
        set_index(['resource', 'initial', 'maint']). \
        to_dict(orient='index')

    resources = data['resources'] = sd.SuperDict(resources).to_dictdict()

    def_resources = {'type': '1', 'capacities': [], 'states': {}, 'code': ''}

    for r in resources:
        data['resources'][r].update(def_resources)



    pass

def export_excel_template(path):
    path = '/home/pchtsp/Documents/projects/optima_dassault/data/template_out.xlsx'
    sheets = ['sol_maints', 'sol_etats']
    tables = {sh: pd.read_excel(path, sheet_name=sh) for sh in sheets}

    pass

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

if __name__ == "__main__":
    # get_model_data()
    path_solution = "/home/pchtsp/Documents/projects/PIE/glouton/solution.csv"
    path_input = "/home/pchtsp/Documents/projects/PIE/glouton/parametres_DGA_final.xlsm"
    model_data = get_model_data(path_input)
    import_pie_solution(path_solution, path_input)


    def _function(cell):
        # cell = "LUXEUIL_5F$23"
        if pd.isna(cell):
            return None
        result = re.search("(.*)\$", cell)
        if result is not None:
            return result.group(1)
        return None

    table = pd.read_csv(path_solution, sep=';')
    periods = [i for i in range(1, len(table.columns))]
    table.columns = ['resource'] + periods
    table = pd.melt(table, value_vars=periods, id_vars=['resource'])
    table.value = table.value.apply(_function)
    table = table[~pd.isnull(table.value)]
    len(table)
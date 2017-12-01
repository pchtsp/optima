import os
import pandas as pd
import re
import unidecode
import package.aux as aux

def make_name(name):
    # we take out spaces and later weird accents
    # we replace parenthesis with an underscore
    name = re.sub(pattern=r'\(', string=name, repl='_')
    name = re.sub("\s[a-z]", lambda m: m.group(0)[1].upper(), name)
    name = re.sub(pattern=r'[\s\n\):\+\?]', string=name, repl='')
    return unidecode.unidecode(name)


def make_names(names):
    return [make_name(name) for name in names]


def generate_data_from_source(source=r'../data/raw/parametres_DGA_final.xlsm'):
    excel_file = pd.ExcelFile(source)

    sheets = excel_file.sheet_names

    excel_info = {make_name(sheet): excel_file.parse(sheet) for sheet in sheets}
    # sheets = list(excel_info.keys())

    for sheet in excel_info:
        excel_info[sheet].columns = make_names(excel_info[sheet].columns)

    return excel_info


def export_data(excel_info, destination="../data/csv"):

    for sheet in excel_info:
        excel_info[sheet].to_csv(destination + r'/{}.csv'.format(sheet), index=False)


def generate_data_from_csv(directory=r'../data/csv/'):

    csvfiles = os.listdir(directory)
    csvfiles_dict = {path: os.path.splitext(path)[0] for path in csvfiles}
    return {csvfiles_dict[csv]: pd.read_csv(directory + csv) for csv in csvfiles}


def get_model_data():
    # we import the data set.
    table = generate_data_from_source()

    # df3 = df3.assign(velo=df3.dist / df3.duration*3600/1000)  # for km/h

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

    tasks_data = table['Missions']
    tasks_data = \
        tasks_data.assign(start=tasks_data.AnneeDeDebut.apply(str) + '-' +
                                tasks_data.MoisDeDebut.apply(lambda x: str(x).zfill(2)),
                          end=tasks_data.AnneeDeFin.apply(str) + '-' +
                              tasks_data.MoisDeFin.apply(lambda x: str(x).zfill(2)))

    tasks_data.set_index('IdMission', inplace=True)

    capacites_col = [col for col in tasks_data if col.startswith("Capacite")]
    capacites_mission = tasks_data.reset_index(). \
        melt(id_vars=["IdMission"], value_vars=capacites_col) \
        [['IdMission', "value"]]
    capacites_mission = capacites_mission[~capacites_mission.value.isna()].set_index('value')

    maint = table['DefinitionMaintenances']

    avions = table['Avions_Capacite']

    capacites_col = ['Capacites'] + [col for col in avions if col.startswith("Unnamed")]
    capacites_avion = avions.melt(id_vars=["IdAvion"], value_vars=capacites_col)[['IdAvion', "value"]]

    capacites_avion = capacites_avion[~capacites_avion.value.isna()].set_index('value')

    num_capacites = capacites_mission.reset_index().groupby("IdMission"). \
        agg(len).reset_index()
    capacites_join = capacites_mission.join(capacites_avion)
    capacites_join = capacites_join.reset_index(). \
        groupby(['IdMission', 'IdAvion']).agg(len).reset_index()

    mission_aircraft = \
        pd.merge(capacites_join, num_capacites, on=["IdMission", "value"]) \
            [["IdMission", "IdAvion"]]

    # TODO: I'm missing for some reason half the missions that do not
    # have at least one aircraft as candidate...

    avions_state = table['Avions_Potentiels']

    model_data = {}
    model_data['parameters'] = {
        'max_used_time': maint.GainPotentielHoraire_heures.values.min()
        ,'max_elapsed_time': maint.GainPotentielCalendaire_mois.values.min()
        ,'maint_duration': maint.DureeMaintenance_mois.values.max()
        ,'maint_capacity': params_gen['Maintenance max par mois']
        ,'start': horizon["Début"]
        ,'end': horizon["Fin"]
    }

    model_data['tasks'] = {
        'start': tasks_data.start.to_dict()
        , 'end': tasks_data.end.to_dict()
        , 'consumption': tasks_data['MaxPu/avion/mois'].to_dict()
        , 'num_resource': tasks_data.nombreRequisA1.to_dict()
        , 'candidates': mission_aircraft.groupby("IdMission")['IdAvion'].apply(lambda x: x.tolist()).to_dict()
    }

    model_data['resources'] = {
        'initial_used': avions_state.set_index("IdAvion")['PotentielHoraire_HdV'].to_dict()
        ,'initial_elapsed': avions_state.set_index("IdAvion")['PotentielCalendaire'].to_dict()

    }

    return model_data


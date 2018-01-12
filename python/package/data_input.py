import os
import pandas as pd
import re
import unidecode
import package.aux as aux
import numpy as np
import copy
import json
import pickle


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


def get_model_data():
    # we import the data set.
    table = generate_data_from_source()

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
    num_capacites = capacites_mission.reset_index().groupby("IdMission"). \
        agg(len).reset_index()
    capacites_join = capacites_mission.join(capacites_avion)
    capacites_join = capacites_join.reset_index(). \
        groupby(['IdMission', 'IdAvion']).agg(len).reset_index()

    mission_aircraft = \
        pd.merge(capacites_join, num_capacites, on=["IdMission", "value"]) \
            [["IdMission", "IdAvion"]]

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

    av_tasks = np.intersect1d(mission_aircraft.IdMission.unique(), tasks_data.index)
    # [task]
    model_data['tasks'] = {
        task: {
            'start': tasks_data.start.to_dict()[task]
            , 'end': tasks_data.end.to_dict()[task]
            , 'consumption': tasks_data['MaxPu/avion/mois'].to_dict()[task]
            , 'num_resource': tasks_data.nombreRequisA1.to_dict()[task]
            , 'candidates': mission_aircraft.groupby("IdMission")['IdAvion'].
                apply(lambda x: x.tolist()).to_dict()[task]
        } for task in av_tasks
    }

    av_resource = np.intersect1d(avions_state.IdAvion, avions.IdAvion)

    model_data['resources'] = {
        resource: {
            'initial_used': avions_state.set_index("IdAvion")['PotentielHoraire_HdV'].to_dict()[resource]
            , 'initial_elapsed': avions_state.set_index("IdAvion")['PotentielCalendaire'].to_dict()[resource]
            , 'code': avions.set_index("IdAvion")['MatriculeAvion'].to_dict()[resource]
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


def export_data(path, obj, name=None, file_type="pickle"):
    if not os.path.exists(path):
        os.mkdir(path)
    if name is None:
        name = aux.get_timestamp()
    path = os.path.join(path, name + "." + file_type)
    if file_type == "pickle":
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    if file_type == 'json':
        with open(path, 'w') as f:
            json.dump(obj, f)
    return True


def get_log_info_gurobi(path):
    # path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201712121208/results.log"
    with open(path, 'r') as f:
        content = f.read()
    numberSearch = r'([\de\.\+]+)'
    regex = r'Best objective {0}, best bound {0}, gap {0}%'.format(numberSearch)
    # "Best objective 6.500000000000e+01, best bound 5.800000000000e+01, gap 10.7692%"
    solution = re.search(regex, content)

    regex = r'Optimize a model with {0} rows, {0} columns and {0} nonzeros'.format(numberSearch)
    size = re.search(regex, content)

    return {
        'bound_out': float(solution.group(2)),
        'objective_out': float(solution.group(1)),
        'gap_out': float(solution.group(3)),
        'cons': int(size.group(1)),
        'vars': int(size.group(2)),
        'nonzeros': int(size.group(3))
    }


def get_log_info_cplex(path):
    # path = "/home/pchtsp/Documents/projects/OPTIMA_documents/results/experiments/201801110004/results.log"
    with open(path, 'r') as f:
        content = f.read()
    numberSearch = r'([\de\.\+]+)'
    wordsSearch = r'([\w, ]+)'
    # MIP - Integer optimal solution:  Objective =  1.5000000000e+01
    # MIP - Time limit exceeded, integer feasible:  Objective =  2.0300000000e+01
    # Current MIP best bound =  1.2799071852e+02 (gap = 20.0093, 13.52%)
    regex = r'MIP - {1}:  Objective =  {0}\n'.format(numberSearch, wordsSearch)
    solution = re.search(regex, content, flags=re.MULTILINE)

    if solution is None:
        return {}

    objective = float(solution.group(2))
    bound = objective
    gap = 0
    if solution.group(1) == "Time limit exceeded, integer feasible":
        regex = r'Current MIP best bound =  {0} \(gap = {0}, {0}%\)'.format(numberSearch)
        solution = re.search(regex, content)
        bound = float(solution.group(1))
        gap = float(solution.group(3))

    regex = r'Reduced MIP has {0} rows, {0} columns, and {0} nonzeros'.format(numberSearch)
    size = re.search(regex, content)

    if size is None:
        return {}
    regex = r'Root relaxation solution time = {0} sec\. \({0} ticks\)'.format(numberSearch)
    result = re.search(regex, content)
    rootTime = float(result.group(1))
    regex = r'Elapsed time = {0} sec\. \({0} ticks, tree = {0} MB, solutions = {0}\)'.format(numberSearch)
    result = re.search(regex, content)
    cutsTime = float(result.group(1))
    regex = r'{1} cuts applied:  {0}'.format(numberSearch, wordsSearch)
    result = re.findall(regex, content)
    cuts = {}
    if result:
        cuts = {k[0]: int(k[1]) for k in result}
    regex = r'LP Presolve eliminated {0} rows and {0} columns'.format(numberSearch)
    result = re.findall(regex, content)
    presolve = {}
    if result:
        presolve = {
            'rows': result[0][0],
            'cols': result[0][1]
        }
    regex = r'Presolve time = {0} sec. \({0} ticks\)'.format(numberSearch)
    result = re.findall(regex, content)
    if result:
        presolve['time'] = result[0]

    regex = r'^\*?\s+{0}\s+{0}\s+{0}?\s+{0}?\s+{0}?\s+(Cuts: )?{0}?\s+{0}\s+{0}?\%?$'.format(numberSearch)
    result = re.findall(regex, content, flags=re.MULTILINE)
    after_cuts = -1
    first_relax = -1
    if result:
        first_relax = float(result[0][2])
        for r in result:
            if r[0] == '0' and r[1] == '2':
                after_cuts = float(r[2])
                break

    return {
        'bound_out': bound,
        'objective_out': objective,
        'gap_out': gap,
        'cons': int(size.group(1)),
        'vars': int(size.group(2)),
        'nonzeros': int(size.group(3)),
        'cuts': cuts,
        'rootTime': rootTime,
        'cutsTime': cutsTime,
        'presolve': presolve,
        'first_relaxed': first_relax,
        'after_cuts': after_cuts
    }


if __name__ == "__main__":
    get_model_data()
    pass

import pandas as pd
import re
import data.data_dga as dd
import package.auxiliar as aux


def import_pie_solution(path_solution, path_input):
    # path_solution = "/home/pchtsp/Documents/projects/PIE/glouton/solution.csv"
    # path_input = "/home/pchtsp/Documents/projects/PIE/glouton/parametres_DGA_final.xlsm"
    model_data = dd.get_model_data(path_input)
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


if __name__ == "__main__":
    # get_model_data()
    path_solution = "/home/pchtsp/Documents/projects/PIE/glouton/solution.csv"
    path_input = "/home/pchtsp/Documents/projects/PIE/glouton/parametres_DGA_final.xlsm"
    model_data = dd.get_model_data(path_input)
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
import os
import pandas as pd
import re
import unidecode


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
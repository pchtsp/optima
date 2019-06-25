import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from desktop_app.gui import Ui_MainWindow

import package.auxiliar as aux
import package.data_input as di
import package.instance as inst
import package.params as params
import package.heuristics as heur
import os
import arrow
import pprint as pp
from io import StringIO

import package.template_data as td
import package.exec as exec
import package.heuristics_maintfirst as mf

# TODO: add dialog when it finds a solution


class MainWindow_EXCEC():

    def __init__(self, options):
        self.options = options
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(MainWindow)
        self.ui.chooseFile.clicked.connect(self.choose_file)
        self.ui.readExcel.clicked.connect(self.read_file)
        self.input_path = params.PATHS['data'] + 'raw/parametres_DGA_final.xlsm'
        self.solution_path = params.PATHS['data'] + 'raw/solution.xlsx'
        self.ui.excel_path.setText(self.input_path)
        self.ui.max_time.setText('3600')
        self.input_data = {}
        self.ui.generateSolution.clicked.connect(self.generate_solution)

        MainWindow.show()
        sys.exit(app.exec_())

    def choose_file(self):
        QFileDialog = QtWidgets.QFileDialog
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            caption="QFileDialog.getOpenFileName()",
            directory=params.PATHS['data'],
            filter="Excel files (*.xlsx *.xlsm);;All Files (*)",
            options=options)
        if not fileName:
            return False
        self.input_path = fileName
        self.ui.excel_path.setText(fileName)
        return True

    def read_file(self):
        # input_file = params.PATHS['data'] + 'raw/parametres_DGA_final.xlsm'
        self.input_data = di.get_model_data(self.input_path)
        # historic_data = di.generate_solution_from_source()
        # self.input_data = di.combine_data_states(self.input_data, historic_data)
        black_list = ['O8']  # this task has less candidates than what it asks.
        self.input_data['tasks'] = \
            {k: v for k, v in self.input_data['tasks'].items() if k not in black_list}
        start, end = self.input_data['parameters']['start'], self.input_data['parameters']['end']
        self.ui.start_date.setDate(arrow.get(start).datetime)
        self.ui.end_date.setDate(arrow.get(end).datetime)

    def load_template(self):
        options = self.options
        model_data = td.import_input_template(options['input_template_path'])
        self.instance = inst.Instance(model_data)
        return self.instance

    def generate_solution(self):
        # exec.config_and_solve(self.options)
        options = self.options
        instance = self.load_template()
        self.experiment = experiment = mf.MaintenanceFirst(instance, solution=None)
        output_path = options['path']

        experiment.solve(options)
        # old_stdout = sys.stdout
        # sys.stdout = mystdout = StringIO()
        errors = experiment.check_solution()
        errors = {k: v.to_dictdict() for k, v in errors.items()}

        if len(errors) == 0:
            self.ui.solution_log.setText("Everthing went fine!")
        else:
            self.ui.solution_log.setText(pp.pprint(errors))
        try:
            _kwags = dict(file_type='json', exclude_aux=True)
            di.export_data(output_path, experiment.instance.data, name="data_in", **_kwags)
            di.export_data(output_path, experiment.solution.data, name="data_out", **_kwags)
            td.export_output_template(options['output_template_path'], experiment.solution.data)
        except:
            print("Solution could not be exported")
        log_path = os.path.join(output_path, 'output.log')
        try:
            with open(log_path) as f:
                res = f.readlines()
        except:
            print('Error reading log file')
            res = ''
        self.ui.solution_log.setText(res)


if __name__ == "__main__":
    MainWindow_EXCEC({})




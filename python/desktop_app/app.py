import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from desktop_app.gui import Ui_MainWindow

import package.aux as aux
import package.data_input as di
import package.instance as inst
import package.model as md
import package.params as params
import package.heuristics as heur
import os
import arrow
import pprint as pp
from io import StringIO

class MainWindow_EXCEC():

    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(MainWindow)
        self.ui.readExcel.clicked.connect(self.choose_file)
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
        self.read_file()
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

    def generate_solution(self):
        self.input_data['parameters']['start'] = \
            self.ui.start_date.text()
        self.input_data['parameters']['end'] = \
            self.ui.end_date.text()
        options = {
            'timeLimit': int(self.ui.max_time.text())
            , 'gap': 0
            , 'solver': "CPLEX"
            , 'path':
                os.path.join(params.PATHS['experiments'], aux.get_timestamp()) + '/'
        }
        instance = inst.Instance(self.input_data)
        heur_obj = heur.Greedy(instance)
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        heur_obj.solve()
        check = heur_obj.check_solution()
        if len(check) == 0:
            self.ui.solution_log.setText("Everthing went fine!")
        else:
            self.ui.solution_log.setText(pp.pprint(check))
        try:
            heur_obj.export_solution(self.solution_path, 'solution')
        except:
            print("Solution could not be exported")
        sys.stdout = old_stdout
        self.ui.solution_log.setText(mystdout.getvalue())

if __name__ == "__main__":
    MainWindow_EXCEC()




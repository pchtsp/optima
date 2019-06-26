import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from desktop_app.gui import Ui_MainWindow

import package.auxiliar as aux
import package.data_input as di
import package.instance as inst
import package.solution as sol
import package.params as params
import os
import arrow
import pprint as pp
import logging
from io import StringIO
import multiprocessing as multi
import json

import package.template_data as td
import package.exec as exec
import package.heuristics_maintfirst as mf
import package.experiment as exp
import pytups.superdict as sd

# TODO: add dialog when it finds a solution
# TODO: import solution
# TODO: check solution
# TODO: export solution
# TODO: generate graph?
# TODO:


class MainWindow_EXCEC():

    def __init__(self, options):
        self.options = options
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(MainWindow)
        self.ui.chooseFile.clicked.connect(self.choose_file)
        self.solution_path = options['output_template_path']
        self.ui.excel_path.setText(options['input_template_path'])
        self.ui.max_time.setText(str(options['timeLimit']))

        self.instance = None
        self.solution = None

        self.input_data = {}

        # below buttons:
        self.ui.generateSolution.clicked.connect(self.generate_solution)
        self.ui.checkSolution.clicked.connect(self.check_solution)
        self.ui.exportSolution.clicked.connect(self.export_solution)
        self.ui.exportSolution_to.clicked.connect(self.export_solution_to)


        MainWindow.show()
        sys.exit(app.exec_())

    def choose_file(self):
        QFileDialog = QtWidgets.QFileDialog
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        dirName = QFileDialog.getExistingDirectory(
            caption="QFileDialog.getOpenFileName()",
            directory=params.PATHS['data'],
            options=options)
        # filter = "Excel files (*.xlsx *.xlsm);;All Files (*)",
        if not dirName:
            return False
        exec.udpdate_case_read_options(self.options, dirName + '/')
        self.ui.excel_path.setText(dirName)
        self.load_template()

        return True

    def read_dir(self):
        # input_file = params.PATHS['data'] + 'raw/parametres_DGA_final.xlsm'
        # self.input_data = di.get_model_data(self.input_path)
        # historic_data = di.generainput_pathte_solution_from_source()
        # self.input_data = di.combine_data_states(self.input_data, historic_data)
        # black_list = ['O8']  # this task has less candidates than what it asks.
        # self.input_data['tasks'] = \
        #     {k: v for k, v in self.input_data['tasks'].items() if k not in black_list}
        # start, end = self.input_data['parameters']['start'], self.input_data['parameters']['end']
        # self.ui.start_date.setDate(arrow.get(start).datetime)
        # self.ui.end_date.setDate(arrow.get(end).datetime)
        self.load_template()

    def show_message(self, title, text):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(text)
        # msg.setInformativeText("No template_in file found in directory.")
        msg.setWindowTitle(title)
        retval = msg.exec_()
        return

    # def activate_label(self, label, set_active=True):


    def load_template(self):
        options = self.options
        if not os.path.exists(options['input_template_path']):
            self.show_message(title="Missing files", text="No template_in.xlsx file found in directory.")
            return
        try:
            model_data = td.import_input_template(options['input_template_path'])
        except:
            self.show_message(title="Error in template_in.xlsx", text="There's been an error reading the input file.")
            return
        self.instance = inst.Instance(model_data)
        # we update labels with status:
        self.ui.instCheck.setText('Instance loaded')
        self.ui.instCheck.setStyleSheet("QLabel { color : green; }")

        # TODO: load solution from files?
        if os.path.exists(options['output_template_path']):
            sol_data = td.import_output_template(options['output_template_path'])
            self.solution = sol.Solution(sol_data)
            self.ui.solCheck.setText('Solution loaded')
            self.ui.solCheck.setStyleSheet("QLabel { color : green; }")

        return True

    def check_solution(self):
        if not self.solution:
            self.show_message(title="Missing files", text="No solution is loaded, can't verify it.")
            return
        experiment = exp.Experiment(self.instance, self.solution)
        errors = experiment.check_solution()
        # errors = sd.SuperDict.from_dict(errors)
        errors = {k: v.to_dictdict() for k, v in errors.items()}
        text = json.dumps(errors, cls=di.MyEncoder, indent=4)
        msg = ScrollMessageBox(QtWidgets.QMessageBox.Critical,"The following errors were found:", text)
        return

    def generate_solution(self):
        # exec.config_and_solve(self.options)
        options = self.options
        if not self.instance:
            self.show_message(title="Loading needed", text='No instance loaded, so not possible to solve.')
            return
        experiment = mf.MaintenanceFirst(self.instance, solution=self.solution)
        output_path = options['path']

        # TODO: update log live and use worker with multiprocessings
        # options['log_handler'] = QPlainTextEditLogger(self)
        self.solution = experiment.solve(options)
        self.ui.solCheck.setText('Solution loaded')
        self.ui.solCheck.setStyleSheet("QLabel { color : green; }")
        # def _dummy_run(experiment, options):
        #     experiment.solve(options)
        # p = multi.Process(target=_dummy_run, args=[experiment, options])
        # p.start()
        # while p.is_alive():
        #     continue
        # old_stdout = sys.stdout
        # sys.stdout = mystdout = StringIO()
        # errors = experiment.check_solution()
        # errors = {k: v.to_dictdict() for k, v in errors.items()}
        #
        # if len(errors) == 0:
        #     self.ui.solution_log.setText("Everthing went fine!")
        # else:
        #     self.ui.solution_log.setText(pp.pprint(errors))
        log_path = os.path.join(output_path, 'output.log')
        try:
            with open(log_path) as f:
                res = f.read()
        except:
            print('Error reading log file')
            res = ''
        self.ui.solution_log.setText(res)

    def export_solution_gen(self, output_path, export_input=False):
        if not os.path.exists(output_path) or not os.path.isdir(output_path):
            self.show_message('Error', "Path doesn't exist or is not a directory.")
            return
        if (not self.instance or not self.solution):
            self.show_message('Error', 'No solution can be exported because there is no loaded solution.')
            return
        experiment = exp.Experiment(self.instance, self.solution)
        _kwags = dict(file_type='json', exclude_aux=True)
        di.export_data(output_path, experiment.instance.data, name="data_in", **_kwags)
        di.export_data(output_path, experiment.solution.data, name="data_out", **_kwags)
        _dir = os.path.join(output_path, 'template_out.xlsx')
        td.export_output_template(_dir, experiment.solution.data)
        if export_input:
            _dir = os.path.join(output_path, 'template_in.xlsx')
            td.export_input_template(_dir, experiment.instance.data)

    def export_solution(self):
        output_path = self.options['path']
        self.export_solution_gen(output_path)
        return

    def export_solution_to(self):
        QFileDialog = QtWidgets.QFileDialog
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        dirName = QFileDialog.getExistingDirectory(
            caption="QFileDialog.getOpenFileName()",
            directory=params.PATHS['data'],
            options=options)
        # filter = "Excel files (*.xlsx *.xlsm);;All Files (*)",
        if not dirName:
            return False
        self.export_solution_gen(dirName, export_input=True)
        return

class QPlainTextEditLogger(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.widget = parent.ui.solution_log
        self.widget.setReadOnly(True)

    def emit(self, record):
        msg = self.format(record)
        self.widget.append(msg)

class ScrollMessageBox(QtWidgets.QMessageBox):
   def __init__(self, *args, **kwargs):
       QtWidgets.QMessageBox.__init__(self, *args, **kwargs)
       chldn = self.children()
       scrll = QtWidgets.QScrollArea(self)
       scrll.setWidgetResizable(True)
       grd = self.findChild(QtWidgets.QGridLayout)
       lbl = QtWidgets.QLabel(chldn[1].text(), self)
       lbl.setWordWrap(True)
       scrll.setWidget(lbl)
       scrll.setMinimumSize (400,200)
       grd.addWidget(scrll,0,1)
       chldn[1].setText('')
       self.exec_()

# class ScrollMessageBox(QtWidgets.QMessageBox):
#    def __init__(self, *args, **kwargs):
#         QtWidgets.QMessageBox.__init__(self, *args, **kwargs)
#         scroll = QtWidgets.QScrollArea(self)
#         scroll.setWidgetResizable(True)
#         self.content = QtWidgets.QWidget()
#         scroll.setWidget(self.content)
#         lay = QtWidgets.QVBoxLayout(self.content)
#         for item in l:
#            lay.addWidget(QtWidgets.QLabel(item, self))
#         self.layout().addWidget(scroll, 0, 0, 1, self.layout().columnCount())
#         self.setStyleSheet("QScrollArea{min-width:300 px; min-height: 400px}")

if __name__ == "__main__":
    # to compile desktop_app.gui, we need the following:
    # pyuic5 -o filename.py file.ui
    # if we add -x flag we make it executable
    # example: pyuic5 gui.ui -o gui.py
    MainWindow_EXCEC({})




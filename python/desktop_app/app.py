import sys
# import PyQt5
from PyQt5 import QtWidgets, QtCore, QtGui

import os
import logging
import json
import package.auxiliar as aux

import package.instance as inst
import package.solution as sol
import package.params as params
import package.exec as exec
import package.experiment as exp

import data.data_input as di
import data.template_data as td

import solvers.heuristics_maintfirst as mf
import desktop_app.gui as gui
import reports.gantt as gantt


# TODO: integrate json viewer
# TODO: deploy icon
# TODO: correct gantt


class MainWindow_EXCEC():

    def __init__(self, options):
        self.options = options
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()

        # set icon
        if getattr(sys, 'frozen', False):
            scriptDir = sys._MEIPASS
            self.examplesDir = scriptDir + '/examples/'
        else:
            scriptDir = os.path.dirname(os.path.realpath(__file__))
            self.examplesDir = scriptDir + '/../../data/template/'
        icon_path = os.path.join(scriptDir, "plane.ico")
        MainWindow.setWindowIcon(QtGui.QIcon(icon_path))

        self.ui = gui.Ui_MainWindow()
        self.ui.setupUi(MainWindow)
        self.solution_path = options['output_template_path']

        self.ui.excel_path.setText(options['input_template_path'])
        self.update_ui()
        # self.ui.max_time.setText(str(options['timeLimit']))

        self.instance = None
        self.solution = None

        # menu actions:
        self.ui.actionOpen_from.triggered.connect(self.choose_file)
        self.ui.actionSave.triggered.connect(self.export_solution)
        self.ui.actionSave_As.triggered.connect(self.export_solution_to)
        self.ui.actionExit.triggered.connect(QtCore.QCoreApplication.exit)

        # below buttons:
        self.ui.chooseFile.clicked.connect(self.choose_file)
        self.ui.generateSolution.clicked.connect(self.generate_solution)
        self.ui.checkSolution.clicked.connect(self.check_solution)
        self.ui.exportSolution.clicked.connect(self.export_solution)
        self.ui.exportSolution_to.clicked.connect(self.export_solution_to)
        self.ui.generateGantt.clicked.connect(self.generate_gantt)

        # other
        self.ui.max_time.textEdited.connect(self.update_options)
        # self.ui.start_date.valueChanged.connect(self.update_options)
        self.ui.num_period.textEdited.connect(self.update_options)

        MainWindow.show()
        sys.exit(app.exec_())

    def update_options(self):
        try:
            self.options['timeLimit'] = float(self.ui.max_time.text())
            # self.options['start'] = self.ui.start_date.text()
            self.options['num_period'] = float(self.ui.num_period.text())
        except:
            return 0

    # def update_time(self):
    #     try:
    #         self.options['timeLimit'] = float(self.ui.max_time.text())
    #     except:
    #         return 0
    #     return 1

    def update_ui(self):
        aux.month_to_arrow(self.options['start'])
        self.ui.max_time.setText(str(self.options['timeLimit']))
        # self.ui.start_date.setDate(str())
        self.ui.num_period.setText(str(self.options['num_period']))
        return 1

    def choose_file(self):
        QFileDialog = QtWidgets.QFileDialog
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        dirName = QFileDialog.getExistingDirectory(
            caption="Choose a directory to load",
            directory=self.examplesDir,
            options=options)
        # filter = "Excel files (*.xlsx *.xlsm);;All Files (*)",
        if not dirName:
            return False
        exec.udpdate_case_read_options(self.options, dirName + '/')
        self.ui.excel_path.setText(dirName)
        self.load_template()
        self.update_ui()
        return True

    def read_dir(self):
        self.load_template()

    def show_message(self, title, text, icon='critical'):
        msg = QtWidgets.QMessageBox()
        if icon=='critical':
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
        experiment = mf.MaintenanceFirst(self.instance, self.solution)
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
        if self.solution:
            self.show_message('Finished', 'A solution was found.', icon='Success')
        else:
            self.show_message('Problem occured', 'A solution was not found.')
        return 1

    def export_solution_gen(self, output_path, export_input=False):
        if not os.path.exists(output_path) or not os.path.isdir(output_path):
            self.show_message('Error', "Path doesn't exist or is not a directory.")
            return 0
        if (not self.instance or not self.solution):
            self.show_message('Error', 'No solution can be exported because there is no loaded solution.')
            return 0
        experiment = exp.Experiment(self.instance, self.solution)
        _kwags = dict(file_type='json', exclude_aux=True)
        di.export_data(output_path, experiment.instance.data, name="data_in", **_kwags)
        di.export_data(output_path, experiment.solution.data, name="data_out", **_kwags)
        _dir = os.path.join(output_path, 'template_out.xlsx')
        td.export_output_template(_dir, experiment.solution.data)
        if export_input:
            _dir = os.path.join(output_path, 'template_in.xlsx')
            td.export_input_template(_dir, experiment.instance.data)
        self.show_message('Success', 'Solution successfully exported.', icon='Success')
        return 1

    def export_solution(self):
        output_path = self.options['path']
        self.export_solution_gen(output_path)
        return 1

    def export_solution_to(self):
        QFileDialog = QtWidgets.QFileDialog
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        dirName = QFileDialog.getExistingDirectory(
            caption="QFileDialog.getOpenFileName()",
            directory=self.options['path'],
            options=options)
        # filter = "Excel files (*.xlsx *.xlsm);;All Files (*)",
        if not dirName:
            return False
        self.export_solution_gen(dirName, export_input=True)
        return 1

    def generate_gantt(self):
        if not (self.instance and self.solution):
            self.show_message('Error', 'Gantt cannot be created because a complete instance is needed.')
            return 0
        path = self.options['path']
        try:
            gantt.make_gantt_from_experiment(path=path)
        except ValueError:
            self.show_message('Error', 'A problem occurred. Be sure to have a valid solution exported in the directory.')
            return 0
        return 1

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




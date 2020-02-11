# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'desktop_app/gui.ui',
# licensing of 'desktop_app/gui.ui' applies.
#
# Created: Fri Dec  6 12:57:54 2019
#      by: pyside2-uic  running on PySide2 5.13.0
#
# WARNING! All changes made in this file will be lost!

from PySide2 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(668, 559)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.Config = QtWidgets.QWidget()
        self.Config.setObjectName("Config")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.Config)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.chooseFile = QtWidgets.QPushButton(self.Config)
        self.chooseFile.setDefault(False)
        self.chooseFile.setObjectName("chooseFile")
        self.horizontalLayout_2.addWidget(self.chooseFile)
        self.excel_path = QtWidgets.QLabel(self.Config)
        self.excel_path.setMinimumSize(QtCore.QSize(300, 0))
        self.excel_path.setBaseSize(QtCore.QSize(0, 0))
        self.excel_path.setText("")
        self.excel_path.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextEditable)
        self.excel_path.setObjectName("excel_path")
        self.horizontalLayout_2.addWidget(self.excel_path)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.instCheck = QtWidgets.QLabel(self.Config)
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(True)
        self.instCheck.setFont(font)
        self.instCheck.setStyleSheet("QLabel { color : red; }")
        self.instCheck.setTextFormat(QtCore.Qt.AutoText)
        self.instCheck.setObjectName("instCheck")
        self.verticalLayout_2.addWidget(self.instCheck)
        self.solCheck = QtWidgets.QLabel(self.Config)
        font = QtGui.QFont()
        font.setWeight(75)
        font.setBold(True)
        self.solCheck.setFont(font)
        self.solCheck.setStyleSheet("QLabel { color : red; }")
        self.solCheck.setObjectName("solCheck")
        self.verticalLayout_2.addWidget(self.solCheck)
        self.horizontalLayout_2.addLayout(self.verticalLayout_2)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setObjectName("gridLayout")
        self.max_time_lab = QtWidgets.QLabel(self.Config)
        self.max_time_lab.setObjectName("max_time_lab")
        self.gridLayout.addWidget(self.max_time_lab, 2, 0, 1, 1)
        self.num_period = QtWidgets.QLineEdit(self.Config)
        self.num_period.setEnabled(False)
        self.num_period.setObjectName("num_period")
        self.gridLayout.addWidget(self.num_period, 1, 1, 1, 1)
        self.log_level = QtWidgets.QComboBox(self.Config)
        self.log_level.setObjectName("log_level")
        self.log_level.addItem("")
        self.log_level.addItem("")
        self.gridLayout.addWidget(self.log_level, 3, 1, 1, 1)
        self.log_level_lab = QtWidgets.QLabel(self.Config)
        self.log_level_lab.setObjectName("log_level_lab")
        self.gridLayout.addWidget(self.log_level_lab, 3, 0, 1, 1)
        self.num_periods_lab = QtWidgets.QLabel(self.Config)
        self.num_periods_lab.setObjectName("num_periods_lab")
        self.gridLayout.addWidget(self.num_periods_lab, 1, 0, 1, 1)
        self.start_date = QtWidgets.QDateEdit(self.Config)
        self.start_date.setEnabled(False)
        self.start_date.setCurrentSection(QtWidgets.QDateTimeEdit.YearSection)
        self.start_date.setCalendarPopup(True)
        self.start_date.setTimeSpec(QtCore.Qt.TimeZone)
        self.start_date.setObjectName("start_date")
        self.gridLayout.addWidget(self.start_date, 0, 1, 1, 1)
        self.start_date_lab = QtWidgets.QLabel(self.Config)
        self.start_date_lab.setObjectName("start_date_lab")
        self.gridLayout.addWidget(self.start_date_lab, 0, 0, 1, 1)
        self.max_time = QtWidgets.QLineEdit(self.Config)
        self.max_time.setEnabled(True)
        self.max_time.setMaximumSize(QtCore.QSize(167772, 16777215))
        self.max_time.setMaxLength(4)
        self.max_time.setObjectName("max_time")
        self.gridLayout.addWidget(self.max_time, 2, 1, 1, 1)
        self.reuse_sol_tab = QtWidgets.QLabel(self.Config)
        self.reuse_sol_tab.setObjectName("reuse_sol_tab")
        self.gridLayout.addWidget(self.reuse_sol_tab, 4, 0, 1, 1)
        self.reuse_sol = QtWidgets.QCheckBox(self.Config)
        self.reuse_sol.setEnabled(False)
        self.reuse_sol.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.reuse_sol.setText("")
        self.reuse_sol.setObjectName("reuse_sol")
        self.gridLayout.addWidget(self.reuse_sol, 4, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.generateSolution = QtWidgets.QPushButton(self.Config)
        self.generateSolution.setDefault(False)
        self.generateSolution.setObjectName("generateSolution")
        self.horizontalLayout.addWidget(self.generateSolution)
        self.generateSolution_missions = QtWidgets.QPushButton(self.Config)
        self.generateSolution_missions.setObjectName("generateSolution_missions")
        self.horizontalLayout.addWidget(self.generateSolution_missions)
        self.checkSolution = QtWidgets.QPushButton(self.Config)
        self.checkSolution.setObjectName("checkSolution")
        self.horizontalLayout.addWidget(self.checkSolution)
        self.exportSolution = QtWidgets.QPushButton(self.Config)
        self.exportSolution.setDefault(False)
        self.exportSolution.setObjectName("exportSolution")
        self.horizontalLayout.addWidget(self.exportSolution)
        self.exportSolution_to = QtWidgets.QPushButton(self.Config)
        self.exportSolution_to.setDefault(False)
        self.exportSolution_to.setObjectName("exportSolution_to")
        self.horizontalLayout.addWidget(self.exportSolution_to)
        self.generateGantt = QtWidgets.QPushButton(self.Config)
        self.generateGantt.setObjectName("generateGantt")
        self.horizontalLayout.addWidget(self.generateGantt)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.solution_log = QtWidgets.QTextBrowser(self.Config)
        self.solution_log.setObjectName("solution_log")
        self.verticalLayout.addWidget(self.solution_log)
        self.horizontalLayout_4.addLayout(self.verticalLayout)
        self.tabWidget.addTab(self.Config, "")
        self.horizontalLayout_3.addWidget(self.tabWidget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 668, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen_from = QtWidgets.QAction(MainWindow)
        self.actionOpen_from.setObjectName("actionOpen_from")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.menuFile.addAction(self.actionOpen_from)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtWidgets.QApplication.translate("MainWindow", "OPTIMA v0.20200210", None, -1))
        self.Config.setToolTip(QtWidgets.QApplication.translate("MainWindow", "<html><head/><body><p>configuration</p></body></html>", None, -1))
        self.chooseFile.setText(QtWidgets.QApplication.translate("MainWindow", "Browse", None, -1))
        self.instCheck.setText(QtWidgets.QApplication.translate("MainWindow", "No instance loaded", None, -1))
        self.solCheck.setText(QtWidgets.QApplication.translate("MainWindow", "No solution loaded", None, -1))
        self.max_time_lab.setText(QtWidgets.QApplication.translate("MainWindow", "Max solving time                                                                                                         ", None, -1))
        self.log_level.setItemText(0, QtWidgets.QApplication.translate("MainWindow", "INFO", None, -1))
        self.log_level.setItemText(1, QtWidgets.QApplication.translate("MainWindow", "DEBUG", None, -1))
        self.log_level_lab.setText(QtWidgets.QApplication.translate("MainWindow", "Logging level", None, -1))
        self.num_periods_lab.setText(QtWidgets.QApplication.translate("MainWindow", "Number of periods", None, -1))
        self.start_date.setDisplayFormat(QtWidgets.QApplication.translate("MainWindow", "yyyy-MM", None, -1))
        self.start_date_lab.setText(QtWidgets.QApplication.translate("MainWindow", "Start date", None, -1))
        self.reuse_sol_tab.setText(QtWidgets.QApplication.translate("MainWindow", "Reuse previous solution", None, -1))
        self.generateSolution.setText(QtWidgets.QApplication.translate("MainWindow", "Generate (maints)", None, -1))
        self.generateSolution_missions.setText(QtWidgets.QApplication.translate("MainWindow", "Generate (missions)", None, -1))
        self.checkSolution.setText(QtWidgets.QApplication.translate("MainWindow", "Check solution", None, -1))
        self.exportSolution.setText(QtWidgets.QApplication.translate("MainWindow", "Export solution", None, -1))
        self.exportSolution_to.setText(QtWidgets.QApplication.translate("MainWindow", "Export solution to", None, -1))
        self.generateGantt.setText(QtWidgets.QApplication.translate("MainWindow", "Draw gantt chart", None, -1))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Config), QtWidgets.QApplication.translate("MainWindow", "Config", None, -1))
        self.menuFile.setTitle(QtWidgets.QApplication.translate("MainWindow", "File", None, -1))
        self.actionOpen_from.setText(QtWidgets.QApplication.translate("MainWindow", "Open from...", None, -1))
        self.actionExit.setText(QtWidgets.QApplication.translate("MainWindow", "Exit", None, -1))
        self.actionSave.setText(QtWidgets.QApplication.translate("MainWindow", "Export", None, -1))
        self.actionSave_As.setText(QtWidgets.QApplication.translate("MainWindow", "Export As...", None, -1))


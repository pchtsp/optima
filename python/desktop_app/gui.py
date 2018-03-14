# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(593, 488)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 10, 781, 541))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 10, 582, 384))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 10, 10, 10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.readExcel = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.readExcel.setDefault(False)
        self.readExcel.setObjectName("readExcel")
        self.verticalLayout.addWidget(self.readExcel)
        self.excel_path = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.excel_path.setText("")
        self.excel_path.setObjectName("excel_path")
        self.verticalLayout.addWidget(self.excel_path)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout.setObjectName("gridLayout")
        self.labe45 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.labe45.setObjectName("labe45")
        self.gridLayout.addWidget(self.labe45, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 0, 1, 1)
        self.start_date = QtWidgets.QDateEdit(self.verticalLayoutWidget)
        self.start_date.setCurrentSection(QtWidgets.QDateTimeEdit.MonthSection)
        self.start_date.setCalendarPopup(True)
        self.start_date.setTimeSpec(QtCore.Qt.TimeZone)
        self.start_date.setObjectName("start_date")
        self.gridLayout.addWidget(self.start_date, 0, 1, 1, 1)
        self.end_date = QtWidgets.QDateEdit(self.verticalLayoutWidget)
        self.end_date.setCurrentSection(QtWidgets.QDateTimeEdit.MonthSection)
        self.end_date.setCalendarPopup(True)
        self.end_date.setObjectName("end_date")
        self.gridLayout.addWidget(self.end_date, 1, 1, 1, 1)
        self.max_time = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.max_time.setEnabled(True)
        self.max_time.setMaximumSize(QtCore.QSize(167772, 16777215))
        self.max_time.setMaxLength(4)
        self.max_time.setObjectName("max_time")
        self.gridLayout.addWidget(self.max_time, 2, 1, 1, 1)
        self.label2345 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label2345.setObjectName("label2345")
        self.gridLayout.addWidget(self.label2345, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.generateSolution = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.generateSolution.setDefault(False)
        self.generateSolution.setObjectName("generateSolution")
        self.verticalLayout.addWidget(self.generateSolution)
        self.solution_log = QtWidgets.QTextBrowser(self.verticalLayoutWidget)
        self.solution_log.setObjectName("solution_log")
        self.verticalLayout.addWidget(self.solution_log)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabWidget.addTab(self.tab_2, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 593, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.readExcel.setText(_translate("MainWindow", "Read Excel"))
        self.labe45.setText(_translate("MainWindow", "End date"))
        self.label_3.setText(_translate("MainWindow", "Max time                                                                                                         "))
        self.start_date.setDisplayFormat(_translate("MainWindow", "yyyy-MM"))
        self.end_date.setDisplayFormat(_translate("MainWindow", "yyyy-MM"))
        self.label2345.setText(_translate("MainWindow", "Start date"))
        self.generateSolution.setText(_translate("MainWindow", "generate solution"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Tab 1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Page"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())



# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'gui.ui'
##
## Created by: Qt User Interface Compiler version 6.8.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateEdit,
    QDateTimeEdit, QGridLayout, QHBoxLayout, QLabel,
    QLayout, QLineEdit, QMainWindow, QMenu,
    QMenuBar, QPushButton, QSizePolicy, QStatusBar,
    QTabWidget, QTextBrowser, QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(668, 559)
        self.actionOpen_from = QAction(MainWindow)
        self.actionOpen_from.setObjectName(u"actionOpen_from")
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.actionSave = QAction(MainWindow)
        self.actionSave.setObjectName(u"actionSave")
        self.actionSave_As = QAction(MainWindow)
        self.actionSave_As.setObjectName(u"actionSave_As")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_3 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setSizeConstraint(QLayout.SetNoConstraint)
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.Config = QWidget()
        self.Config.setObjectName(u"Config")
        self.horizontalLayout_4 = QHBoxLayout(self.Config)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.chooseFile = QPushButton(self.Config)
        self.chooseFile.setObjectName(u"chooseFile")

        self.horizontalLayout_2.addWidget(self.chooseFile)

        self.excel_path = QLabel(self.Config)
        self.excel_path.setObjectName(u"excel_path")
        self.excel_path.setMinimumSize(QSize(300, 0))
        self.excel_path.setBaseSize(QSize(0, 0))
        self.excel_path.setTextInteractionFlags(Qt.LinksAccessibleByMouse|Qt.TextEditable)

        self.horizontalLayout_2.addWidget(self.excel_path)

        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.instCheck = QLabel(self.Config)
        self.instCheck.setObjectName(u"instCheck")
        font = QFont()
        font.setBold(True)
        self.instCheck.setFont(font)
        self.instCheck.setStyleSheet(u"QLabel { color : red; }")
        self.instCheck.setTextFormat(Qt.AutoText)

        self.verticalLayout_2.addWidget(self.instCheck)

        self.solCheck = QLabel(self.Config)
        self.solCheck.setObjectName(u"solCheck")
        self.solCheck.setFont(font)
        self.solCheck.setStyleSheet(u"QLabel { color : red; }")

        self.verticalLayout_2.addWidget(self.solCheck)


        self.horizontalLayout_2.addLayout(self.verticalLayout_2)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.max_time_lab = QLabel(self.Config)
        self.max_time_lab.setObjectName(u"max_time_lab")

        self.gridLayout.addWidget(self.max_time_lab, 2, 0, 1, 1)

        self.num_period = QLineEdit(self.Config)
        self.num_period.setObjectName(u"num_period")
        self.num_period.setEnabled(False)

        self.gridLayout.addWidget(self.num_period, 1, 1, 1, 1)

        self.log_level = QComboBox(self.Config)
        self.log_level.addItem("")
        self.log_level.addItem("")
        self.log_level.setObjectName(u"log_level")

        self.gridLayout.addWidget(self.log_level, 3, 1, 1, 1)

        self.log_level_lab = QLabel(self.Config)
        self.log_level_lab.setObjectName(u"log_level_lab")

        self.gridLayout.addWidget(self.log_level_lab, 3, 0, 1, 1)

        self.num_periods_lab = QLabel(self.Config)
        self.num_periods_lab.setObjectName(u"num_periods_lab")

        self.gridLayout.addWidget(self.num_periods_lab, 1, 0, 1, 1)

        self.start_date = QDateEdit(self.Config)
        self.start_date.setObjectName(u"start_date")
        self.start_date.setEnabled(False)
        self.start_date.setCurrentSection(QDateTimeEdit.YearSection)
        self.start_date.setCalendarPopup(True)
        self.start_date.setTimeSpec(Qt.TimeZone)

        self.gridLayout.addWidget(self.start_date, 0, 1, 1, 1)

        self.start_date_lab = QLabel(self.Config)
        self.start_date_lab.setObjectName(u"start_date_lab")

        self.gridLayout.addWidget(self.start_date_lab, 0, 0, 1, 1)

        self.max_time = QLineEdit(self.Config)
        self.max_time.setObjectName(u"max_time")
        self.max_time.setEnabled(True)
        self.max_time.setMaximumSize(QSize(167772, 16777215))
        self.max_time.setMaxLength(4)

        self.gridLayout.addWidget(self.max_time, 2, 1, 1, 1)

        self.reuse_sol_tab = QLabel(self.Config)
        self.reuse_sol_tab.setObjectName(u"reuse_sol_tab")

        self.gridLayout.addWidget(self.reuse_sol_tab, 4, 0, 1, 1)

        self.reuse_sol = QCheckBox(self.Config)
        self.reuse_sol.setObjectName(u"reuse_sol")
        self.reuse_sol.setEnabled(False)
        self.reuse_sol.setLayoutDirection(Qt.LeftToRight)

        self.gridLayout.addWidget(self.reuse_sol, 4, 1, 1, 1)


        self.verticalLayout.addLayout(self.gridLayout)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.generateSolution = QPushButton(self.Config)
        self.generateSolution.setObjectName(u"generateSolution")

        self.horizontalLayout.addWidget(self.generateSolution)

        self.generateSolution_missions = QPushButton(self.Config)
        self.generateSolution_missions.setObjectName(u"generateSolution_missions")

        self.horizontalLayout.addWidget(self.generateSolution_missions)

        self.checkSolution = QPushButton(self.Config)
        self.checkSolution.setObjectName(u"checkSolution")

        self.horizontalLayout.addWidget(self.checkSolution)

        self.exportSolution = QPushButton(self.Config)
        self.exportSolution.setObjectName(u"exportSolution")

        self.horizontalLayout.addWidget(self.exportSolution)

        self.exportSolution_to = QPushButton(self.Config)
        self.exportSolution_to.setObjectName(u"exportSolution_to")

        self.horizontalLayout.addWidget(self.exportSolution_to)

        self.generateGantt = QPushButton(self.Config)
        self.generateGantt.setObjectName(u"generateGantt")

        self.horizontalLayout.addWidget(self.generateGantt)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.solution_log = QTextBrowser(self.Config)
        self.solution_log.setObjectName(u"solution_log")

        self.verticalLayout.addWidget(self.solution_log)


        self.horizontalLayout_4.addLayout(self.verticalLayout)

        self.tabWidget.addTab(self.Config, "")

        self.horizontalLayout_3.addWidget(self.tabWidget)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 668, 21))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.actionOpen_from)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)
        self.chooseFile.setDefault(False)
        self.generateSolution.setDefault(False)
        self.exportSolution.setDefault(False)
        self.exportSolution_to.setDefault(False)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"OPTIMA v0.20200317", None))
        self.actionOpen_from.setText(QCoreApplication.translate("MainWindow", u"Open from...", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.actionSave.setText(QCoreApplication.translate("MainWindow", u"Export", None))
        self.actionSave_As.setText(QCoreApplication.translate("MainWindow", u"Export As...", None))
#if QT_CONFIG(tooltip)
        self.Config.setToolTip(QCoreApplication.translate("MainWindow", u"<html><head/><body><p>configuration</p></body></html>", None))
#endif // QT_CONFIG(tooltip)
        self.chooseFile.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.excel_path.setText("")
        self.instCheck.setText(QCoreApplication.translate("MainWindow", u"No instance loaded", None))
        self.solCheck.setText(QCoreApplication.translate("MainWindow", u"No solution loaded", None))
        self.max_time_lab.setText(QCoreApplication.translate("MainWindow", u"Max solving time                                                                                                         ", None))
        self.log_level.setItemText(0, QCoreApplication.translate("MainWindow", u"INFO", None))
        self.log_level.setItemText(1, QCoreApplication.translate("MainWindow", u"DEBUG", None))

        self.log_level_lab.setText(QCoreApplication.translate("MainWindow", u"Logging level", None))
        self.num_periods_lab.setText(QCoreApplication.translate("MainWindow", u"Number of periods", None))
        self.start_date.setDisplayFormat(QCoreApplication.translate("MainWindow", u"yyyy-MM", None))
        self.start_date_lab.setText(QCoreApplication.translate("MainWindow", u"Start date", None))
        self.reuse_sol_tab.setText(QCoreApplication.translate("MainWindow", u"Reuse previous solution", None))
        self.reuse_sol.setText("")
        self.generateSolution.setText(QCoreApplication.translate("MainWindow", u"Generate (maints)", None))
        self.generateSolution_missions.setText(QCoreApplication.translate("MainWindow", u"Generate (missions)", None))
        self.checkSolution.setText(QCoreApplication.translate("MainWindow", u"Check solution", None))
        self.exportSolution.setText(QCoreApplication.translate("MainWindow", u"Export solution", None))
        self.exportSolution_to.setText(QCoreApplication.translate("MainWindow", u"Export solution to", None))
        self.generateGantt.setText(QCoreApplication.translate("MainWindow", u"Draw gantt chart", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Config), QCoreApplication.translate("MainWindow", u"Config", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
    # retranslateUi


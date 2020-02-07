# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'res/main_view.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.spinBox_amount = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_amount.setGeometry(QtCore.QRect(300, 180, 181, 31))
        self.spinBox_amount.setObjectName("spinBox_amount")
        self.label_even_odd = QtWidgets.QLabel(self.centralwidget)
        self.label_even_odd.setGeometry(QtCore.QRect(300, 230, 67, 17))
        self.label_even_odd.setObjectName("label_even_odd")
        self.pushButton_reset = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_reset.setGeometry(QtCore.QRect(300, 270, 89, 25))
        self.pushButton_reset.setObjectName("pushButton_reset")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_even_odd.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton_reset.setText(_translate("MainWindow", "reset"))


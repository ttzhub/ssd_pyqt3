# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\PythonSpace\ssd_pyqt2\MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(902, 624)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.label_imgshow = QtWidgets.QLabel(self.centralWidget)
        self.label_imgshow.setGeometry(QtCore.QRect(10, 10, 720, 480))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(11)
        self.label_imgshow.setFont(font)
        self.label_imgshow.setFrameShape(QtWidgets.QFrame.Box)
        self.label_imgshow.setText("")
        self.label_imgshow.setObjectName("label_imgshow")
        self.pushButton_start = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_start.setGeometry(QtCore.QRect(740, 160, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(11)
        self.pushButton_start.setFont(font)
        self.pushButton_start.setObjectName("pushButton_start")
        self.pushButton_end = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_end.setGeometry(QtCore.QRect(740, 290, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(11)
        self.pushButton_end.setFont(font)
        self.pushButton_end.setObjectName("pushButton_end")
        self.textEdit = QtWidgets.QTextEdit(self.centralWidget)
        self.textEdit.setGeometry(QtCore.QRect(10, 500, 881, 111))
        self.textEdit.setObjectName("textEdit")
        self.pushButton_open = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_open.setGeometry(QtCore.QRect(740, 50, 151, 31))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(11)
        self.pushButton_open.setFont(font)
        self.pushButton_open.setObjectName("pushButton_open")
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SSD300视频流检测"))
        self.pushButton_start.setText(_translate("MainWindow", "开始"))
        self.pushButton_end.setText(_translate("MainWindow", "结束"))
        self.pushButton_open.setText(_translate("MainWindow", "打开视频文件"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


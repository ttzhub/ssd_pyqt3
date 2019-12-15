from PyQt5 import QtWidgets
import os

from MainWindow import MainWindow

if __name__ == "__main__":
    import sys

    work_dir = os.path.split(sys.argv[0])[0]
    os.chdir(work_dir)

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
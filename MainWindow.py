# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""
import numpy as np
import cv2
import os

from PyQt5.QtCore import pyqtSlot, QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap

from functools import wraps

from Ui_MainWindow import Ui_MainWindow
from ssd.SSD_test import SSD_test, color_gen


def debug_class(calss_name='MainWindow'):
    # print('in debug_class')
    def debug(f):
        # print('in debug')
        @wraps(f)
        def print_debug(*args, **kwargs):
            # print('in print_debug')
            try:
                return f(*args, **kwargs)
            except Exception as e:
                print(calss_name + '.' + f.__name__ + '() 报错！')
                print('错误信息：', e)
        return print_debug
    return debug


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
    @debug_class('MainWindow')
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # 初始化界面
        self.label_imgshow.setScaledContents(True)          # 图片自适应显示
        self.img_none = np.ones((420, 720, 3), dtype=np.uint8)*255
        self.show_img(self.img_none)

        # 初始化SSD
        # 目标名称，按顺序
        self.obj_names = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                          'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                          'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                          'Sheep', 'Sofa', 'Train', 'Tvmonitor']
        # 需要显示的目标list， 用于过滤
        self.include_class = self.obj_names
        self.ssd = SSD_test(weight_path='./ssd/weights/weights_SSD300.hdf5', class_nam_list=self.obj_names)


        # 视频文件路径
        self.camera_index = None
        self.FPS = None

        # 初始化计时器
        self.timer = QTimer(self)               # 更新计时器
        self.timer.timeout.connect(self.timer_update)       # 超时信号连接对应的槽函数

    @debug_class('MainWindow')
    def show_img(self, img):
        """
        将np格式的图片显示到界面上的label中
        :param img:
        :return:
        """
        showImg = QImage(img.data, img.shape[1], img.shape[0],
                         img.shape[1] * 3,     # 每行数据个数，3通道 所以width*3
                         QImage.Format_RGB888)
        self.label_imgshow.setPixmap(QPixmap.fromImage(showImg))  # 展示图片

    @debug_class('MainWindow')
    @pyqtSlot()
    def on_pushButton_start_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        # 获取数据流
        self.cap = cv2.VideoCapture(self.camera_index)
        if self.cap.isOpened():
            # 获取视频的FPS
            # FPS ---- 每秒多少帧
            self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
            if isinstance(self.FPS, float):         # 正常获取的FPS是float
                self.FPS = int(self.FPS)            # 如果正确获取FPS就保存在变量
            else:
                self.FPS = 20                       # 没正确获取则设为 20帧/s

            # 计时器开始计时
            # 计时器的参数为 ms 为了正常速度播放，计时器的参数计算为 1/FPS * 1000 = 1000/FPS
            self.timer.start(int(1000/self.FPS))

            # 锁定开始按钮
            self.pushButton_start.setEnabled(False)
        else:
            QMessageBox.warning(self, '数据流打开警告', '数据流打开错误！\n请重新尝试。')

    @debug_class('MainWindow')
    def timer_update(self):
        """
        计时器槽函数
        :return:
        """
        if self.cap.isOpened():
            # 读取图像
            ret, self.img_scr = self.cap.read()
            # 视频读取完毕
            if not ret:
                # 计时器停止计时
                self.timer.stop()
                # 对话框提示
                QMessageBox.information(self, '播放提示', '视频已播放完毕！')
                # 释放摄像头
                if hasattr(self, 'cap'):
                    self.cap.release()
                    del self.cap
                # 释放‘开始’按钮
                self.pushButton_start.setEnabled(True)

            # 预处理图片
            # 转为RGB
            self.img_scr = cv2.cvtColor(self.img_scr, cv2.COLOR_BGR2RGB)

            # 检测
            self.preds = self.ssd.Predict(self.img_scr)
            # 过滤
            self.preds = self.filter(self.preds, inclued_class=self.include_class)
            self.img_scr = self.draw_img(self.img_scr, self.preds)

            h, w = self.img_scr.shape[:2]
            self.text = self.decode_preds(self.preds, w=w, h=h)
            self.textEdit.setText(self.text)

            # 显示图像
            self.show_img(self.img_scr)

            # 响应UI
            QApplication.processEvents()
        else:
            self.textEdit.setText('数据流未打开！！！\n请检查')
            self.resst_detector()

    @debug_class('MainWindow')
    def resst_detector(self):
        """
        重设检测器，为下一次检测准备
        :return:
        """
        # 释放摄像头
        if hasattr(self, 'cap'):
            self.cap.release()
            del self.cap
        # 释放‘开始’按钮
        self.pushButton_start.setEnabled(True)
        # 显示空白图片
        self.show_img(self.img_none)
        # 停止计时器
        self.timer.stop()

    @debug_class('MainWindow')
    @pyqtSlot()
    def on_pushButton_end_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        # 重设
        self.resst_detector()
        # 清除显示
        self.textEdit.clear()

    @debug_class('MainWindow')
    def draw_img(self, img, preds):
        """
        检测结果绘制在图像上
        :param img_scr:
        :param preds:
        :return:
        """
        h, w = img.shape[:2]
        offset = round(h * 0.02)
        text_height = (h * 0.0015)
        line_thickness = round(h * 0.005)
        text_thickness = round(h * 0.004)
        gen_color = color_gen()
        for i, pred in enumerate(preds):
            lab, score, xmin, ymin, xmax, ymax = pred
            text = self.obj_names[lab] + ' {:.3f}'.format(score)
            xmin = int(round(xmin * w))
            ymin = int(round(ymin * h))
            xmax = int(round(xmax * w))
            ymax = int(round(ymax * h))
            if ymin - offset <= 0:
                T_x = xmin + offset
                T_y = ymin + round(2.5 * offset)
            else:
                T_x = xmin + offset
                T_y = ymin - offset
            color = gen_color.__next__()
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, line_thickness)
            cv2.putText(img, text, (T_x, T_y), cv2.FONT_HERSHEY_SIMPLEX, text_height,
                        color, text_thickness)
        return img

    def decode_preds(self, preds, w, h):
        """
        将检测结果转为字符串
        :param preds:
        :return:
        """
        title = '序号\t目标\t置信度\t坐标\n'
        text_temp = ''
        for i, pred in enumerate(preds):
            lab, score, xmin, ymin, xmax, ymax = pred
            text = str(i)+'\t'+self.obj_names[lab] + '\t{:.3f}'.format(score)
            xmin = int(round(xmin * w))
            ymin = int(round(ymin * h))
            xmax = int(round(xmax * w))
            ymax = int(round(ymax * h))
            text += '\t({}, {}, {}, {})\n'.format(xmin, ymin, xmax, ymax)
            text_temp += text
        return title + text_temp

    def filter(self, preds, inclued_class):
        """
        过滤检测结果
        :param preds:
        :param inclued_class:
        :return:
        """
        out = []
        for lab, score, xmin, ymin, xmax, ymax in preds:
            if self.obj_names[lab] in inclued_class:
                out.append([lab, score, xmin, ymin, xmax, ymax])
        return out
    
    @pyqtSlot()
    def on_pushButton_open_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        # 打开文件对话框
        path = QFileDialog.getOpenFileName(self, '打开待检测视频', './', '*.avi;;*.mp4;;AllFile(*.*)', '')
        if path[0] != '':
            path = os.path.normpath(os.path.abspath(path[0]))
            self.camera_index = path
            self.textEdit.setText('{}已选中！'.format(path))
        else:
            self.textEdit.setText('当前未选中任何文件')

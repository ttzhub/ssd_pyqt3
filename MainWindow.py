# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""
import numpy as np
import cv2
import os

from PyQt5.QtCore import pyqtSlot, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap

from functools import wraps
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import RawArray, RawValue

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

class Recv_res(QThread):
    """
    检测结果接收线程
    """
    res_signal = pyqtSignal(list)
    @debug_class('Recv_res')
    def __init__(self, parent, queue:Queue):
        """
        构造函数
        :param parent: 父实例 QObj ，Qt中父实例析构相应的子线程会安全退出，不用人工处理
        :param queue: 管道
        """
        super(Recv_res, self).__init__(parent=parent)
        self.queue = queue

    @debug_class('Recv_res')
    def run(self):
        while True:
            flg, res = self.queue.get()
            print(flg, res)
            if flg == 0:    # 对应检测子进程已安全退出
                print('接收线程已安全退出！')
                self.res_signal.emit([0, res])
                break
            else:
                self.res_signal.emit([flg, res])


@debug_class('Main')
def filter(obj_names, preds, inclued_class):
    """
    过滤检测结果
    :param preds:
    :param inclued_class:
    :return:
    """
    out = []
    for lab, score, xmin, ymin, xmax, ymax in preds:
        if obj_names[lab] in inclued_class:
            out.append([lab, score, xmin, ymin, xmax, ymax])
    return out


@debug_class('Main')
def decode_preds(obj_names, preds, w, h):
    """
    将检测结果转为list     [obj1, obj2, obj3 ...]
                            obj = [name, score, xmin，ymin, xmax, ymax]
    :param preds:
    :return:
    """
    res = []
    for pred in preds:
        lab, score, xmin, ymin, xmax, ymax = pred
        xmin = int(round(xmin * w))
        ymin = int(round(ymin * h))
        xmax = int(round(xmax * w))
        ymax = int(round(ymax * h))
        res.append([obj_names[lab], score, xmin, ymin, xmax, ymax])
    return res


def detector_process(img_share, img_shape, process_flg, img_get_flg, res_queue):
    # 初始化SSD
    weight_path = './ssd/weights/weights_SSD300.hdf5'
    weight_path = os.path.normpath(os.path.abspath(weight_path))
    obj_names = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                      'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                      'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                      'Sheep', 'Sofa', 'Train', 'Tvmonitor']
    include_class = obj_names
    ssd = SSD_test(weight_path=weight_path, class_nam_list=obj_names)

    # 通知UI模型加载成功
    res_queue.put((3, '模型加载成功'))

    # 构建检测循环
    while True:
        # print('process_flg:{}  img_get_flg:{}'.format(process_flg.value, img_get_flg.value))
        # 判断检测器状态
        if process_flg.value == 0:
            print('安全退出检测进程！')
            # self.res_queue.put((0, '检测进程已安全退出！'))
            break

        # 判断共享内存当前状态
        if img_get_flg.value == 0:
            # print('开始检测！')
            try:
                img = np.array(img_share[:], dtype=np.uint8)
                img_scr = np.reshape(img, (img_shape[0], img_shape[1], 3))

                # 预处理图片

                # 检测
                preds = ssd.Predict(img_scr)
                # 过滤
                preds = filter(obj_names, preds, inclued_class=include_class)

                h, w = img_shape[:2]
                res = decode_preds(obj_names, preds, w=w, h=h)  # 列表

                # 管道返回检测结果
                res_queue.put((1, res))

            except:
                print('图片检测失败')
                res_queue.put((2, '当前图像检测失败!'))
            finally:
                # 释放图像共享内存占用
                img_get_flg.value = 1


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

        # 图像大小
        self.img_shape = (480, 720)

        # 初始化界面
        self.label_imgshow.setScaledContents(True)          # 图片自适应显示
        self.label_imgshow_res.setScaledContents(True)      # 检测结果图片自适应显示
        self.img_none = np.ones((480, 720, 3), dtype=np.uint8)*255
        self.show_img(self.img_none)

        # SSD检测初始化
        self.weight_path = './ssd/weights/weights_SSD300.hdf5'
        self.weight_path = os.path.normpath(os.path.abspath(self.weight_path))
        self.obj_names = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                          'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                          'Dog', 'Horse', 'Motorbike', 'Person', 'Pottedplant',
                          'Sheep', 'Sofa', 'Train', 'Tvmonitor']
        # 需要显示的目标list， 用于过滤
        self.include_class = self.obj_names
        # 检测子进程
        self.queue = Queue()
        # self.detector_process = SSD_detector(ssd_weight=self.weight_path,
        #                                      obj_names=self.obj_names,
        #                                      include_class=self.include_class,
        #                                      img_shape=self.img_shape,
        #                                      res_queue=self.queue)
        # 多进程之间的共享图片内存
        self.img_share = RawArray('I', self.img_shape[0] * self.img_shape[1] * 3)
        # 标识当前进程的状态，非0：保持检测；0：停止检测
        self.process_flg = RawValue('I', 1)
        # 当前图像共享内存 img_share 的状态，非0：主进程使用中；0：子进程使用中
        self.img_get_flg = RawValue('I', 1)
        self.detector_process = Process(target=detector_process,
                                        args=(self.img_share,
                                              self.img_shape,
                                              self.process_flg,
                                              self.img_get_flg,
                                              self.queue))
        self.detector_process.start()


        # 接收线程
        self.recv_thread = Recv_res(parent=self, queue=self.queue)
        # 连接信号
        self.recv_thread.res_signal.connect(self.show_res)
        self.recv_thread.start()

        # 视频文件路径
        self.camera_index = 0
        self.FPS = None

        # 初始化计时器
        self.timer = QTimer(self)               # 更新计时器
        self.timer.timeout.connect(self.timer_update)       # 超时信号连接对应的槽函数

        # 等待加载模型
        self.textEdit.setText('正在加载模型，请稍后......')
        self.pushButton_start.setEnabled(False)
        self.pushButton_open.setEnabled(False)
        self.pushButton_pause.setEnabled(False)
        self.lineEdit_cameraIndex.setEnabled(False)

        # 暂停
        self.pause = False


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
    def show_img_res(self, img):
        """
        将np格式的图片显示到界面上的label中
        :param img:
        :return:
        """
        showImg = QImage(img.data, img.shape[1], img.shape[0],
                         img.shape[1] * 3,  # 每行数据个数，3通道 所以width*3
                         QImage.Format_RGB888)
        self.label_imgshow_res.setPixmap(QPixmap.fromImage(showImg))  # 展示图片

    @debug_class('MainWindow')
    @pyqtSlot()
    def on_pushButton_start_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        # 获取数据流
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # 设置缓存为1张图片，保证获得的图像最新
        if self.cap.isOpened():
            # 尝试获取视频的FPS
            # FPS ---- 每秒多少帧
            try:
                self.FPS = self.cap.get(cv2.CAP_PROP_FPS)
                if isinstance(self.FPS, float):         # 正常获取的FPS是float
                    self.FPS = int(self.FPS)            # 如果正确获取FPS就保存在变量
                    if self.FPS == 0:
                        self.FPS = 20
            except:
                self.FPS = 20                       # 没正确获取则设为 20帧/s

            # 获取视频中图像的大小
            # ret, fame = self.cap.read()
            # self.img_shape = fame.shape[:2]

            # 计时器开始计时
            # 计时器的参数为 ms 为了正常速度播放，计时器的参数计算为 1/FPS * 1000 = 1000/FPS
            self.timer.start(int(1000/self.FPS))

            # 锁定开始按钮
            self.pushButton_start.setEnabled(False)
            # 锁定其他按钮
            self.pushButton_open.setEnabled(False)
            self.lineEdit_cameraIndex.setEnabled(False)
            # 释放暂停键
            self.pushButton_pause.setEnabled(True)
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
            # ### 视频读取完毕
            if not ret:
                # 计时器停止计时
                self.timer.stop()
                # 不检测
                self.img_get_flg.value = 1
                # 对话框提示
                QMessageBox.information(self, '播放提示', '视频已播放完毕！')
                # 释放摄像头
                if hasattr(self, 'cap'):
                    self.cap.release()
                    del self.cap
                # 释放‘开始’按钮
                self.pushButton_start.setEnabled(True)
                # 禁止暂停并初始化其功能
                self.pause = False
                self.pushButton_pause.setText('暂停')
                self.pushButton_pause.setEnabled(False)
                # 释放视频流选择
                self.pushButton_open.setEnabled(True)
                self.lineEdit_cameraIndex.setEnabled(True)
                return

            # 图像预处理
            self.img_scr = cv2.resize(self.img_scr, (self.img_shape[1], self.img_shape[0]))
            # 转为RGB
            self.img_scr = cv2.cvtColor(self.img_scr, cv2.COLOR_BGR2RGB)

            if hasattr(self, 'detector_process'):
                # ### 抽帧
                if self.img_get_flg.value == 1:
                    # print('开始抽帧')
                    self.img_temp = self.img_scr.copy()         # 用于显示检测结果
                    self.img_share[:] = self.img_scr.reshape(-1).tolist()  # 抽帧保存在中间缓存
                    self.img_get_flg.value = 0  # 不再抽取 直到检测完成
                    # print('结束抽帧')

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
        self.show_img_res(self.img_none)
        # 禁止暂停并初始化其功能
        self.pause = False
        self.pushButton_pause.setText('暂停')
        self.pushButton_pause.setEnabled(False)
        # 释放视频流更改
        self.pushButton_open.setEnabled(True)
        self.lineEdit_cameraIndex.setEnabled(True)
    
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

    @debug_class()
    def show_res(self, l):
        """
        显示结果
        :return:
        """
        flg, res = l
        # 正常检测
        if flg == 1:
            # 绘制结果依赖
            h, w = self.img_shape
            offset = round(h * 0.02)
            text_height = (h * 0.0015)
            line_thickness = round(h * 0.005)
            text_thickness = round(h * 0.004)
            gen_color = color_gen()

            # 文本显示
            head = '序号\t类名\t置信度\t位置\n'
            temp = ''
            for i, obj in enumerate(res):
                name, score, xmin, ymin, xmax, ymax = obj

                if ymin - offset <= 0:
                    T_x = xmin + offset
                    T_y = ymin + round(2.5 * offset)
                else:
                    T_x = xmin + offset
                    T_y = ymin - offset
                color = gen_color.__next__()
                self.img_temp = cv2.rectangle(self.img_temp, (xmin, ymin), (xmax, ymax), color, line_thickness)
                txt = '{:} {:.2f}'.format(name, score)
                self.img_temp = cv2.putText(self.img_temp, txt, (T_x, T_y), cv2.FONT_HERSHEY_SIMPLEX, text_height,
                            color, text_thickness)
                # 显示检测结果
                self.show_img_res(self.img_temp)

                text = '{i:}\t{name:}\t{score:.2f}\t' \
                       '({xmin:},{ymin:},{xmax:},{ymax:})\n'.format(i=i, name=name, score=score,
                                                                    xmin=xmin, ymin=ymin,
                                                                    xmax=xmax, ymax=ymax)
                temp += text

            self.textEdit.setText(head + temp)

        # 检测失败
        if flg == 2:
            self.textEdit.setText('当前帧检测失败')

        # 模型加载完成
        if flg == 3:
            self.textEdit.setText('模型加载完成！')
            self.pushButton_open.setEnabled(True)
            self.pushButton_start.setEnabled(True)
            self.lineEdit_cameraIndex.setEnabled(True)

    def closeEvent(self, a0):
        """
        关闭窗口时间函数
        :param a0:
        :return:
        """
        self.process_flg.value = 0          # 退出子进程
        self.detector_process.join()
    
    @pyqtSlot(bool)
    def on_pushButton_pause_clicked(self, checked):
        """
        暂停与开始
        
        @param checked DESCRIPTION
        @type bool
        """
        # TODO: not implemented yet
        self.pause = not self.pause
        if self.pause:      # 暂停
            self.timer.stop()
            self.pushButton_pause.setText('继续')
        else:               # 继续
            self.timer.start(int(1000 / self.FPS))
            self.pushButton_pause.setText('暂停')
    
    @pyqtSlot(str)
    def on_lineEdit_cameraIndex_textChanged(self, p0):
        """
        Slot documentation goes here.
        
        @param p0 DESCRIPTION
        @type str
        """
        # TODO: not implemented yet
        try:
            index = int(p0)
            self.camera_index = index
        except:
            self.camera_index = p0
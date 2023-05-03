import os
import random
import shutil
import sys
import threading
from datetime import datetime
from os import listdir
from time import sleep

from PyQt5.QtCore import QPoint, Qt, QTimer
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import (QApplication, QComboBox,
                             QHBoxLayout, QLabel,
                             QPushButton, QShortcut, QStyle,
                             QVBoxLayout, QWidget)


class App(QWidget):
    def __init__(self, image_path, save_path, username, times_image=1, wait_time=5, classification_time=50):
        super().__init__()
        self.title = "ImageView"

        self.username = username
        self.image_path = "{}/".format(image_path)
        self.save_path = "{}/{}/".format(save_path, username)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.predict_path = f'{self.save_path}/predict.txt'

        if not os.path.exists(self.predict_path):
            f = open(self.predict_path, "w")

        self.data_list_file = {}

        with open(f"{os.getcwd()}/player/public/dataList.csv") as data_list_file:
            for line in data_list_file.readlines():
                key, value = line.split(',')
                self.data_list_file[f"{os.getcwd()}/player/bci_image/{value.strip()}"] = int(key)

        self.image_number = 0
        self.original_files = []
        self.image_classification = []
        self.files = []

        self.timer = None
        self.timer_button = None
        self.open_eyes = None
        self.play_media = False

        self.times_image = times_image
        self.timer_play = wait_time  # Time for start the firts image
        self.timer_open_eyes = wait_time  # Time of black screen
        self.timer_start_button = wait_time  # Time of image in white screen
        self.timer_show_classification = classification_time  # Time of image classification

        layout = QVBoxLayout()
        self.label = QLabel(self)
        layout.addWidget(self.label)
        layout.addLayout(self.init_control())
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.setAttribute(Qt.WA_NoSystemBackground, True)

        self.widescreen = True
        self.setAcceptDrops(True)
        self.setWindowTitle(self.title)

        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setGeometry(700, 400, 400, 290)

        # shortcuts
        self.shortcut = QShortcut(QKeySequence("q"), self)
        self.shortcut.activated.connect(self.handle_quit)
        self.shortcut = QShortcut(QKeySequence("f"), self)
        self.shortcut.activated.connect(self.handle_fullscreen)
        self.shortcut = QShortcut(QKeySequence("s"), self)
        self.shortcut.activated.connect(self.toggle_slider)

        self.hide_control_panel()
        self.show()
        self.show_black_view()
        self.quit_app = False
        self.process_thread = threading.Thread(target=self.run_communication, args=())
        self.predict_value = None
        self.real_value = None
        self.is_green_screen = False
        self.next_step = False

    def run_communication(self):
        old_len = 1
        try:
            while True:
                if self.quit_app:
                    return
                with open(self.predict_path, 'r') as file:
                    read_len = len(file.readlines())
                    file.seek(0)
                    if old_len < read_len:
                        value = file.readlines()[-1]
                if old_len < read_len:
                    self.predict_value = int(value.split(',')[0].strip())
                    self.real_value = self.data_list_file[self.files[self.image_number - 1]]
                    if self.is_green_screen and self.predict_value == self.real_value:
                        self.next_step = True
                    old_len = read_len
                sleep(.25)
        except Exception as ex:
            print(ex)

    def toggle_slider(self):
        if self.combo_box.isVisible():
            self.hide_control_panel()
        else:
            self.show_control_panel()

    def hidden_button(self, py_button):
        py_button.hide()

    def show_button(self, py_button):
        py_button.show()

    def wheelEvent(self, event):
        m_width = self.frameGeometry().width()
        m_left = self.frameGeometry().left()
        m_top = self.frameGeometry().top()
        m_scale = event.angleDelta().y() / 5
        if self.widescreen:
            self.setGeometry(m_left, m_top, m_width + m_scale, (m_width + m_scale) / 1.778)
        else:
            self.setGeometry(m_left, m_top, m_width + m_scale, (m_width + m_scale) / 1.33)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - QPoint(self.frameGeometry().width() / 2, self.frameGeometry().height() / 2))
            event.accept()

    def show_control_panel(self):
        self.combo_box.show()
        self.py_button.show()
        m_width = self.frameGeometry().width()
        m_left = self.frameGeometry().left()
        m_top = self.frameGeometry().top()
        if self.widescreen:
            self.setGeometry(m_left, m_top, m_width, m_width / 1.55)
        else:
            self.setGeometry(m_left, m_top, m_width, m_width / 1.33)

    def hide_control_panel(self):
        self.combo_box.hide()
        self.py_button.hide()
        m_width = self.frameGeometry().width()
        m_left = self.frameGeometry().left()
        m_top = self.frameGeometry().top()
        if self.widescreen:
            self.setGeometry(m_left, m_top, m_width, m_width / 1.778)
        else:
            self.setGeometry(m_left, m_top, m_width, m_width / 1.33)

    def handle_fullscreen(self):
        if self.windowState() & Qt.WindowFullScreen:
            self.showNormal()
        else:
            self.showFullScreen()

    def handle_button(self):
        if self.play_media:
            self.handle_quit()
        else:
            self.play()

    def init_control(self):

        self.combo_box = QComboBox()
        self.combo_box.setStyleSheet("background-color: white")

        self.py_button = QPushButton(' Start', self)
        self.py_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.py_button.clicked.connect(self.handle_button)
        self.py_button.resize(50, 30)
        self.py_button.move(50, 50)
        self.py_button.setStyleSheet("background-color: white")

        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(5, 0, 5, 0)
        controlLayout.addWidget(self.combo_box)
        controlLayout.addWidget(self.py_button)

        return controlLayout

    def get_files(self):
        return [self.image_path + f for f in listdir(self.image_path) if f.find("png") >= 1]

    def save_image_classification(self):
        path = "{}classification.txt".format(self.save_path)
        with open(path, "w+") as file:
            for times in self.image_classification:
                file.write(times)
        os.chmod(path, 0o777)

    def random_list(self):
        self.files = []
        path = "{}data_random.txt".format(self.save_path)

        for image_file in self.original_files:
            for _ in range(self.times_image):
                self.files.append(image_file)

        if self.times_image != 1:
            random.shuffle(self.files)

        with open(path, "w+") as file:
            for image_file in self.files:
                file.write("{}\n".format(image_file.replace(self.image_path, "")))

        os.chmod(path, 0o777)

    def handle_quit(self):
        shutil.copyfile(f"{os.getcwd()}/player/public/dataList.csv", f"{self.save_path}/dataList.csv")
        self.stop()
        self.close()

    def stop_timer(self, timer):
        if timer:
            timer.stop()

    def play(self):
        self.image_number = 0

        self.play_media = True
        self.process_thread.start()
        self.py_button.setText("Stop")
        self.py_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))

        self.original_files = self.get_files()
        self.random_list()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_image)
        self.timer.start(self.timer_play * 1000)

    def stop(self):
        self.quit_app = True
        self.stop_timer(self.open_eyes)
        self.stop_timer(self.timer)
        self.stop_timer(self.timer_button)

        self.image_number = 0
        self.save_image_classification()

        self.play_media = False
        self.py_button.setText("Play")
        self.show_black_view()
        self.py_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def next_image(self):
        self.is_green_screen = False
        self.stop_timer(self.timer)

        if self.image_number == len(self.files):
            self.handle_quit()
        else:
            self.image_number += 1
            self.handle_open_eyes()

    def handle_open_eyes(self):

        self.show_black_view()

        self.open_eyes = QTimer(self)
        self.open_eyes.timeout.connect(self.show_image)
        self.open_eyes.start(self.timer_open_eyes * 1000)

    def show_image(self):
        self.stop_timer(self.open_eyes)

        self.hidden_classification_color()

        self.timer_button = QTimer(self)
        self.timer_button.timeout.connect(self.show_classification)
        self.timer_button.start(self.timer_start_button * 1000)

    def show_classification(self):
        self.stop_timer(self.timer_button)
        self.is_green_screen = True

        self.show_classification_color()

        self.count_interval = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.wait_time)
        self.timer.setInterval(100)
        # self.timer.start(self.timer_show_classification * 1000)
        self.timer.start(100)

    def wait_time(self):
        self.count_interval += 1
        # print(self.next_step)
        if self.next_step or self.count_interval == self.timer_show_classification * 10:
            self.next_step = False
            self.image_classification.append(f"{datetime.now()}\n")
            self.next_image()

    def show_black_view(self):
        pixmap = QPixmap(os.getcwd() + "/player/public/None.png")
        self.label.setScaledContents(True)
        pixmap = pixmap.scaled(int(self.frameGeometry().width()), int(
            self.frameGeometry().height() * 0.85), Qt.KeepAspectRatio)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setPixmap(pixmap)

    def show_classification_color(self):

        pixmap = QPixmap(self.files[self.image_number - 1])
        pixmap = pixmap.scaled(int(self.frameGeometry().width()),
                               int(self.frameGeometry().height() * 0.85) * 0.98,
                               Qt.KeepAspectRatio)

        pixmap2 = QPixmap(os.getcwd() + "/player/public/fx3.png")
        pixmap2 = pixmap2.scaled(int(self.frameGeometry().width()),
                                 int(self.frameGeometry().height() * 0.85),
                                 Qt.KeepAspectRatio)

        painter = QPainter()
        painter.begin(pixmap2)
        painter.drawPixmap(
            int(self.frameGeometry().width() * 0.85) * 0.085,
            int(self.frameGeometry().height() * 0.85) * 0.01,
            pixmap)
        painter.end()
        self.label.setPixmap(pixmap2)

        self.image_classification.append(f"{datetime.now()}\n")

        self.label.setScaledContents(False)
        self.label.setAlignment(Qt.AlignCenter)

    def hidden_classification_color(self):

        QSound.play(os.getcwd() + "/player/public/openEyes-FX.wav")
        self.label.setScaledContents(False)
        self.label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap(self.files[self.image_number - 1])
        pixmap = pixmap.scaled(int(self.frameGeometry().width()),
                               int(self.frameGeometry().height() * 0.85) * 0.98,
                               Qt.KeepAspectRatio)

        pixmap2 = QPixmap(os.getcwd() + "/player/public/fx2.png")
        pixmap2 = pixmap2.scaled(int(self.frameGeometry().width()),
                                 int(self.frameGeometry().height() * 0.85),
                                 Qt.KeepAspectRatio)

        painter = QPainter()
        painter.begin(pixmap2)
        painter.drawPixmap(
            int(self.frameGeometry().width() * 0.85) * 0.085,
            int(self.frameGeometry().height() * 0.85) * 0.01,
            pixmap)
        painter.end()
        self.label.setPixmap(pixmap2)


def main(save_path, username, times_image=1, wait_time=5, classification_time=50):
    image_path = f"{os.getcwd()}/player/bci_image"
    app = QApplication(sys.argv)
    ex = App(image_path, save_path, username, times_image, wait_time, classification_time)
    sys.exit(app.exec_())

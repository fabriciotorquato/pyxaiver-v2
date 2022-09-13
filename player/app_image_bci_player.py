import os
import shutil
import sys
from datetime import datetime
from os import listdir

from PyQt5.QtCore import QPoint, Qt, QTimer
from PyQt5.QtGui import *
from PyQt5.QtMultimedia import *
from PyQt5.QtWidgets import (QApplication, QComboBox,
                             QHBoxLayout, QLabel,
                             QPushButton, QShortcut, QStyle,
                             QVBoxLayout, QWidget)


class App(QWidget):
    def __init__(self, image_path, save_path, username):
        super().__init__()
        self.title = "ImageView"
        self.username = username
        self.image_path = "{}/".format(image_path)
        self.save_path = "{}/{}/".format(save_path, username)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.image_number = 0
        self.image_classification = []
        self.ser = None
        self.timer = None
        self.timer_button = None
        self.timer_key_button = None
        self.enabled_classification = False
        self.visible = False
        self.play_media = False
        self.files = self.get_files()

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
        # self.showMaximized()
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setGeometry(700, 400, 400, 290)

        #### shortcuts ####
        self.shortcut = QShortcut(QKeySequence("q"), self)
        self.shortcut.activated.connect(self.handle_quit)
        self.shortcut = QShortcut(QKeySequence("f"), self)
        self.shortcut.activated.connect(self.handle_fullscreen)
        self.shortcut = QShortcut(QKeySequence("s"), self)
        self.shortcut.activated.connect(self.toggle_slider)

        self.hide_control_panel()
        self.show()
        self.next_image()

    def toggle_slider(self):
        if self.combo_box.isVisible():
            self.hide_control_panel()
        else:
            self.show_control_panel()

    def show_control_panel(self):
        self.combo_box.show()
        self.py_button.show()
        m_width = self.frameGeometry().width()
        m_height = self.frameGeometry().height()
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
        m_height = self.frameGeometry().height()
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
        onlyfiles = []
        for f in listdir(self.image_path):
            if f.find("jpg") >= 1 or f.find("png") >= 1:
                onlyfiles.append(self.image_path + f)
        return onlyfiles

    def next_image(self):
        if self.visible:
            if self.image_number == len(self.files):
                self.handle_quit()
            else:
                self.image_number = self.image_number + 1
                self.stop_timer(self.timer)
                self.enabled_classification = False
                if self.image_number == 1 or (self.image_number - 1) % 10 == 0:
                    self.start_time_button()
                    self.handle_open_eyes()
                else:
                    self.show_classification()
        else:
            if self.timer:
                self.timer.start(5000)
            self.show_black_view()
            self.visible = True

    def handle_button(self):
        if self.play_media:
            self.handle_quit()
        else:
            self.play()

    def play(self):
        self.files = self.get_files()
        self.random_list()
        self.image_number = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_image)
        self.timer.start(5000)
        self.play_media = True
        self.py_button.setText("Stop")
        self.py_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))

    def stop(self):
        self.stop_timer(self.timer)
        self.enabled_classification = False
        self.image_number = 0
        self.visible = True
        self.play_media = False
        self.save_image_classification(self.image_classification)
        self.show_black_view()
        self.py_button.setText("Play")
        self.py_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def save_image_classification(self, array_file):
        path = self.save_path + "classification.txt"

        with open(path, "w+") as file:
            for (index, value) in enumerate(array_file):
                file.write(str(index) + " " + value + "\n")

        os.chmod(path, 0o777)

    def random_list(self):
        original_files = self.files
        self.files = []

        for image_file in original_files:
            for _ in range(10):
                self.files.append(image_file)

        path = self.save_path + "data_random.txt"
        with open(path, "w+") as file:
            for (index, value) in enumerate(self.files):
                file.write(str(index) + " " +
                           value.replace(self.image_path, "") + "\n")

        os.chmod(path, 0o777)

    def handle_open_eyes(self):
        self.open_eyes = QTimer(self)
        self.open_eyes.timeout.connect(self.show_image)
        self.open_eyes.start(2000)

    def show_image(self):
        QSound.play(os.getcwd() + "/player/public/openEyes-FX.wav")
        self.stop_timer(self.open_eyes)
        self.label.setScaledContents(False)
        self.label.setAlignment(Qt.AlignCenter)
        self.hidden_classification_color()

    def show_black_view(self):
        self.enabled_classification = False
        pixmap = QPixmap(os.getcwd() + "/player/public/None.png")
        self.label.setScaledContents(True)
        pixmap = pixmap.scaled(int(self.frameGeometry().width()), int(
            self.frameGeometry().height() * 0.85), Qt.KeepAspectRatio)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setPixmap(pixmap)

    def start_time_button(self):
        self.timer_button = QTimer(self)
        self.timer_button.timeout.connect(self.show_classification)
        self.timer_button.start(5000)

    def show_classification(self):
        self.show_classification_color()
        if self.timer_button:
            self.stop_timer(self.timer_button)
        self.label.setScaledContents(False)
        self.label.setAlignment(Qt.AlignCenter)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_image)
        self.timer.start(5000)

    def show_classification_color(self):
        if self.image_number % 10 == 0:
            self.visible = False
        self.enabled_classification = True
        pixmap = QPixmap(self.files[self.image_number - 1])
        pixmap = pixmap.scaled(int(self.frameGeometry().width()), int(self.frameGeometry().height() * 0.85) * 0.98,
                               Qt.KeepAspectRatio)

        pixmap2 = QPixmap(os.getcwd() + "/player/public/fx3.png")
        pixmap2 = pixmap2.scaled(int(self.frameGeometry().width()), int(self.frameGeometry().height() * 0.85),
                                 Qt.KeepAspectRatio)

        painter = QPainter()
        painter.begin(pixmap2)
        painter.drawPixmap(int(
            self.frameGeometry().width() * 0.85) * 0.085, int(
            self.frameGeometry().height() * 0.85) * 0.01, pixmap)
        painter.end()
        self.label.setPixmap(pixmap2)
        self.image_classification.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def hidden_classification_color(self):
        self.enabled_classification = False
        pixmap = QPixmap(self.files[self.image_number - 1])
        pixmap = pixmap.scaled(int(self.frameGeometry().width()), int(self.frameGeometry().height() * 0.85) * 0.98,
                               Qt.KeepAspectRatio)

        pixmap2 = QPixmap(os.getcwd() + "/player/public/fx2.png")
        pixmap2 = pixmap2.scaled(int(self.frameGeometry().width()), int(self.frameGeometry().height() * 0.85),
                                 Qt.KeepAspectRatio)

        img_aspect_ratio = pixmap.size().height()
        painter = QPainter()
        painter.begin(pixmap2)
        painter.drawPixmap(int(
            self.frameGeometry().width() * 0.85) * 0.085, int(self.frameGeometry().height() * 0.85) * 0.01, pixmap)
        painter.end()
        self.label.setPixmap(pixmap2)

    def stop_timer(self, timer):
        if timer:
            timer.stop()
            timer = None

    def hidden_button(self, py_button):
        py_button.hide()

    def show_button(self, py_button):
        py_button.show()

    def wheelEvent(self, event):
        m_width = self.frameGeometry().width()
        m_height = self.frameGeometry().height()
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

    def handle_quit(self):
        shutil.copyfile("{}/player/public/dataList.csv".format(os.getcwd()), "{}/dataList.csv".format(self.save_path))
        self.stop()
        self.close()


def main(save_path, username):
    image_path = os.getcwd() + "/player/bci_image"
    app = QApplication(sys.argv)
    ex = App(image_path, save_path, username)
    sys.exit(app.exec_())

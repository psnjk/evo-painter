from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QListWidget, QTextEdit, \
    QLineEdit, QLabel, QMessageBox, QFileDialog
import os
import json
from PIL import Image, ImageQt





class ImageProcessor():
    def __init__(self):
        self.target_image = None
        self.image_path = None

    def open_image(self, image_path):
        self.image_path = image_path
        target_image = Image.open(self.image_path).convert('RGB')
        if target_image.width < 64 or target_image.height < 64:
            popup = QMessageBox()
            popup.setWindowTitle('Error')
            popup.setText('The selected image must be larger then 64x64')
            popup.exec()
            return

        self.target_image = target_image

        if target_image.width != target_image.height:
            popup = QMessageBox()
            crop_button = popup.addButton('Crop', QMessageBox.ButtonRole.AcceptRole)
            popup.addButton('Resize', QMessageBox.ButtonRole.RejectRole)
            popup.setText('Selected image is not square')
            popup.exec()
            option = popup.clickedButton()
            if option == crop_button:
                if target_image.width > target_image.height:
                    self.target_image = target_image.crop(((target_image.width-target_image.height)/2, 0,
                                                      (target_image.width-target_image.height)/2 + target_image.height,
                                                       target_image.height))
                else:
                    self.target_image = target_image.crop(((0, (target_image.height-target_image.width)/2,
                                                       target_image.width,
                                                       (target_image.height - target_image.width)/2 + target_image.width)))

            else:
                if target_image.width < target_image.height:
                    new_size = (target_image.width, target_image.width)

                else:
                    new_size = (target_image.height, target_image.height)

                self.target_image = target_image.resize(new_size)
        print(self.target_image.size)
        self.show_target_image()


    def show_target_image(self):
        image_qt = ImageQt.ImageQt(self.target_image)
        pixmap = QPixmap.fromImage(image_qt)
        w, h = original_image.width(), original_image.height()
        pixmap = pixmap.scaled(w, h, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
        original_image.setPixmap(pixmap)





def show_noimage(image_path, image):
    image_qt = ImageQt.ImageQt(image_path)
    pixmap = QPixmap.fromImage(image_qt)
    w, h = original_image.width(), original_image.height()
    pixmap = pixmap.scaled(w, h, aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio)
    image.setPixmap(pixmap)


dirigeur = ImageProcessor()
workdir = ''
app = QApplication([])
main_window = QWidget()
main_window.setFixedSize(1080, 600)
find_image = QPushButton('Open image')
save_image = QPushButton('Save image')
play_button = QPushButton('Play generation')
stop_button = QPushButton('Stop generation')
original_image = QLabel()
original_image.setFixedSize(512, 512)
show_noimage('images/NOIMAGE.png', original_image)

edited_image = QLabel('Here will be edited image')
edited_image.setFixedSize(512, 512)
show_noimage('images/NOIMAGE.png', edited_image)


main_layout = QVBoxLayout()

sub_layout1 = QHBoxLayout()
sub_layout1_1 = QVBoxLayout()
sub_layout1_2 = QVBoxLayout()

sub_layout2 = QHBoxLayout()
sub_layout2_1 = QVBoxLayout()
sub_layout2_2 =QVBoxLayout()

sub_layout3 = QHBoxLayout()
sub_layout3_1 = QVBoxLayout()
sub_layout3_2 = QVBoxLayout()




main_window.setLayout(main_layout)
main_layout.addLayout(sub_layout1)
main_layout.addLayout(sub_layout2)
main_layout.addLayout(sub_layout3)

sub_layout1.addLayout(sub_layout1_1)
sub_layout1.addLayout(sub_layout1_2)
sub_layout2.addLayout(sub_layout2_1)
sub_layout2.addLayout(sub_layout2_2)
sub_layout3.addLayout(sub_layout3_1)
sub_layout3.addLayout(sub_layout3_2)

sub_layout1_1.addWidget(play_button)
sub_layout1_2.addWidget(stop_button)

sub_layout2_1.addWidget(original_image)
sub_layout2_2.addWidget(edited_image)

sub_layout3_1.addWidget(find_image)
sub_layout3_2.addWidget(save_image)






def open_image():
    path, _ = QFileDialog.getOpenFileName()
    if not path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        popup = QMessageBox()
        popup.setWindowTitle('Error')
        popup.setText('Selected file must be an image')
        popup.exec()
        return
    dirigeur.open_image(path)







find_image.clicked.connect(open_image)











































main_window.show()
app.exec()
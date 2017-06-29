import sys
import os
import numpy as np
import requests
import urllib.request
import PIL.Image
import json

from io import StringIO,BytesIO
from constants import *

class ConvWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(ConvWindow, self).__init__(parent)
        self.setGeometry(50, 50, 900, 500)
        self.setWindowTitle("DETAILS")
        self.setWindowIcon(QtGui.QIcon('dp2.png'))
        self.setStyleSheet("background-color: black")
        self.image_name = None
        self.part_counter = -1
        self.class_string = "CLASS : ???"
        self.home()

    def home(self):
        gen_button = QtGui.QPushButton("Generate Image", self)
        gen_button.clicked.connect(self.generate_image)
        gen_button.resize(200, 40)
        gen_button.move(140,455)
        gen_button.setStyleSheet('QPushButton {background-color: gray; color: white;}')
        gen_button.setFont(font)

        next_button = QtGui.QPushButton("Next", self)
        next_button.clicked.connect(self.next_image)
        next_button.resize(200, 40)
        next_button.move(570,355)
        next_button.setStyleSheet('QPushButton {background-color: gray; color: white;}')
        next_button.setFont(font)

        pred_button = QtGui.QPushButton("Predict", self)
        pred_button.clicked.connect(self.predict)
        pred_button.resize(200, 40)
        pred_button.move(570,400)
        pred_button.setStyleSheet('QPushButton {background-color: gray; color: white;}')
        pred_button.setFont(font)

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        qp.setPen(QColor(Qt.white))

        if self.image_name is None:
            #BIG IMAGE
            qp.drawRect(50, 50, 200, 200)
            qp.drawText(140, 140, "IMG1")
            qp.drawRect(250, 50, 200, 200)
            qp.drawText(340, 140, "IMG2")
            qp.drawRect(50, 250, 200, 200)
            qp.drawText(140, 340, "IMG3")
            qp.drawRect(250, 250, 200, 200)
            qp.drawText(340, 340, "IMG4")
        else:
            myPixmap = QtGui.QPixmap(self.image_name)
            myScaledPixmap = myPixmap.scaled(400, 400)
            qp.drawPixmap(50,50,myScaledPixmap)

        #PREDICT LABEL PART
        qp.drawRect(590, 40, 150, 50)
        qp.drawText(630, 60, self.class_string)
        if self.part_counter == -1:
            #IMAGE PARTS
            qp.drawRect(550, 100, 250, 250)
            qp.drawText(630, 220, "CONVOLUTION PART")
        else:
            myPixmap = QtGui.QPixmap('parts/' + str(self.part_counter) + '-part.png')
            myScaledPixmap = myPixmap.scaled(250, 250)
            qp.drawPixmap(550,100,myScaledPixmap)
        qp.end()

    def generate_image(self):
        r = requests.post(URL_MERGED)
        #print(r.json()[0])
        for url in r.json():
            print(url[34:])
            if "merged" in url:
                html = BytesIO(urllib.request.urlopen(url).read())
                newImage = PIL.Image.open(html)
                newImage.save(url[34:],"png")
            else:
                html = BytesIO(urllib.request.urlopen(url).read())
                newImage = PIL.Image.open(html)
                newImage.save(url[34:],"png")
        self.image_name = "merged.png"
        self.update()
        print("generate image")

    def next_image(self):
        self.part_counter += 1
        self.part_counter %= 12
        self.update()

    def predict(self):
        files = {'file':open("parts/" + str(self.part_counter) + "-part.png", 'rb')}
        r = requests.post(URL_PREDICT, files=files)

        label = r.json().get('label')
        print(label)

        if label == "3":
            label="NERVEOUS"
        elif label == "2":
            label="MUSCLE"
        elif label == "1":
            label="EPITHELIAL"
        elif label == "0":
            label="CONNECTIVE"

        self.class_string = "CLASS :" + label
        self.update()
        print("predict image")

class DetailsWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(DetailsWindow, self).__init__(parent)
        self.setGeometry(50, 50, 900, 500)
        self.setWindowTitle("DETAILS")
        self.setWindowIcon(QtGui.QIcon('dp2.png'))
        self.setStyleSheet("background-color: white")

    def paintEvent(self, event):
       qp = QPainter()
       qp.begin(self)
       qp.setPen(QColor(Qt.black))

       f_map = [None] * 1000
       x = 10
       y = 20
       for index, img_dir in enumerate(img_dir_list):

           qp.setPen(QColor(Qt.black))
           qp.drawText(x, y-5, str(index + 1) + ". CONVOLUTION FEATUREMAPS")
           qp.setPen(QColor(Qt.white))

           for index, f in enumerate(os.listdir(img_dir)):
               f_map[index] = QtGui.QPixmap(os.path.join(img_dir, f))
               f_map[index] = f_map[index].scaled(100, 100)
               if x < 880:
                   qp.drawPixmap(x, y, f_map[index])
                   x+=110
               else:
                   x = 10

                   y += 120
           x = 10
           y += 120
       #qp.drawRect(300,70,300,300)
       qp.end()

class ModelCreationWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(ModelCreationWindow, self).__init__(parent)
        self.setGeometry(50, 50, 900, 500)
        self.setWindowTitle("Model Creator")
        self.setWindowIcon(QtGui.QIcon('dp2.png'))
        self.home()

    def home(self):
        train_button = QtGui.QPushButton("Start Training", self)
        train_button.clicked.connect(self.train_created_model)
        train_button.resize(200, 50)
        train_button.move(350,400)
        train_button.setStyleSheet('QPushButton {background-color: gray; color: white;}')
        train_button.setFont(font)

        '''IMAGE INFO LABEL AND TEXT FIELDS'''
        self.width_label = QtGui.QLabel("WIDTH : ",self)
        self.width_label.move(80,60)
        self.width_label.setFont(font_2)
        self.width_label.resize(40,20)
        self.width_label.setStyleSheet("color: white;")

        self.w_textEdit = QtGui.QTextEdit("Enter Width", self)
        self.w_textEdit.move(140,60)
        self.w_textEdit.resize(80,25)
        self.w_textEdit.setStyleSheet("background-color: white")

        self.height_label = QtGui.QLabel("HEIGHT : ",self)
        self.height_label.move(80,100)
        self.height_label.setFont(font_2)
        self.height_label.resize(40,20)
        self.height_label.setStyleSheet("color: white;")

        self.h_textEdit = QtGui.QTextEdit("Enter Height", self)
        self.h_textEdit.move(140,100)
        self.h_textEdit.resize(80,25)
        self.h_textEdit.setStyleSheet("background-color: white")

        self.channel_label = QtGui.QLabel("CHANNEL : ",self)
        self.channel_label.move(80,140)
        self.channel_label.setFont(font_2)
        self.channel_label.resize(50,20)
        self.channel_label.setStyleSheet("color: white;")

        self.c_textEdit = QtGui.QTextEdit("Enter Channel", self)
        self.c_textEdit.move(140,140)
        self.c_textEdit.resize(80,25)
        self.c_textEdit.setStyleSheet("background-color: white")

        self.class_label = QtGui.QLabel("CLASS : ",self)
        self.class_label.move(80,180)
        self.class_label.setFont(font_2)
        self.class_label.resize(50,20)
        self.class_label.setStyleSheet("color: white;")

        self.class_textEdit = QtGui.QTextEdit("Enter Class", self)
        self.class_textEdit.move(140,180)
        self.class_textEdit.resize(80,25)
        self.class_textEdit.setStyleSheet("background-color: white")

        '''HYPERPARAMETERS LABEL AND TEXT FIELDS-LEFT'''
        self.lr_label = QtGui.QLabel("LR : ",self)
        self.lr_label.move(330,60)
        self.lr_label.setFont(font_2)
        self.lr_label.resize(50,20)
        self.lr_label.setStyleSheet("color: white;")

        self.lr_textEdit = QtGui.QTextEdit("Enter LR", self)
        self.lr_textEdit.move(390,60)
        self.lr_textEdit.resize(80,25)
        self.lr_textEdit.setStyleSheet("background-color: white")

        self.ep_label = QtGui.QLabel("Epoch : ",self)
        self.ep_label.move(330,100)
        self.ep_label.setFont(font_2)
        self.ep_label.resize(50,20)
        self.ep_label.setStyleSheet("color: white;")

        self.ep_textEdit = QtGui.QTextEdit("Enter Epoch", self)
        self.ep_textEdit.move(390,100)
        self.ep_textEdit.resize(80,25)
        self.ep_textEdit.setStyleSheet("background-color: white")

        self.batch_label = QtGui.QLabel("Batch: ",self)
        self.batch_label.move(330,140)
        self.batch_label.setFont(font_2)
        self.batch_label.resize(50,20)
        self.batch_label.setStyleSheet("color: white;")

        self.batch_textEdit = QtGui.QTextEdit("Enter Batch", self)
        self.batch_textEdit.move(390,140)
        self.batch_textEdit.resize(80,25)
        self.batch_textEdit.setStyleSheet("background-color: white")

        self.c1_label = QtGui.QLabel("Conv1K: ",self)
        self.c1_label.move(330,180)
        self.c1_label.setFont(font_2)
        self.c1_label.resize(50,20)
        self.c1_label.setStyleSheet("color: white;")

        self.c1_textEdit = QtGui.QTextEdit("Enter Conv1K", self)
        self.c1_textEdit.move(390,180)
        self.c1_textEdit.resize(80,25)
        self.c1_textEdit.setStyleSheet("background-color: white")

        self.c2_label = QtGui.QLabel("Conv2K: ",self)
        self.c2_label.move(330,220)
        self.c2_label.setFont(font_2)
        self.c2_label.resize(50,20)
        self.c2_label.setStyleSheet("color: white;")

        self.c2_textEdit = QtGui.QTextEdit("Enter Conv2K", self)
        self.c2_textEdit.move(390,220)
        self.c2_textEdit.resize(80,25)
        self.c2_textEdit.setStyleSheet("background-color: white")

        self.c3_label = QtGui.QLabel("Conv3K: ",self)
        self.c3_label.move(330,260)
        self.c3_label.setFont(font_2)
        self.c3_label.resize(50,20)
        self.c3_label.setStyleSheet("color: white;")

        self.c3_textEdit = QtGui.QTextEdit("Enter Conv3K", self)
        self.c3_textEdit.move(390,260)
        self.c3_textEdit.resize(80,25)
        self.c3_textEdit.setStyleSheet("background-color: white")

        self.c4_label = QtGui.QLabel("Conv4K: ",self)
        self.c4_label.move(330,300)
        self.c4_label.setFont(font_2)
        self.c4_label.resize(50,20)
        self.c4_label.setStyleSheet("color: white;")

        self.c4_textEdit = QtGui.QTextEdit("Enter Conv4K", self)
        self.c4_textEdit.move(390,300)
        self.c4_textEdit.resize(80,25)
        self.c4_textEdit.setStyleSheet("background-color: white")

        '''HYPERPARAMETERS LABEL AND TEXT FIELDS-RIGHT'''
        self.c5_label = QtGui.QLabel("Conv5K : ",self)
        self.c5_label.move(510,60)
        self.c5_label.setFont(font_2)
        self.c5_label.resize(50,20)
        self.c5_label.setStyleSheet("color: white;")

        self.c5_textEdit = QtGui.QTextEdit("Enter Conv5K", self)
        self.c5_textEdit.move(570,60)
        self.c5_textEdit.resize(80,25)
        self.c5_textEdit.setStyleSheet("background-color: white")

        self.c1o_label = QtGui.QLabel("Conv1-O : ",self)
        self.c1o_label.move(510,100)
        self.c1o_label.setFont(font_2)
        self.c1o_label.resize(50,20)
        self.c1o_label.setStyleSheet("color: white;")

        self.c1o_textEdit = QtGui.QTextEdit("Enter Conv1O", self)
        self.c1o_textEdit.move(570,100)
        self.c1o_textEdit.resize(80,25)
        self.c1o_textEdit.setStyleSheet("background-color: white")

        self.c2o_label = QtGui.QLabel("Conv2-O : ",self)
        self.c2o_label.move(510,140)
        self.c2o_label.setFont(font_2)
        self.c2o_label.resize(50,20)
        self.c2o_label.setStyleSheet("color: white;")

        self.c2o_textEdit = QtGui.QTextEdit("Enter Conv2O", self)
        self.c2o_textEdit.move(570,140)
        self.c2o_textEdit.resize(80,25)
        self.c2o_textEdit.setStyleSheet("background-color: white")

        self.c3o_label = QtGui.QLabel("Conv3-O : ",self)
        self.c3o_label.move(510,180)
        self.c3o_label.setFont(font_2)
        self.c3o_label.resize(50,20)
        self.c3o_label.setStyleSheet("color: white;")

        self.c3o_textEdit = QtGui.QTextEdit("Enter Conv3O", self)
        self.c3o_textEdit.move(570,180)
        self.c3o_textEdit.resize(80,25)
        self.c3o_textEdit.setStyleSheet("background-color: white")

        self.c4o_label = QtGui.QLabel("Conv4-O : ",self)
        self.c4o_label.move(510,220)
        self.c4o_label.setFont(font_2)
        self.c4o_label.resize(50,20)
        self.c4o_label.setStyleSheet("color: white;")

        self.c4o_textEdit = QtGui.QTextEdit("Enter Conv4O", self)
        self.c4o_textEdit.move(570,220)
        self.c4o_textEdit.resize(80,25)
        self.c4o_textEdit.setStyleSheet("background-color: white")

        self.c5o_label = QtGui.QLabel("Conv5-O : ",self)
        self.c5o_label.move(510,260)
        self.c5o_label.setFont(font_2)
        self.c5o_label.resize(50,20)
        self.c5o_label.setStyleSheet("color: white;")

        self.c5o_textEdit = QtGui.QTextEdit("Enter Conv5O", self)
        self.c5o_textEdit.move(570,260)
        self.c5o_textEdit.resize(80,25)
        self.c5o_textEdit.setStyleSheet("background-color: white")

        self.hl_label = QtGui.QLabel("HL : ",self)
        self.hl_label.move(510,300)
        self.hl_label.setFont(font_2)
        self.hl_label.resize(50,20)
        self.hl_label.setStyleSheet("color: white;")

        self.hl_textEdit = QtGui.QTextEdit("Enter HL", self)
        self.hl_textEdit.move(570,300)
        self.hl_textEdit.resize(80,25)
        self.hl_textEdit.setStyleSheet("background-color: white")

    def paintEvent(self, event):
       qp = QPainter()
       qp.begin(self)
       qp.setPen(QColor(Qt.white))

       qp.drawText(120, 45, "IMAGE INFO")
       qp.drawRect(50, 50, 200, 300)

       qp.drawText(450, 45, "HYPER PARAMETERS")
       qp.drawRect(300, 50, 400, 300)

       qp.drawText(780, 45, "EXTRAS")
       qp.drawRect(750, 50, 100, 300)

       qp.end()

    def train_created_model(self):
        width = self.w_textEdit.toPlainText()
        height = self.h_textEdit.toPlainText()
        channel = self.c_textEdit.toPlainText()

        c1 = self.c1_textEdit.toPlainText()
        c2 = self.c2_textEdit.toPlainText()
        c3 = self.c3_textEdit.toPlainText()
        c4 = self.c4_textEdit.toPlainText()
        c5 = self.c5_textEdit.toPlainText()

        c1o = self.c1o_textEdit.toPlainText()
        c2o = self.c2o_textEdit.toPlainText()
        c3o = self.c3o_textEdit.toPlainText()
        c4o = self.c4o_textEdit.toPlainText()
        c5o = self.c5o_textEdit.toPlainText()

        hl = self.hl_textEdit.toPlainText()
        batch = self.batch_textEdit.toPlainText()
        epoch = self.ep_textEdit.toPlainText()
        lr = self.lr_textEdit.toPlainText()
        cl = self.class_textEdit.toPlainText()

        params = {'width':width, 'height':height, 'channel':channel, 'hl':hl, 'batch':batch, 'lr':lr, 'class':cl,
                  'epoch':epoch, 'conv1':c1, 'conv2':c2, 'conv3':c3, 'conv4':c4, 'conv5':c5,
                  'conv1O':c1o, 'conv2O':c2o, 'conv3O':c3o, 'conv4O':c4o, 'conv5O':c5o}

        r = requests.post(URL_TRAINING, params=params)
        label = r.json().get('accuracy')
        print(label)

class PredictWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(PredictWindow, self).__init__(parent)

        self.image_name = None

        self.setGeometry(50, 50, 900, 500)
        self.setWindowTitle("Prediction")
        self.setWindowIcon(QtGui.QIcon('dp2.png'))
        self.setStyleSheet("background-color: black")

        self.details_window = DetailsWindow(self)
        self.statusBar()
        self.mainMenu = self.menuBar()
        self.home()


    def home(self):
        self.image_label = QtGui.QLabel("",self)
        self.image_label.move(100,20)
        self.image_label.setFont(font_2)
        self.image_label.resize(800,20)
        self.image_label.setStyleSheet("color: white;")

        self.result_label = QtGui.QLabel(" RESULTS",self)
        self.result_label.move(280,380)
        self.result_label.resize(800,20)
        self.result_label.setFont(font_2)
        self.result_label.setStyleSheet("color: white;")

        predict_btn = QtGui.QPushButton("PREDICT", self)
        predict_btn.clicked.connect(self.predict)
        predict_btn.resize(300, 100)
        predict_btn.move(400,400)
        predict_btn.setStyleSheet('QPushButton {background-color: black; color: white;}')
        predict_btn.setFont(font)

        details_btn = QtGui.QPushButton("DETAILS", self)
        details_btn.clicked.connect(self.show_details)
        details_btn.resize(300, 100)
        details_btn.move(200,400)
        details_btn.setStyleSheet('QPushButton {background-color: black; color: white;}')
        details_btn.setFont(font)

        openFile = QtGui.QAction("&Open File", self)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip('Open File')
        openFile.triggered.connect(self.file_open)

        self.fileMenu = self.mainMenu.addMenu('&File')
        self.fileMenu.addAction(openFile)

    def paintEvent(self, event):
       qp = QPainter()
       qp.begin(self)
       qp.setPen(QColor(Qt.white))

       if self.image_name is not None:
           myPixmap = QtGui.QPixmap(self.image_name)
           myScaledPixmap = myPixmap.scaled(450, 300)
           qp.drawPixmap(290,70,myScaledPixmap)
       else:
           qp.drawText(310,250, "None Image")
           qp.drawRect(300,70,300,300)
       qp.end()

    def show_details(self):
        files = {'file':open(self.image_name, 'rb')}
        r = requests.post(URL_FEATURE_MAP, files=files)
        #print(r.json()[0])
        for url in r.json():
            #print(url[34:])
            html = BytesIO(urllib.request.urlopen(url).read())
            newImage = PIL.Image.open(html)
            newImage.save('featuremaps/' + url[34:],"png")
        self.details_window.show()

    def predict(self):
        print("Predict")
        files = {'file':open(self.image_name, 'rb')}
        r = requests.post(URL_PREDICT, files=files)

        label = r.json().get('label')
        print(label)

        if label == "3":
            label="NERVEOUS"
        elif label == "2":
            label="MUSCLE"
        elif label == "1":
            label="EPITHELIAL"
        elif label == "0":
            label="CONNECTIVE"

        self.result_label.setText(" RESULTS -> " + label)
        #self.result_label.setText(" RESULTS -> [CLASS : NERVEOUS CONFIDENCE : %97.6]")

    def file_open(self):
        self.image_name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        self.image_label.setText("IMAGE : " + str(self.image_name))
        self.update()

class Window(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)
        self.setGeometry(50, 50, 900, 500)
        self.setWindowTitle("Tissue Deep Learning Tool")
        self.setWindowIcon(QtGui.QIcon('dp.png'))
        self.setStyleSheet("background-color: black")
        self.create_model_win = ModelCreationWindow(self)
        self.predict_model_win = PredictWindow(self)
        self.conv_win = ConvWindow(self)
        self.home()

    def home(self):

        self.menu_label = QtGui.QLabel("MAIN MENU",self)
        self.menu_label.move(400,20)
        self.menu_label.setFont(font_1)
        self.menu_label.resize(200,20)
        self.menu_label.setStyleSheet("color: white;")

        pred_menu_button = QtGui.QPushButton("Predict With Trained", self)
        pred_menu_button.clicked.connect(self.predict)
        pred_menu_button.resize(400, 200)
        pred_menu_button.move(50,50)
        pred_menu_button.setStyleSheet('QPushButton {background-color: gray; color: white;}')
        pred_menu_button.setFont(font_3)

        model_menu_button = QtGui.QPushButton("Create Model", self)
        model_menu_button.clicked.connect(self.create_model)
        model_menu_button.resize(400,200)
        model_menu_button.move(50,250)
        model_menu_button.setStyleSheet('QPushButton {background-color: gray; color: white;}')
        model_menu_button.setFont(font_3)

        conv_menu_button = QtGui.QPushButton("Big Convolution Test", self)
        conv_menu_button.clicked.connect(self.convolutional_test)
        conv_menu_button.resize(400,400)
        conv_menu_button.move(450,50)
        conv_menu_button.setStyleSheet('QPushButton {background-color: gray; color: white;}')
        conv_menu_button.setFont(font_3)

        self.show()

    def predict(self):
        self.predict_model_win.show()
        print("predicted")
        #sys.exit()

    def create_model(self):
        self.create_model_win.show()
        print("new model created")
        #sys.exit()
    def convolutional_test(self):
        self.conv_win.show()
def run():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    #ex = sWidget()
    sys.exit(app.exec_())


run()

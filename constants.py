
from PyQt4 import QtGui, QtCore

from PyQt4.QtGui import *
from PyQt4.QtCore import *

img_dir_list = ["C:/Users/recep/Desktop/MyFiles/PythonCodes/G_GUI/featuremaps/c1",
                "C:/Users/recep/Desktop/MyFiles/PythonCodes/G_GUI/featuremaps/c2",
                "C:/Users/recep/Desktop/MyFiles/PythonCodes/G_GUI/featuremaps/c3",
                "C:/Users/recep/Desktop/MyFiles/PythonCodes/G_GUI/featuremaps/c4",
                "C:/Users/recep/Desktop/MyFiles/PythonCodes/G_GUI/featuremaps/c5",]

URL_PREDICT = "http://208.101.10.219:5000/predict"
URL_FEATURE_MAP = "http://208.101.10.219:5000/features"
URL_TRAINING = "http://208.101.10.219:5000/train"
URL_MERGED = "http://208.101.10.219:5000/merged"

font = QtGui.QFont("ComicSans", 15, QtGui.QFont.Bold)
font.setBold(True)
font.setItalic(False)

font_1 = QtGui.QFont("ComicSans", 12, QtGui.QFont.Bold)
font_1.setBold(True)
font_1.setItalic(False)

font_2 = QtGui.QFont("ComicSans", 8, QtGui.QFont.Bold)
font_2.setBold(True)
font_2.setItalic(False)

font_3 = QtGui.QFont("ComicSans", 20, QtGui.QFont.Bold)
font_3.setBold(True)
font_3.setItalic(False)

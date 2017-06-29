import numpy as np
from scipy import misc,ndimage
import random

def oneHotEncodingWithLabel(label, classCount):
	oneHotArray = [0] * classCount
	oneHotArray[int(label) - 1] = 1
	return oneHotArray

def readFileContent(filename):
	imageNamesAndLabels = []
	
	with open(filename, 'r') as f:
		for content in f:
			imageNamesAndLabels.append(content)
	return imageNamesAndLabels

def SplitImage(image,size):
	splittedImages = [image[i:i+size,j:j+size,:] for i in range(0,np.array(image).shape[0]-size,120) for j in range(0,np.array(image).shape[1] - size,120)]

	return splittedImages	

def readIMGAndLabels(filename):
	data = []
	stringSet = readFileContent(filename)
	amount = len(stringSet)
	classCount = 4
	for i in range(amount):
		image = stringSet[i].split()[0]
		label = stringSet[i].split()[1]
		pixels = misc.imread("histologyDS2828/imgs/"+image)
		#s_images = SplitImage(pixels,300)
		#print(np.array(s_images[0]).shape)
		#print(len(s_images))
		#newSize = ndimage.interpolation.zoom(pixels,(0.1,0.1,1.0))
		label_oh = oneHotEncodingWithLabel(int(label),classCount)
		#for img in s_images:
			#print(img.shape)
		#	data.append([img,label_oh])
		data.append([pixels,label_oh])
	random.shuffle(data)
	return data

def CreateTrainTestSetsAndLabels(filename, percent = 0.01):

	data = readIMGAndLabels(filename)
	data = np.array(data)
	test_data_size = int(percent * len(data))

	train_x = list(data[:,0][:])
	train_y = list(data[:,1][:])
	
	test_x = list(data[:,0][-test_data_size:])
	test_y = list(data[:,1][-test_data_size:])

	return train_x, train_y, test_x, test_y


#CreateTrainTestSetsAndLabels("idxrenata2828.txt")


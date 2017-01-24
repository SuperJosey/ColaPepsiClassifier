from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets
from sklearn.cross_validation import train_test_split
import numpy as np
import cv2
import os
import pickle
import sys

def rgbHistogram(image):
	bins=(8,8,8)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
	cv2.normalize(hist, hist)

	return hist.flatten()


#Get the path
if len(sys.argv) > 1 :
	path = str(sys.argv[1]) #Path to image to process
else:
	path = 'not defined'

#Extract the feature of the image we want to predict's class:
features = []
img = cv2.imread(path)
hist = rgbHistogram(img)
features.append(hist)
features = np.array(features)

#Load the model:
filename = "bestKnn_model.pkl"
loadedModel = pickle.load(open(filename, 'rb'))

#Prediction:
print loadedModel.predict(features)

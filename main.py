from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets
from sklearn.cross_validation import train_test_split
import numpy as np
import argparse
import cv2
import os
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import pickle

def resizeFlattenPixel(image):
	size=(32, 32)
	return cv2.resize(image, size).flatten()

def rgbHistogram(image):
	bins=(8,8,8)
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
	cv2.normalize(hist, hist)

	return hist.flatten()

def listFileFromPath():
	tmpImgPath = list()
	for filename in os.listdir('/Users/clement/Desktop/ColaPepsi/dataset_hard/'):
		if ".jpg" in filename :
			#print "returning: "+filename #Debugging
			tmpImgPath.append(filename)
	return tmpImgPath

def plot_zone(X, y, k):
	h = .01 #pas du mesh

	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

	for weights in ['uniform', 'distance']:
		clf = neighbors.KNeighborsClassifier(k, weights=weights)
		clf.fit(X, y)

		x_min, x_max = X[:,0].min() - 1, X[:, 0].max() + 1
		y_min, y_max = X[:,1].min() - 1, X[:, 1].max() + 1

		xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

		Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
		Z = Z.reshape(xx.shape)
		plt.figure()
		plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

		#training points:
		plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
		plt.xlim(xx.min(), xx.max())
		plt.ylim(yy.min(), yy.max())
		plt.title("Classification diagram for the K-nn")

	plt.show()

rawImages = [] #Flatten image features
features = [] #Histogram features
labels = [] #String labels
labelsGr = [] #Integer labels used for ploting	

imagePath = list()
imagePath = listFileFromPath()

#print imagePath #debugging

for (i, image) in enumerate(imagePath):

	img = cv2.imread("/Users/clement/Desktop/ColaPepsi/dataset_hard/"+image)
	classe = image.split(os.path.sep)[-1].split("-")[0]
	#Load a class vector for the ploting
	if "cola" in classe :
		labelsGr.append(1)
	elif "pepsi" in classe :
		labelsGr.append(2)
	#print "Image: "+image+"Classe: "+classe

	#Features extraction for both methods:
	pixels = resizeFlattenPixel(img)
	hist = rgbHistogram(img)
	#print hist

	rawImages.append(pixels)
	features.append(hist)
	labels.append(classe)
	if i > 0 and i % 100 == 0 :
		print "Processing..."
#endfor

#Converty array to numpy matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
labelsGr = np.array(labelsGr)

#Debug
#print rawImages.shape
#print features.shape
#print labels.shape
#print labelsGr

#Split the dataset into datatraining & datatest:
(trainRawI, testRawI, trainLabels, testLabels) = train_test_split(rawImages, labels, test_size=0.25, random_state=4)
(trainFeatures, testFeatures, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=4)

#Entrainement et test du K-NN:
# print("=====Pixel=====")
# model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
# model.fit(trainRawI, trainLabels)
# acc = model.score(testRawI, testLabels)
# print("Taux de classification: {:.2f}%".format(acc * 100))

print "==================Histogram Method===================="
print "Empiric method for the k-NN hyper-param:"

k = 1
tmpAcc = 1
acc = 0
cont = True

while (cont == True and k <10) :
	model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
 	model.fit(trainFeatures, trainLabels)
 	acc = model.score(testFeatures, testLabels)
 	print"Taux de classification correcte: ",
 	print acc*100,
 	print "%"
 	print "Score for k="+ str(k)
  	if tmpAcc >= acc :
  		acc = tmpAcc
  		cont = True
  	else:
  		cont = False
  	k = k+1

#Print prediction
#print model.predict(testFeatures)
#print testLabels

#Plot decisions zone :  
#X = features[:, :2]
#y = labelsGr
#k = 1
#plot_zone(X, y, k)

#Best param given from empiric method is 8 with a limit of 10 neighbours to avoid overfitting
#Save the model giving the best classification accuracy:
modelToSave = KNeighborsClassifier(n_neighbors=8, n_jobs=-1)
modelToSave.fit(trainFeatures, trainLabels)
externalModel = "bestKnn_model.pkl"
pickle.dump(model, open(externalModel, "wb"))





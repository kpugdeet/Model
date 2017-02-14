##################################################################
# Date    : 2016-11-15											
# Author  : Krittaphat Pugdeethosapol (krittaphat.pug@gmail.com)
# Version : 1.0						
##################################################################

import numpy as np
import math
import time
import random
import pickle
import ConfigParser
from numba import vectorize, cuda

class RBMK:
	def __init__(self, configFile, typeRBM):
		# Reading Config file
		Config = ConfigParser.ConfigParser()
		Config.read(configFile)
		self.numHidden = int(Config.get(typeRBM, 'numHidden'))
		self.numVisible = int(Config.get(typeRBM, 'numVisible'))
		self.startLearningRate = float(Config.get(typeRBM, 'learningRate'))
		self.maxEpochs = int(Config.get(typeRBM, 'maxEpochs'))
		self.batchSize = int(Config.get(typeRBM, 'batchSize'))
		self.weightsObject = Config.get(typeRBM, 'weightsObject')
		self.hBiasObject = Config.get(typeRBM, 'hBiasObject')
		self.vBiasObject = Config.get(typeRBM, 'vBiasObject')
		self.screenObject = Config.get(typeRBM, 'screenObject')
		self.k = int(Config.get(typeRBM, 'k'))
		self.numpyRng = np.random.RandomState(random.randrange(0, 100))

		# Initial with zero mean and 0.01 std
		try:
			self.weights = pickle.load(open(self.weightsObject, 'rb' ))
		except:
			self.weights = np.asarray(self.numpyRng.normal(0, 0.01, size=(self.k, self.numVisible, self.numHidden)), dtype=np.float32)

		# Inital hidden Bias
		try:
			self.hBias = pickle.load(open(self.hBiasObject, 'rb' ))
		except:
			self.hBias = np.zeros(self.numHidden, dtype=np.float32)

		# Inital visible Bias
		self.vBias = np.zeros((self.k, self.numVisible), dtype=np.float32)

		# Initial Screen
		try:
			self.screen = pickle.load(open(self.screenObject, 'rb' ))
		except:
			self.screen = [1] * self.numVisible

		self.exclude = None

	# Sigmoid
	def sigmoid (self, x):
		return 1.0/(1+np.exp(-x))

	# Calculate and return Positive hidden states and probabilities
	def positiveProb (self, visible):
		for i in range(self.k):
			if i == 0:
				posHiddenActivations = np.dot(visible[i], self.weights[i])
			else:
				posHiddenActivations += np.dot(visible[i], self.weights[i])
		posHiddenActivations += self.hBias

		posHiddenProbs = self.sigmoid(posHiddenActivations)
		posHiddenStates = posHiddenProbs > np.random.rand(len(visible[0]), self.numHidden)

		return [posHiddenStates, posHiddenProbs]

	# Calculate and return Negative hidden states and probs
	def negativeProb (self, hidden, exclude, step = 1):
		for i in range (step):
			visActivations = [[] for x in range(self.k)]
			for i in range(self.k):
				visActivations[i] = np.dot(hidden, self.weights[i].T)+self.vBias[i]

			visProbs = [[] for x in range(self.k)]
			# for i in range(self.k):
			# 	visProbs[i] = self.sigmoid(visActivations[i])
			# 	visProbs[i] = visProbs[i] * self.screen

			for i in range(self.k):
				visProbs[i] = np.exp(visActivations[i])

			sumvisibleProbs = np.array(visProbs[0])
			for i in range(1, self.k):
				sumvisibleProbs += visProbs[i]

			sumvisibleProbs[sumvisibleProbs == 0] = 1
			visProbs /= sumvisibleProbs
			for i in range(self.k):
				visProbs[i] = visProbs[i] * self.screen

			# Missing Value
			visProbs = visProbs * exclude

			# Get back to calculate hidden again
			hidden, hiddenProbs = self.positiveProb(visProbs)

		return [visProbs, hiddenProbs]

	# Get hidden state
	def getHidden (self, visible):
		for i in range(self.k):
			if i == 0:
				hiddenActivations = np.dot(visible[i], self.weights[i])
			else:
				hiddenActivations += np.dot(visible[i], self.weights[i])
		hiddenActivations += self.hBias

		hiddenProbs = self.sigmoid(hiddenActivations)
		hiddenStates = hiddenProbs > np.random.rand(len(visible[0]), self.numHidden)

		np.set_printoptions(threshold=np.nan)

		return hiddenStates

	# Get visivle state
	def getVisible (self, hidden):
		visibleActivations = [[] for x in range(self.k)]
		for i in range(self.k):
			visibleActivations[i] = np.dot(hidden, self.weights[i].T)+self.vBias[i]

		visibleProbs = [[] for x in range(self.k)]
		# for i in range(self.k):
		# 	visibleProbs[i] = self.sigmoid(visibleActivations[i])
		# 	visibleProbs[i] = visibleProbs[i] * self.screen

		for i in range(self.k):
			visibleProbs[i] = np.exp(visibleActivations[i])

		sumvisibleProbs = np.array(visibleProbs[0])
		for i in range(1, self.k):
			sumvisibleProbs += visibleProbs[i]

		sumvisibleProbs[sumvisibleProbs == 0] = 1
		visibleProbs /= sumvisibleProbs
		for i in range(self.k):
			visibleProbs[i] = visibleProbs[i] * self.screen

		return visibleProbs

	# Train RMB model
	def train (self, data):
		# Screen some visible that always 0
		self.screen = [1] * self.numVisible
		for column in range(data.shape[2]):
			tmpBias = 0
			for row in data:
				tmpBias += sum(row2[column] for row2 in row)
			if (tmpBias < 1):
				self.screen[column] = 0
		data = data * self.screen

		# Exclude missing value
		self.exclude = [[1 for i in range(self.numVisible) ] for j in range(data.shape[1])]
		for us in range(data.shape[1]):
			for column in range(data.shape[2]):
				self.exclude[us][column] = sum(iK[column] for iK in data[:,us])

		# Clear the weight of some visibile that never appear and Add vBias
		self.weights = np.asarray(self.numpyRng.normal(0, 0.01, size=(self.k, self.numVisible, self.numHidden)), dtype=np.float32)
		for i in range(self.k):
			self.weights[i] = (self.weights[i].T * self.screen).T
		self.hBias = np.zeros(self.numHidden, dtype=np.float32)
		self.vBias = np.zeros((self.k, self.numVisible), dtype=np.float32)

		# Start at CD1
		step = 1
		learningRate = self.startLearningRate
		# Loop for how many iterations
		for epoch in range (self.maxEpochs):
			if (epoch != 0 and epoch%10 == 0):
				step += 2

			startTime = time.time()

			# Divide in to batch
			totalBatch = math.ceil(data.shape[1]/self.batchSize)
			if data.shape[1]%self.batchSize != 0:
				totalBatch += 1

			# Loop for each batch
			for batchIndex in range (int(totalBatch)):
				# Get the data for each batch
				tmpData = [[] for x in range(self.k)]
				for i in range(self.k):
					tmpData[i] = data[i][batchIndex*self.batchSize: (batchIndex+1)*self.batchSize]
				tmpExclude = self.exclude[batchIndex*self.batchSize: (batchIndex+1)*self.batchSize]
				tmpData = np.array(tmpData)
				numExamples = tmpData.shape[1]

				# Caculate positive probs and Expectation for Sigma(ViHj) data
				posHiddenStates, posHiddenProbs = self.positiveProb(tmpData)
				posAssociations = [[] for x in range(self.k)]
				posVisibleBias = [[] for x in range(self.k)]
				for i in range(self.k):
					posAssociations[i] = np.dot(tmpData[i].T, posHiddenProbs)
					posVisibleBias[i] = np.dot(tmpData[i].T, np.ones(tmpData.shape[1]).T)
				posHiddenBias = np.dot(np.ones(tmpData.shape[1]),posHiddenProbs)


				# Calculate negative probs and Expecatation for Sigma(ViHj) recon with k = step of gibs
				negVisibleProbs, negHiddenProbs = self.negativeProb(posHiddenStates, tmpExclude, step = step)
				negAssociations = [[] for x in range(self.k)]
				negVisibleBias = [[] for x in range(self.k)]
				for i in range(self.k):
					negAssociations[i] = np.dot(negVisibleProbs[i].T, negHiddenProbs)
					negVisibleBias[i] = np.dot(negVisibleProbs[i].T, np.ones(tmpData.shape[1]).T)
				negHiddenBias = np.dot(np.ones(tmpData.shape[1]),negHiddenProbs)

				# Update weight
				for i in range(self.k):
					self.weights[i] += learningRate*((posAssociations[i]-negAssociations[i])/numExamples)
					self.vBias[i] += learningRate*(((posVisibleBias[i]-negVisibleBias[i])*self.screen)/numExamples)
				self.hBias += learningRate*((posHiddenBias-negHiddenBias)/numExamples)

			# Check error for each epoch
			tmpHidden = self.getHidden(data)
			tmpVisible = self.getVisible(tmpHidden)

			rmseRrror = math.sqrt(np.sum((data - tmpVisible) ** 2) / (data.shape[0]*data.shape[1]*data.shape[2]))
			print ('RMSE : {0}'.format(rmseRrror))

			tmpVisible = tmpVisible * data
			rmseRrror = math.sqrt(np.sum((data-tmpVisible)**2)/np.sum(data == 1))
			totalTime = time.time()-startTime
			# print ('{0:7}Epoch : {1} Time : {2}'.format('INFO', epoch, totalTime))
			print ('{0:7}Epoch : {1} Train RMSE : {2} Time : {3}'.format('INFO', epoch, rmseRrror, totalTime))

		# Save weights
		pickle.dump(self.weights, open(self.weightsObject,'wb'))
		pickle.dump(self.hBias, open(self.hBiasObject,'wb'))
		pickle.dump(self.vBias, open(self.vBiasObject,'wb'))
		pickle.dump(self.screen, open(self.screenObject,'wb'))

if __name__ == '__main__':
	threshold = 3
	userRBMK = RBMK('../data/Config.ini', 'UserRBMK')

	countEX = 0
	# Read Data
	print('Read Data')
	filePointer = open('../data/MovieUserInfo.dat')
	iterLines = iter(filePointer)
	rateEx = []
	idEx = []
	dataID = []
	data = [[] for x in range(userRBMK.k)]
	dataTest = [[] for x in range(userRBMK.k)]
	for lineNum, line in enumerate(iterLines):
		tmp = [ [0 for i in range(userRBMK.numVisible) ] for j in range(userRBMK.k) ]
		ID = line.split('::')[0]
		line = line.split('::')[1:]
		exID = np.random.randint(len(line))
		for offset, ele in enumerate(line):
			idTmp = ele.split(',')[0]
			rate = ele.split(',')[1]
			if int(idTmp) < 10000:
				tmp[int(float(rate)-1)][int(idTmp)] = int(1)
				if offset == exID:
					if int(float(rate)-1) >= threshold:
						print('Ex {0} {1}'.format(lineNum,offset))
						rateEx.append(int(float(rate)-1))
						idEx.append(int(idTmp))
						tmp[int(float(rate)-1)][int(idTmp)] = int(0)
						if lineNum >= 500:
							countEX += 1
					else:
						exID += 1
		for i in range(userRBMK.k):
			if lineNum < 500:
				data[i].append(tmp[i])
			else:
				dataTest[i].append(tmp[i])
		dataID.append(ID)
	data = np.array(data)
	dataTest = np.array(dataTest)
	print (data.shape)
	print (dataTest.shape)
	print (countEX)

	# Train
	print('Training')
	a = time.time()
	userRBMK.train (data)
	print('Time = {0}'.format(time.time()-a))

	# Calculate all output
	print('Recall')
	tmpHidden = userRBMK.getHidden(dataTest)
	tmpVisible = userRBMK.getVisible(tmpHidden)
	tmpVisible = np.array(tmpVisible)

	############################################################
	# Change value to 1 and 0 evaluation
	# Calculate Output to select right pos for each visible unit
	print('Calculate output')
	output = [[None for i in range(tmpVisible.shape[2])] for j in range(tmpVisible.shape[1])]
	for userID in range(tmpVisible.shape[1]):
		for docID in range(tmpVisible.shape[2]):
			maxValue = tmpVisible[0][userID][docID]
			maxPos = 0
			for dim in range(1,userRBMK.k):
				if (tmpVisible[dim][userID][docID] > tmpVisible[dim-1][userID][docID]):
					maxValue = tmpVisible[dim][userID][docID]
					maxPos = dim
			output[userID][docID] = dict({'doc':docID, 'pos':maxPos, 'value':maxValue})
	totalError = 0
	sumValue = 0
	countCheck = 0
	countCheck1 = 0
	for eachUser in range(tmpVisible.shape[1]):
		check1 = 0
		check2 = 0
		if output[eachUser][idEx[eachUser]]['pos'] >= threshold:
			check1 = 1
			countCheck1 += 1
		if rateEx[eachUser] >= threshold:
			check2 = 1
			countCheck += 1
		if check1 != check2:
			totalError += 1
	print('Accuracy = {0}%'.format((1-(float(totalError)/float(dataTest.shape[1])))*100))
	tmpVisible = tmpVisible * dataTest
	rmseError = math.sqrt(np.sum((dataTest - tmpVisible) ** 2) / np.sum(dataTest == 1))
	print ('Testing RMSE : {0}'.format(rmseError))
	print(countCheck)
	print(countCheck1)

	# # MAP Evaluation
	# print('MAP Evaulation')
	# filePointer = open('../data/MovieKMeansCat20.dat')
	# iterLines = iter(filePointer)
	# catDict = {'key': 'value'}
	# for lineNum, line in enumerate(iterLines):
	# 	line = line.rstrip()
	# 	catDict[line.split('::')[0]] = line.split('::')[1]
    #
	# output = [[None for i in range(tmpVisible.shape[2])] for j in range(tmpVisible.shape[1])]
	# for userID in range(tmpVisible.shape[1]):
	# 	for docID in range(tmpVisible.shape[2]):
	# 		maxValue = tmpVisible[0][userID][docID]
	# 		maxPos = 0
	# 		for dim in range(1,userRBMK.k):
	# 			if (tmpVisible[dim][userID][docID] > tmpVisible[dim-1][userID][docID]):
	# 				maxValue = tmpVisible[dim][userID][docID]
	# 				maxPos = dim
	# 		output[userID][docID] = dict({'doc':docID, 'pos':maxPos, 'value':maxValue})
    #
	# outputArray = {'key':'value'}
	# output = np.array(output)
	# for i in range(output.shape[0]):
	# 	maxTop = 5
	# 	countTop = 0
	# 	tmpValue = ''
	# 	tmpList = sorted([x for x in output[i]], key=lambda k:k['value'], reverse=True)
	# 	for j in range(len(tmpList)):
	# 		if userRBMK.exclude[i][int(tmpList[j]['doc'])] == 0:
	# 			if countTop != 0:
	# 				tmpValue += '::'
	# 			tmpValue += str(tmpList[j]['doc'])
	# 			countTop = countTop + 1
	# 			if countTop == maxTop:
	# 				break
	# 	outputArray[dataID[i]] = tmpValue
	# print(outputArray)

	# tmpVisible = tmpVisible * dataTest
	# rmseError = math.sqrt(np.sum((dataTest - tmpVisible) ** 2) / np.sum(dataTest == 1))
	# print ('Testing RMSE : {0}'.format(rmseError))
    #

	# Calculate all output
	print('Recall')
	tmpHidden = userRBMK.getHidden(dataTest)
	tmpVisible = userRBMK.getVisible(tmpHidden)
	tmpVisible = np.array(tmpVisible)
	############################################################
	# Change value to 1 and 0 evaluation
	# Calculate Output to select right pos for each visible unit
	print('Calculate output')
	output = [[None for i in range(tmpVisible.shape[2])] for j in range(tmpVisible.shape[1])]
	for userID in range(tmpVisible.shape[1]):
		for docID in range(tmpVisible.shape[2]):
			maxValue = tmpVisible[0][userID][docID]
			maxPos = 0
			for dim in range(1,userRBMK.k):
				if (tmpVisible[dim][userID][docID] > tmpVisible[dim-1][userID][docID]):
					maxValue = tmpVisible[dim][userID][docID]
					maxPos = dim
			output[userID][docID] = dict({'doc':docID, 'pos':maxPos, 'value':maxValue})
	# Sort and make to correct structure
	print('Ranking with specific rating')
	f = open('../data/MovieUserInfoOut.dat','w')
	outputArray = {'key':'value'}
	output = np.array(output)
	for i in range(output.shape[0]):
		maxTop = 100
		countTop = 0
		tmpValue = ''
		# posPoint = 1
		# tmpList = sorted([x for x in output[i] if x['pos']==posPoint], key=lambda k:k['value'], reverse=True)
		tmpList = sorted([x for x in output[i] if int(x['pos']) >= threshold], key=lambda k:k['value'], reverse=True)
		print(output[i][idEx[i]]['pos'])
		for j in range(len(tmpList)):
			print ('{0} {1} {2}'.format(i, tmpList[j]['doc'], np.sum(dataTest[:, i, int(tmpList[j]['doc'])])))
			if np.sum(dataTest[:, i, int(tmpList[j]['doc'])]) == 0:
				if countTop != 0:
					tmpValue += '::'
				tmpValue += str(tmpList[j]['doc'])
				countTop = countTop + 1
				if countTop == maxTop:
					break
		outputArray[dataID[i]] = tmpValue
		f.write('{0}::{1}\n'.format(int(dataID[i])+500,tmpValue))

	# pickle.dump(outputArray, open('../data/tmpOutput.object','wb'))
	print('Done')







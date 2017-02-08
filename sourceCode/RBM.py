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

class RBM:
	def __init__(self, configFile, typeRBM):
		np.seterr(all='ignore')
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
		
		self.numpyRng = np.random.RandomState(random.randrange(0, 100))

		# Initial with zero mean and 0.01 std
		try:
			self.weights = pickle.load(open(self.weightsObject, 'rb' ))
		except:
			self.weights = np.asarray(self.numpyRng.normal(0, 0.01, size = (self.numVisible, self.numHidden)), dtype = np.float32)

		# Inital hidden Bias
		try:
			self.hBias = pickle.load(open(self.hBiasObject, 'rb' ))
		except:
			self.hBias = np.zeros(self.numHidden, dtype = np.float32)

		# Inital visible Bias
		try:
			self.vBias = pickle.load(open(self.vBiasObject, 'rb' ))
		except:
			self.vBias = np.zeros(self.numVisible, dtype = np.float32)

		# Initial Screen
		try:
			self.screen = pickle.load(open(self.screenObject, 'rb' ))
		except:
			self.screen = [1] * self.numVisible

	# Sigmoid
	def sigmoid (self, x):
		return 1.0 / (1 + np.exp(-x))

	# Calculate and return Positive hidden states and probabilities
	def positiveProb (self, visible):
		posHiddenActivations = np.dot(visible, self.weights) + self.hBias
		posHiddenProbs = self.sigmoid(posHiddenActivations)
		posHiddenStates = posHiddenProbs > np.random.rand(visible.shape[0], self.numHidden)
		return [posHiddenStates, posHiddenProbs]

	# Calculate and return Negative hidden states and probs
	def negativeProb (self, hidden, k = 1):
		for i in range (k):
			visActivations = np.dot(hidden, self.weights.T) + self.vBias
			visProbs = self.sigmoid(visActivations)
			visProbs = visProbs * self.screen
			hidden, hiddenProbs = self.positiveProb(visProbs)
		return [visProbs, hiddenProbs]

	# Get hidden state
	def getHidden (self, visible):
		hiddenActivations = np.dot(visible, self.weights) + self.hBias
		hiddenProbs = self.sigmoid(hiddenActivations)
		hiddenStates = hiddenProbs > np.random.rand(visible.shape[0], self.numHidden)
		return hiddenStates

	# Get visivle state
	def getVisible (self, hidden):
		visibleActivations = np.dot(hidden, self.weights.T) + self.vBias
		visibleProbs = self.sigmoid(visibleActivations)
		visibleProbs = visibleProbs * self.screen
		return visibleProbs

	# Train RMB model
	def train (self, data, test):
		# Screen some visible that always 0
		self.screen = [1] * self.numVisible
		for column in range(data.shape[1]):
			tmpBias = sum(row[column] for row in data)
			if (tmpBias < 10):
				self.screen[column] = 0
		data = data * self.screen

		# Clear the weight of some visibile that never appear and Add vBias
		self.weights = np.asarray(self.numpyRng.normal(0, 0.01, size=(self.numVisible, self.numHidden)), dtype=np.float32)
		self.weights = (self.weights.T * self.screen).T
		self.hBias = np.zeros(self.numHidden, dtype = np.float32)
		self.vBias = np.zeros(self.numVisible, dtype = np.float32)

		start = time.time()
		# Start at CD1
		step = 1
		learningRate = self.startLearningRate
		# Loop for how many iterations
		for epoch in range (self.maxEpochs): 
			if (epoch != 0 and epoch%20 == 0):
				step += 2

			startTime = time.time()

			# Divide in to batch
			totalBatch = math.ceil(data.shape[0]/self.batchSize)
			if data.shape[0]%self.batchSize != 0:
				totalBatch += 1

			# Loop for each batch
			for batchIndex in range (int(totalBatch)):
				# Get the data for each batch
				tmpData = data[batchIndex*self.batchSize: (batchIndex+1)*self.batchSize]
				numExamples = tmpData.shape[0]

				# Caculate positive probs and Expectation for Sigma(ViHj) data
				posHiddenStates, posHiddenProbs = self.positiveProb(tmpData)
				posAssociations = np.dot(tmpData.T, posHiddenProbs)
				posHiddenBias = np.dot(np.ones(tmpData.shape[0]),posHiddenProbs)
				posVisibleBias = np.dot(tmpData.T, np.ones(tmpData.shape[0]).T)

				# Calculate negative probs and Expecatation for Sigma(ViHj) recon with k = step of gibs
				negVisibleProbs, negHiddenProbs = self.negativeProb(posHiddenStates, k = step)
				negAssociations = np.dot(negVisibleProbs.T, negHiddenProbs)
				negHiddenBias = np.dot(np.ones(tmpData.shape[0]),negHiddenProbs)
				negVisibleBias = np.dot(negVisibleProbs.T, np.ones(tmpData.shape[0]).T)

				# Update weight and Bias
				self.weights += learningRate*((posAssociations-negAssociations)/numExamples)
				self.hBias += learningRate*((posHiddenBias-negHiddenBias)/numExamples)
				self.vBias += learningRate*(((posVisibleBias-negVisibleBias)*self.screen)/numExamples)

			# Check error for each epoch
			tmpHidden = self.getHidden(data)
			tmpVisible = self.getVisible(tmpHidden)
			tmpVisible = tmpVisible * data
			rmseError = math.sqrt(np.sum((data-tmpVisible)**2)/np.sum(data == 1))
			totalTime = time.time()-startTime

			tmpHidden1 = self.getHidden(test)
			tmpVisible1 = self.getVisible(tmpHidden1)
			tmpVisible1 = tmpVisible1 * test
			rmseError1 = math.sqrt(np.sum((test-tmpVisible1)**2)/np.sum(test == 1))

			print ('{0:7}Epoch : {1:5}  Time : {2:15} Train RMSE : {3:10}{4}'.format('INFO', epoch, totalTime, rmseError, rmseError1))

		# Save weights
		print ('{0:7}TotalTime : {1}'.format('INFO', time.time()-start))
		pickle.dump(self.weights, open(self.weightsObject,'wb'))
		pickle.dump(self.hBias, open(self.hBiasObject,'wb'))
		pickle.dump(self.vBias, open(self.vBiasObject,'wb'))
		pickle.dump(self.screen, open(self.screenObject,'wb'))	

if __name__ == '__main__':
	userRbm = RBM ('../data/Config.ini', 'UserRBM')
	filePointer = open('../data/DocumentInfo.dat')
	iterLines = iter(filePointer)

	# Read Data
	print('Loading')
	dataID = []
	data = []
	for lineNum, line in enumerate(iterLines):
		tmp = [0] * userRbm.numVisible
		ID = line.split('::')
		line = line.split('::')[1:]
		for doc in line:
			try:
				tmp[int(doc)] = int(1)
			except:
				tmpFalse = None
		data.append(tmp)
		dataID.append(ID[0])
	data = np.array(data)

	# data = np.array([[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,1,0], [0,0,1,1,0,0],[0,0,1,1,1,0]])
	# dataID = np.array([0,1,2,3,4,5])

	# Divide testing and training data
	print('Training')
	trainPart = 0.8
	trainSize = int(trainPart * len(data))
	train = np.array(data[:trainSize])
	test = np.array(data[trainSize:])
	userRbm.train(train, test)

	# Calculate all output
	print('Recalling')
	tmpHidden = userRbm.getHidden(data)
	tmpVisible = userRbm.getVisible(tmpHidden)
	# np.set_printoptions(threshold=np.nan)
	# print(tmpHidden)

	print('Calculating')
	f = open('../data/tmpOutputTableau.txt','w')
	f2 = open('../data/tmpOutput.txt','w')
	outputArray = {'key':'value'}
	for i in range(tmpVisible.shape[0]):
		maxTop = 10
		countTop = 0
		tmpValue = ''
		tmpList = sorted(range(len(tmpVisible[i])), key=lambda k: tmpVisible[i][k], reverse=True)
		for j in range(tmpVisible.shape[1]):
			if data[i][tmpList[j]] == 0:
				if countTop > -1:
					tmpValue += str(tmpList[j])
					f.write('{0},{1},2\n'.format(dataID[i],tmpList[j]))
					if (countTop < maxTop-1):
						tmpValue += '::'
				countTop = countTop + 1
				if countTop == maxTop:
					break
		outputArray[dataID[i]] = tmpValue
		f2.write('{0} {1}\n'.format(dataID[i],tmpValue))

	pickle.dump(outputArray, open('../data/tmpOutput.object','wb'))
	print('Done')







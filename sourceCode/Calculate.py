##################################################################
# Date    : 2016-10-24											
# Author  : Krittaphat Pugdeethosapol (krittaphat.pug@gmail.com)
# Version : 1.0													
##################################################################

import numpy as np
import math
import time
from datetime import datetime as dt

input1 = open("../data/MovieUserInfo.dat", 'r')
input2 = open("../data/MovieUserInfoOut.dat", 'r')

dataDict = {'key':[]}

iterlines = iter(input1)
for lineNumber, line in enumerate(iterlines):
	userID = line.split('::')[0]
	line = line.split('::')[1:]
	for data in line:
		if userID not in dataDict:
			dataDict[userID] = []
		if int(data.split(',')[0]) >= 3:
			dataDict[userID].append(int(data.split(',')[0]))
input1.close()

count = 0
matchID = 0
iterlines = iter(input2)
for lineNumber, line in enumerate(iterlines):
	line = line.rstrip()
	userID = line.split('::')[0]
	line = line.split('::')[1:50]
	if line[0] != '':
		for match in line:
			if int(match) in dataDict[userID]:
				count += 1
input2.close()
print((float(count)/float(150))*100)







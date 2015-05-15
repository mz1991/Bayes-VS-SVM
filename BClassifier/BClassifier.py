# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import itertools
import random
import collections

def GetAttributeValues(file):
	attrValues=collections.OrderedDict()
	dataLine = []
	for index,line in enumerate(file):
		if line[0] != '@':  #start of actual data
			#self.featureVectors.append(line.strip().lower().split(','))
			#print(line.strip().lower().split(','))
			if (line.strip().lower().split(',')!=[]):
				dataLine.append(line.strip().lower().split(','))
		else:   #feature definitions
			if line.strip().lower().find('@data') == -1 and (not line.lower().startswith('@relation')):
				#self.featureNameList.append(line.strip().split()[1])
				#self.features[self.featureNameList[len(self.featureNameList) - 1]] = line[line.find('{')+1: line.find('}')].strip().split(',')
				#print(line.strip().split()[1])
				#print(line[line.find('{')+1: line.find('}')].strip().split(','))
				attrValues[line.strip().split()[1]] = line[line.find('{')+1: line.find('}')].strip().split(',')
	file.close()
	return attrValues,dataLine

def GetClass(dataset):
	classUno =[]
	classMenoUno = []
	for line in dataset:
		#print(line)
		if (line[-1] == '-1'):
			classMenoUno.append(line)
		else:
			classUno.append(line)
	return classUno,classMenoUno	


def Learn(attributeClass,dataSet,columnIndex,resultType,attributeName,probabilitaResult):
	#calcolo P(colonna = attributeClass | result = 1)
	totLine = len(dataSet)
	dataSetColumn = list(zip(*dataSet))[columnIndex]
	for valueClass in attributeClass:
		totForClass=0
		for line in dataSetColumn:
			if (line == valueClass):
				totForClass+=1
		#print("P("+attributeName+"="+valueClass+"|"+"Result="+resultType+")="+str(totForClass/totLine))
		probabilitaResult[attributeName+"="+valueClass] = totForClass/totLine


def SubSamplingSplit(dataset,splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def Predict(testSet,probabilitaResultUno,probabilitaResultMenoUno,attrValues,type,prior={}):
	#P( result | calonna = k)
	predictionDict={}
	attrValuesCopy = attrValues.copy()
	del attrValuesCopy["Result"]
	for indexLine,line in enumerate(testSet):
		newLine = line[:]
		del newLine[-1]
		probUno=1
		probMenoUno=1

		for index,columnHeader in enumerate(attrValuesCopy):
			probUno = probUno * probabilitaResultUno[columnHeader+"="+newLine[index]]
			
		for index,columnHeader in enumerate(attrValuesCopy):
			probMenoUno= probMenoUno* probabilitaResultMenoUno[columnHeader+"="+newLine[index]]
			
		if (type == "MAP"):
			probMenoUno= probMenoUno* prior["Result=-1"]
			probUno = probUno * prior["Result=1"]
		elif (type == "ML"):
			probMenoUno= probMenoUno
			probUno = probUno
		elif (type == "NAIVE"):
			denNaive=1
			for index,columnHeader in enumerate(attrValuesCopy):
				denNaive*= prior[columnHeader+"="+newLine[index]]
			probMenoUno = probMenoUno / denNaive
			probUno = probUno / denNaive
		if (probMenoUno+probUno)!=0:
			alfa =1/(probMenoUno+probUno)
		else:
			alfa=1

		#print("Prob che vale -1: "+str(probMenoUno*alfa*100))
		#print("Prob che vale 1: "+str(probUno*alfa*100))
		if (probUno)>(probMenoUno):
			predictionDict[indexLine]=1
		else:
			predictionDict[indexLine]=-1	
	return predictionDict

def GetPrior(attributeClass,dataSet,columnIndex,attributeName,probabilitaResult):
	#calcolo P(colonna = attributeClass | result = 1)
	totLine = len(dataSet)
	dataSetColumn = list(zip(*dataSet))[columnIndex]
	for valueClass in attributeClass:
		totForClass=0
		for line in dataSetColumn:
			if (line == valueClass):
				totForClass+=1
		#print("P("+attributeName+"="+valueClass+"="+str(totForClass/totLine))
		probabilitaResult[attributeName+"="+valueClass] = totForClass/totLine

def VerifyPrediction(testSet,predictionDict):
	predictionRight=0
	predictionWrong=0
	for index,line in enumerate(testSet):
		if (str(line[-1]) == str(predictionDict[index])):
			predictionRight+=1
		else:
			predictionWrong+=1
	return (predictionRight/float(len(testSet)) )*100

def KFoldSplit(dataset,nFold):
	return [list(b) for b in zip(*[iter(dataset)]*(len(dataset)//nFold))]

def BayesianClassifier(trainingSet,testSet,attrValues):
	#data set diviso in Classi (Result = 1 e Result = -1)
	dataSetClassUno,dataSetClassMenoUno = GetClass(trainingSet)

	#Priorità a priori (utilizzata nel denominatore di Bayes)
	#Calcolo P(ColumnHeader)
	probabilitaPrior={}
	for index,attribute in enumerate(attrValues):
		GetPrior(attrValues[attribute],trainingSet,index,attribute,probabilitaPrior)


	#	Learn
	#	Calcolo della probabilita condizionata
	#	P(Attributo=value | Result = 1)
	#	P(Attributo=value | Result = -1)
	#	Viene calcolata per ogni possibile value dell'attributo

	#	P(Attr = value | Result =  1)
	probabilitaResultUno={}
	for index,attribute in enumerate(attrValues):
		Learn(attrValues[attribute],dataSetClassUno,index,"1",attribute,probabilitaResultUno)

	#	P(Attr = value | Result =  -1)
	probabilitaResultMenoUno={}
	for index,attribute in enumerate(attrValues):
		Learn(attrValues[attribute],dataSetClassMenoUno,index,"-1",attribute,probabilitaResultMenoUno)


	hMAPPrediction=0
	MLPrediction=0
	NAIVEPrediction=0

	#	HMAP 
	#	HMAP = max [ P(attribute = value| Result=1)* P(Result=1) , P(attribute = value|Result=-1)* P(Result=-1) ]
	predictionDict=Predict(testSet,probabilitaResultUno,probabilitaResultMenoUno,attrValues,"MAP",probabilitaPrior)
	hMAPPrediction=VerifyPrediction(testSet,predictionDict)
	print("HMAP prediction accuracy {0} %".format(hMAPPrediction))

	#	ML
	#	ML = max [ P(attribute = value|Result=1) , P(attribute = value|Result=-1) ] 
	predictionDict=Predict(testSet,probabilitaResultUno,probabilitaResultMenoUno,attrValues,"ML",probabilitaPrior)
	MLPrediction=VerifyPrediction(testSet,predictionDict)
	print("ML prediction accuracy {0} %".format(MLPrediction))

	#	NAIVE
	#	NAIVE = max = [ (P(attribute = value|Result=1)* P(Result=1)) / P(attribute = value) , (P(attribute = value|Result=-1)* P(Result=-1)) / P(attribute = value) ]
	predictionDict=Predict(testSet,probabilitaResultUno,probabilitaResultMenoUno,attrValues,"NAIVE",probabilitaPrior)
	NAIVEPrediction=VerifyPrediction(testSet,predictionDict)
	print("NAIVE prediction accuracy {0} %".format(NAIVEPrediction))
	
	return hMAPPrediction,MLPrediction,NAIVEPrediction


def main():
	filename="phi.arff"
	doShuffle = True
	kFoldSize=10
	#if doShuffle:
	#	random.shuffle(dataset)
	
	file = open(filename, 'r')
	
	# return the dataset line and the attributes dictionary (value and classes)
	attrValues,dataSet=GetAttributeValues(file)
	
	# mischia il dataset
	if doShuffle:
		random.shuffle(dataSet)

	#indice attributi da rimuovere nel dataset (viene rimossa l'intera colonna)
	attributeToRemove=[]
	for dataSetRow in dataSet:
		for index,attIndex in enumerate(attributeToRemove):
			# rimuovo attrIndex - index per evitare outOf Bound exception
			del dataSetRow[attIndex-index]
	
	hMAPsubsampling = 0
	MLsubsampling = 0
	NAIVEsubsampling = 0
	
	splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69]
	#splitRatioS=[]
	#split dataSet to training set and test set
	for index,splitRatio in enumerate(splitRatioS):
		print("Index {0} split ration {1}".format(index,splitRatio))
		trainingSet, testSet = SubSamplingSplit(dataSet,splitRatio)
		hMAPPrediction,MLPrediction,NAIVEPrediction = BayesianClassifier(trainingSet,testSet,attrValues)
		hMAPsubsampling+=hMAPPrediction
		MLsubsampling+=MLPrediction
		NAIVEsubsampling+=NAIVEPrediction
		print("--------------")
	
	hMAPfold = 0
	MLfold = 0
	NAIVEfold = 0

	folds=KFoldSplit(dataSet,kFoldSize)	
	for index,fold in enumerate(folds):
		print("Index {0} split K-Fold {1}".format(index,kFoldSize))
		testSet = fold
		copy = folds[:]
		del copy[index]
		trainingSet = list(itertools.chain(*copy))
		hMAPPrediction,MLPrediction,NAIVEPrediction = BayesianClassifier(trainingSet,testSet,attrValues)
		hMAPfold+=hMAPPrediction
		MLfold+=MLPrediction
		NAIVEfold+=NAIVEPrediction
		print("--------------")


	print("HMAP - subsampling avarage: {0} %".format(hMAPsubsampling/float(len(splitRatioS))))
	print("ML   - subsampling avarage: {0} %".format(MLsubsampling/float(len(splitRatioS))))
	print("NAIVE- subsampling avarage: {0} %".format(NAIVEsubsampling/float(len(splitRatioS))))

	print("HMAP - k-folding avarage: {0} %".format(hMAPfold/float(kFoldSize)))
	print("ML   - k-folding avarage: {0} %".format(MLfold/float(kFoldSize)))
	print("NAIVE- k-folding avarage: {0} %".format(NAIVEfold/float(kFoldSize)))

main()
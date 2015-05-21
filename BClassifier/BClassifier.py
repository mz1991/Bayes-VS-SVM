# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import itertools
import random
import collections
import time
import operator
# import librerie machine learning
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
import matplotlib.pyplot as plt

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
		#probabilitaResult[attributeName+"="+valueClass] = totForClass/totLine
		probabilitaResult[attributeName+"="+valueClass] = totForClass/float(totLine)


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
			sum = 1
			probMenoUno = (probMenoUno*prior["Result=-1"]) / float(sum)
			probUno = (probUno*prior["Result=1"]) / float(sum)
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
		probabilitaResult[attributeName+"="+valueClass] = totForClass/float(totLine)

def VerifyPrediction(testSet,predictionDict):
	predictionRight=0
	predictionWrong=0
	for index,line in enumerate(testSet):
		if (str(line[-1]) == str(predictionDict[index])):
			predictionRight+=1
		else:
			predictionWrong+=1
	return (predictionRight/float(len(testSet)) )*100

def GetTypeError(testSet,predictionDict):
	# per valida si intende: e un sito di phishing (cioe rispondo alla domanda: "il sito e di phishing")
	# sito phishing = 1
	# sito non phishing = -1
	falsiPositivi =0	# ipotesi valida, rifiutata
	falsiNegativi =0	# ipotesi sbagliata accettata
	veriPositivi  =0	# ipotesi valida accettata
	veriNegativi  =0	# ipotesi sbagliata rifiutata
	for index,line in enumerate(testSet):
		if (str(line[-1])=="1") and (str(predictionDict[index])=="1"):
			veriNegativi+=1
		if (str(line[-1])=="-1") and (str(predictionDict[index])=="-1"):
			veriPositivi+=1
		if (str(line[-1])=="1") and (str(predictionDict[index])=="-1"):
			falsiPositivi+=1
		if (str(line[-1])=="-1") and (str(predictionDict[index])=="1"):
			falsiNegativi+=1
	return falsiPositivi,falsiNegativi,veriPositivi,veriNegativi

def KFoldSplit(dataset,nFold):
	return [list(b) for b in zip(*[iter(dataset)]*(len(dataset)//nFold))]

def BayesianClassifier(trainingSet,testSet,attrValues):
	#data set diviso in Classi (Result = 1 e Result = -1)
	dataSetClassUno,dataSetClassMenoUno = GetClass(trainingSet)

	#Priorita a priori (utilizzata nel denominatore di Bayes)
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
	hMAPfalsiPositivi,hMAPfalsiNegativi,hMAPveriPositivi,hMAPveriNegativi = GetTypeError(testSet,predictionDict)
	
	if GLOBAL_verbose:
		print("HMAP prediction accuracy {0} %".format(hMAPPrediction))
		print("HMAP falsi positivi {0} of {1}".format(hMAPfalsiPositivi,len(testSet)))
		print("HMAP falsi negativi {0} of {1}".format(hMAPfalsiNegativi,len(testSet)))
		print("HMAP veri positivi {0} of {1}".format(hMAPveriPositivi,len(testSet)))
		print("HMAP veri negativo {0} of {1}".format(hMAPveriNegativi,len(testSet)))
		print("HMAP sensitivita {0}".format(hMAPveriPositivi/(hMAPveriPositivi+hMAPfalsiNegativi)))
		print("HMAP specificita {0}".format(hMAPveriNegativi/(hMAPfalsiPositivi+hMAPveriNegativi)))

	#	ML
	#	ML = max [ P(attribute = value|Result=1) , P(attribute = value|Result=-1) ] 
	predictionDict=Predict(testSet,probabilitaResultUno,probabilitaResultMenoUno,attrValues,"ML",probabilitaPrior)
	MLPrediction=VerifyPrediction(testSet,predictionDict)
	MLfalsiPositivi,MLfalsiNegativi,MLveriPositivi,MLveriNegativi = GetTypeError(testSet,predictionDict)

	if GLOBAL_verbose:
		print("ML prediction accuracy {0} %".format(MLPrediction))
		print("ML falsi positivi {0} of {1}".format(MLfalsiPositivi,len(testSet)))
		print("ML falsi negativi {0} of {1}".format(MLfalsiNegativi,len(testSet)))
		print("ML veri positivi {0} of {1}".format(MLveriPositivi,len(testSet)))
		print("ML veri negativo {0} of {1}".format(MLveriNegativi,len(testSet)))
		print("ML sensitivita {0}".format(MLveriPositivi/(MLveriPositivi+MLfalsiNegativi)))
		print("ML specificita {0}".format(MLveriNegativi/(MLfalsiPositivi+MLveriNegativi)))

	#	NAIVE
	#	NAIVE = max = [ (P(attribute = value|Result=1)* P(Result=1)) / P(attribute = value) , (P(attribute = value|Result=-1)* P(Result=-1)) / P(attribute = value) ]
	predictionDict=Predict(testSet,probabilitaResultUno,probabilitaResultMenoUno,attrValues,"NAIVE",probabilitaPrior)
	NAIVEPrediction=VerifyPrediction(testSet,predictionDict)
	NAIVEfalsiPositivi,NAIVEfalsiNegativi,NAIVEveriPositivi,NAIVEveriNegativi = GetTypeError(testSet,predictionDict)
	
	if GLOBAL_verbose:
		print("NAIVE prediction accuracy {0} %".format(NAIVEPrediction))
		print("NAIVE falsi positivi {0} of {1}".format(NAIVEfalsiPositivi,len(testSet)))
		print("NAIVE falsi negativi {0} of {1}".format(NAIVEfalsiNegativi,len(testSet)))
		print("NAIVE veri positivi {0} of {1}".format(NAIVEveriPositivi,len(testSet)))
		print("NAIVE veri negativo {0} of {1}".format(NAIVEveriNegativi,len(testSet)))
		print("NAIVE sensitivita {0}".format(NAIVEveriPositivi/(NAIVEveriPositivi+NAIVEfalsiNegativi)))
		print("NAIVE specificita {0}".format(NAIVEveriNegativi/(NAIVEfalsiPositivi+NAIVEveriNegativi)))
	return hMAPPrediction,MLPrediction,NAIVEPrediction


def main(filename,doShuffle,splitRatioS,kFoldSize=10,sizeColumnsToKeep=30):

	# print parameters
	print("Filename: {0} | Cross Validation Size: {1} | Numer of feature selected: {2}".format(filename,kFoldSize,sizeColumnsToKeep))
	GLOBAL_asseX.append(sizeColumnsToKeep)

	# start timer (gestione calcolo tempo di esecuzione)
	start_time = time.time()

	#filename="phi.arff"
	#doShuffle = True
	#kFoldSize=10
	# numero di migliori colonne su cui fare train (max 30!!!)
	#sizeColumnsToKeep=28

	# parametri di Debug
	useLibrary=True
	# Esegui divisione training set e learning set in modo casuale
	doSubSampling = True
	# Esegui cross - validation
	doKFolding = True	
	
	file = open(filename, 'r')
	
	# return the dataset line and the attributes dictionary (value and classes)
	attrValues,dataSet=GetAttributeValues(file)
	
	columnHeader=[]
	for header in attrValues:
		columnHeader.append(header)	

	# randomizza il dataset
	if doShuffle:
		random.shuffle(dataSet)

	if useLibrary:
		# Crea dati per la feature selection
		# Prende tutto il dataset esclusa la colonna target (ultima colonna)
		dataCopy = list()
		for line_aa in dataSet:
			dataCopy.append(line_aa[0:-1])
		
		# colonna target
		lastColumn = list(zip(*dataSet))[-1]

		# Feature selection : http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
		model = ExtraTreesClassifier(n_estimators=100,criterion='entropy')
		model.fit(dataCopy, lastColumn)
		
		# Recupera gli indici delle feature migliori
		featureWeight = model.feature_importances_
		bestIndex=list()
		for i in xrange(0,sizeColumnsToKeep):
			index, value = max(enumerate(featureWeight), key=operator.itemgetter(1))
			#print(index)
			#print(value)
			bestIndex.append(index)
			featureWeight[index]=-1

		def diff(a, b):
			b = set(b)
			return [aa for aa in a if aa not in b]
	
		allIndexes = range(0,30)

		if GLOBAL_verbose: print("Column to keep")
		for i in bestIndex:
			if GLOBAL_verbose: print(columnHeader[i])

		# Indici colonne da rimuovere
		attributeToRemove=diff(allIndexes,bestIndex)
		
		# SUpport Vector Machine
		clf = svm.SVC()
		clf = svm.SVC(gamma=0.001, C=100,cache_size=5,kernel='linear',verbose=False)
		for dataSetRowCopy in dataCopy:
			for index,attIndex in enumerate(attributeToRemove):
				# rimuovo attrIndex - index per evitare outOf Bound exception
				del dataSetRowCopy[attIndex-index]
		# Training
		clf.fit(dataCopy,lastColumn)
		valoriGiusti=0
		for indexAA,a in enumerate(dataCopy):
			# Predizione
			predizioneSVC= clf.predict(a)[0]
			# Verifica predizione
			valoreReale =lastColumn[indexAA]
			if (str(predizioneSVC) == str(valoreReale)):
				valoriGiusti +=1
		
	#attributeToRemove=[0,1,2,3,4,5,6,7,8,9]

	for dataSetRow in dataSet:
		for index,attIndex in enumerate(attributeToRemove):
			# rimuovo attrIndex - index per evitare outOf Bound exception
			del dataSetRow[attIndex-index]
	#rimuovere anche gli attributi (header delle colonne)
	for index,idx in enumerate(attributeToRemove):
		if GLOBAL_verbose: print("Header column removed: {0}".format(columnHeader[idx]))
		del attrValues[columnHeader[idx]]
	
	if doSubSampling:
		hMAPsubsampling = 0
		MLsubsampling = 0
		NAIVEsubsampling = 0
	
		#splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69]
		#split dataSet to training set and test set
		for index,splitRatio in enumerate(splitRatioS):
			if GLOBAL_verbose: print("Index {0} split ration {1}".format(index,splitRatio))
			trainingSet, testSet = SubSamplingSplit(dataSet,splitRatio)
			hMAPPrediction,MLPrediction,NAIVEPrediction = BayesianClassifier(trainingSet,testSet,attrValues)
			hMAPsubsampling+=hMAPPrediction
			MLsubsampling+=MLPrediction
			NAIVEsubsampling+=NAIVEPrediction
			if GLOBAL_verbose: print("--------------")
		
		GLOBAL_asseY_MAP_SUB.append(hMAPsubsampling/float(len(splitRatioS)))
		GLOBAL_asseY_ML_SUB.append(MLsubsampling/float(len(splitRatioS)))
		print("HMAP - subsampling accuracy avarage: {0} %".format(hMAPsubsampling/float(len(splitRatioS))))
		print("ML   - subsampling accuracy avarage: {0} %".format(MLsubsampling/float(len(splitRatioS))))
		#print("NAIVE- subsampling avarage: {0} %".format(NAIVEsubsampling/float(len(splitRatioS))))

	hMAPfold = 0
	MLfold = 0
	NAIVEfold = 0

	if doKFolding:
		folds=KFoldSplit(dataSet,kFoldSize)	
		for index,fold in enumerate(folds):
			if GLOBAL_verbose: print("Index {0} split K-Fold {1}".format(index,kFoldSize))
			testSet = fold
			copy = folds[:]
			del copy[index]
			trainingSet = list(itertools.chain(*copy))
			hMAPPrediction,MLPrediction,NAIVEPrediction = BayesianClassifier(trainingSet,testSet,attrValues)
			hMAPfold+=hMAPPrediction
			MLfold+=MLPrediction
			NAIVEfold+=NAIVEPrediction
			if GLOBAL_verbose: print("--------------")

		GLOBAL_asseY_MAP_KFOL.append(hMAPfold/float(kFoldSize))
		GLOBAL_asseY_ML_KFOL.append(MLfold/float(kFoldSize))
		print("HMAP - k-folding accuracy avarage: {0} %".format(hMAPfold/float(kFoldSize)))
		print("ML   - k-folding accuracy avarage: {0} %".format(MLfold/float(kFoldSize)))
		#print("NAIVE- k-folding avarage: {0} %".format(NAIVEfold/float(kFoldSize)))

	GLOBAL_asseY_SVM.append(valoriGiusti/float(len(dataCopy))*100)
	print("SVM	- accuracy: {0}".format((valoriGiusti/float(len(dataCopy)))*100))
	print("------------------------")
	print("--- %s seconds ---" % (time.time() - start_time))


global GLOBAL_verbose
GLOBAL_verbose= False

global GLOBAL_asseX
GLOBAL_asseX= []

global GLOBAL_asseY_MAP_SUB
GLOBAL_asseY_MAP_SUB= []

global GLOBAL_asseY_MAP_KFOL
GLOBAL_asseY_MAP_KFOL= []

global GLOBAL_asseY_ML_SUB
GLOBAL_asseY_ML_SUB= []

global GLOBAL_asseY_ML_KFOL
GLOBAL_asseY_ML_KFOL= []

global GLOBAL_asseY_SVM
GLOBAL_asseY_SVM= []

#TODO: add falsi positivi ecc

## Parameters array

# number of tests! (30)
sizeColumnsToKeepArray = list(range(1,31))
for x in range(0, 30):
	main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=sizeColumnsToKeepArray[x])


# Test LINE!

#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=1)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=2)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=3)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=4)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=5)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=6)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=7)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=8)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=9)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=10)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=11)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=12)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=13)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=14)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=15)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=16)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=17)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=18)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=19)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=20)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=21)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=22)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=23)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=24)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=25)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=26)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=27)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=28)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=29)
#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=30)

plt.plot(GLOBAL_asseX,GLOBAL_asseY_SVM)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAP_SUB)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_ML_SUB)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAP_KFOL)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_ML_KFOL)
plt.legend(['Support Vector Machine', 'HMAP Subsampling', 'ML Subsampling', 'HMAP Cross Validation','ML Cross Validation'], loc='lower right')
plt.ylabel('Accuracy %')
plt.xlabel('Number of features')
# set asse x interval ( 1 step )
plt.xticks(range(0, int(max(GLOBAL_asseX))+1, 2))
plt.yticks(range(88, 98, 2))
plt.show()
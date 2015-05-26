import random
import itertools
import collections
import time
import operator
# import librerie machine learning
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

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

		# if (probMenoUno+probUno)!=0:
		# 	 alfa =1/(probMenoUno+probUno)
		# else:
		#	 alfa=1

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
	

	dictionaryTypeError={}

	if GLOBAL_verbose:
		print("HMAP prediction accuracy {0} %".format(hMAPPrediction))
		print("HMAP falsi positivi {0} of {1}".format(hMAPfalsiPositivi,len(testSet)))
		print("HMAP falsi negativi {0} of {1}".format(hMAPfalsiNegativi,len(testSet)))
		print("HMAP veri positivi {0} of {1}".format(hMAPveriPositivi,len(testSet)))
		print("HMAP veri negativo {0} of {1}".format(hMAPveriNegativi,len(testSet)))
		print("HMAP sensitivita {0}".format(hMAPveriPositivi/float((hMAPveriPositivi+hMAPfalsiNegativi))))
		print("HMAP specificita {0}".format(hMAPveriNegativi/float((hMAPfalsiPositivi+hMAPveriNegativi))))

	dictionaryTypeError["HMAP_falsipositivi"]=(hMAPfalsiPositivi/float(len(testSet)))*100
	dictionaryTypeError["HMAP_falsinegativi"]=(hMAPfalsiNegativi/float(len(testSet)))*100
	dictionaryTypeError["HMAP_veripositivi"]=(hMAPveriPositivi/float(len(testSet)))*100
	dictionaryTypeError["HMAP_verinegativi"]=(hMAPveriNegativi/float(len(testSet)))*100
	dictionaryTypeError["HMAP_sensitivita"]=hMAPveriPositivi/float((hMAPveriPositivi+hMAPfalsiNegativi))
	dictionaryTypeError["HMAP_specificita"]=hMAPveriNegativi/float((hMAPfalsiPositivi+hMAPveriNegativi))

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
		print("ML sensitivita {0}".format(MLveriPositivi/float((MLveriPositivi+MLfalsiNegativi))))
		print("ML specificita {0}".format(MLveriNegativi/float((MLfalsiPositivi+MLveriNegativi))))

	dictionaryTypeError["ML_falsipositivi"]=(MLfalsiPositivi/float(len(testSet)))*100
	dictionaryTypeError["ML_falsinegativi"]=(MLfalsiNegativi/float(len(testSet)))*100
	dictionaryTypeError["ML_veripositivi"]=(MLveriPositivi/float(len(testSet)))*100
	dictionaryTypeError["ML_verinegativi"]=(MLveriNegativi/float(len(testSet)))*100
	dictionaryTypeError["ML_sensitivita"]=MLveriPositivi/float((MLveriPositivi+MLfalsiNegativi))
	dictionaryTypeError["ML_specificita"]=MLveriNegativi/float((MLfalsiPositivi+MLveriNegativi))

	# NON e' da fare, identico a HMAP!! (stesso denominatore)
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
	return hMAPPrediction,MLPrediction,NAIVEPrediction,dictionaryTypeError

class SVMTestResult:
	def __init__(self):
		self.lenDataCopySVM = 0
		self.valoriGiusti = 0
		self.falsiPositiviSVM = 0  # ipotesi valida, rifiutata
		self.falsiNegativiSVM = 0  # ipotesi sbagliata accettata
		self.veriPositiviSVM = 0	  # ipotesi valida accettata
		self.veriNegativiSVM = 0	  # ipotesi sbagliata rifiutata

def tryTestWithSVM(kernel, dataCopy,testSetSVM, lastColumnTrainingSet,lastColumnTestSet):
	if (kernel == 'poly'):
		#clf = svm.SVC(kernel='poly', C=0.9, degree=3, verbose=False)
		clf = svm.SVC(kernel='poly', verbose=False)
	elif (kernel == 'sigmoid'):
		#clf = svm.SVC(kernel='sigmoid', gamma=1/float(len(dataCopy)), C=100, verbose=False)
		clf = svm.SVC(kernel='sigmoid', C=0.1, degree=1, gamma=0.7, coef0=1, verbose=False)
	else:
		#clf = svm.SVC(kernel=kernel, C=1.0, coef0=0.0, verbose=False)
		clf = svm.SVC(kernel=kernel, verbose=False)

	# Training
	clf.fit(dataCopy, lastColumnTrainingSet)
	print(clf)
	valoriGiusti=0
	lenDataCopySVM = len(testSetSVM)
	# check error type
	falsiPositiviSVM =0	# ipotesi valida, rifiutata
	falsiNegativiSVM =0	# ipotesi sbagliata accettata
	veriPositiviSVM  =0	# ipotesi valida accettata
	veriNegativiSVM  =0	# ipotesi sbagliata rifiutata
	for indexAA,a in enumerate(testSetSVM):
		# Predizione
		predizioneSVC= clf.predict(a)[0]
		# Verifica predizione
		valoreReale =lastColumnTestSet[indexAA]
		if (str(predizioneSVC) == str(valoreReale)):
			valoriGiusti +=1
		
		if (str(valoreReale)=="1") and (str(predizioneSVC)=="1"):
			veriNegativiSVM+=1
		if (str(valoreReale)=="-1") and (str(predizioneSVC)=="-1"):
			veriPositiviSVM+=1
		if (str(valoreReale)=="1") and (str(predizioneSVC)=="-1"):
			falsiPositiviSVM+=1
		if (str(valoreReale)=="-1") and (str(predizioneSVC)=="1"):
			falsiNegativiSVM+=1
	ret = SVMTestResult()
	ret.lenDataCopySVM = lenDataCopySVM
	ret.valoriGiusti = valoriGiusti
	ret.falsiPositiviSVM = falsiPositiviSVM
	ret.falsiNegativiSVM = falsiNegativiSVM
	ret.veriPositiviSVM = veriPositiviSVM
	ret.veriNegativiSVM = veriNegativiSVM
	return ret

def testWithSVM(k, c, d, g, c0, trainingSet, testSet, lastColumnTrainingSet, lastColumnTestSet):
	clf = svm.SVC(kernel=k, C=c, degree=d, gamma=g, coef0=c0, verbose=False)	
	# Training
	clf.fit(trainingSet, lastColumnTrainingSet)
	valoriGiusti=0
	# check error type
	falsiPositiviSVM =0	# ipotesi valida, rifiutata
	falsiNegativiSVM =0	# ipotesi sbagliata accettata
	veriPositiviSVM  =0	# ipotesi valida accettata
	veriNegativiSVM  =0	# ipotesi sbagliata rifiutata
	for i in range(len(testSet)):
		# Predizione
		predizioneSVC = clf.predict(testSet[i])[0]
		# Verifica predizione
		valoreReale = lastColumnTestSet[i]
		if (str(predizioneSVC) == str(valoreReale)):
			valoriGiusti += 1
		
		if (str(valoreReale)=="1") and (str(predizioneSVC)=="1"):
			veriNegativiSVM += 1
		if (str(valoreReale)=="-1") and (str(predizioneSVC)=="-1"):
			veriPositiviSVM += 1
		if (str(valoreReale) == "1") and (str(predizioneSVC)=="-1"):
			falsiPositiviSVM += 1
		if (str(valoreReale) == "-1") and (str(predizioneSVC)=="1"):
			falsiNegativiSVM += 1
	print('risultati giusti', valoriGiusti, len(testSet), valoriGiusti / len(testSet))
 
def processSVM(svmData, globalData, label, startTime):
	globalData.asseY.append(svmData.valoriGiusti/float(svmData.lenDataCopySVM)*100)
	globalData.asseY_fp.append(svmData.falsiPositiviSVM/float(svmData.lenDataCopySVM)*100)
	globalData.asseY_fn.append(svmData.falsiNegativiSVM/float(svmData.lenDataCopySVM)*100)
	globalData.asseY_vp.append(svmData.veriPositiviSVM/float(svmData.lenDataCopySVM)*100)
	globalData.asseY_vn.append(svmData.veriNegativiSVM/float(svmData.lenDataCopySVM)*100)

	globalData.asseY_Sensitivita.append(svmData.veriPositiviSVM/(float(svmData.veriPositiviSVM+svmData.falsiNegativiSVM)))
	globalData.asseY_Specificita.append(svmData.veriNegativiSVM/(float(svmData.falsiPositiviSVM+svmData.veriNegativiSVM)))	

	print("SVM " + label + " - accuracy: {0}".format((svmData.valoriGiusti/float(svmData.lenDataCopySVM))*100))
	print("SVM " + label + " - falsi positivi: {0}".format((svmData.falsiPositiviSVM/float(svmData.lenDataCopySVM))*100))
	print("SVM " + label + " - falsi negativi: {0}".format((svmData.falsiNegativiSVM/float(svmData.lenDataCopySVM))*100))
	print("SVM " + label + " - veri positivi: {0}".format((svmData.veriPositiviSVM/float(svmData.lenDataCopySVM))*100))
	print("SVM " + label + " - veri negativi: {0}".format((svmData.veriNegativiSVM/float(svmData.lenDataCopySVM))*100))
	print("------------------------")
	print("--- %s seconds ---" % (time.time() - startTime))

def main(filename,
         doShuffle,
         splitRatioS,
         kFoldSize=10,
         sizeColumnsToKeep=30,
         typeOfFeatureSelection="TreeClassiffier"):

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
		
		if typeOfFeatureSelection == "TreeClassiffier":
			# Feature selection : http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
			model = ExtraTreesClassifier(n_estimators=100,criterion='entropy')
			model.fit(dataCopy, lastColumn)
			# Recupera gli indici delle feature migliori
			featureWeight = model.feature_importances_

			bestIndex=list()
			for i in range(0,sizeColumnsToKeep):
				index, value = max(enumerate(featureWeight), key=operator.itemgetter(1))
				#print(index)
				#print(value)
				bestIndex.append(index)
				featureWeight[index]=-1

		elif typeOfFeatureSelection == "RecursiveFeatureElimination":
			model = LogisticRegression()
			# create the RFE model and select 3 attributes
			rfe = RFE(model, sizeColumnsToKeep)
			rfe = rfe.fit(dataCopy, lastColumn)
			#print(rfe.support_)
			#print(rfe.ranking_)
			bestIndex=list()
			for index,i in enumerate(rfe.ranking_):
				if i == 1:
					bestIndex.append(index)
				i=-1
		
		def diff(a, b):
			b = set(b)
			return [aa for aa in a if aa not in b]
	
		allIndexes = range(0,30)

		if GLOBAL_verbose: print("Column to keep")
		for i in bestIndex:
			if GLOBAL_verbose: print(columnHeader[i])

		# Indici colonne da rimuovere
		attributeToRemove=diff(allIndexes,bestIndex)

		for dataSetRowCopy in dataCopy:
			for index,attIndex in enumerate(attributeToRemove):
				# rimuovo attrIndex - index per evitare outOf Bound exception
				del dataSetRowCopy[attIndex-index]  
		
		# divide dataset in 50% in trainins set e 50% in tests set
		dcLen = len(dataCopy);
		dcLenM = int(dcLen / 2)
		print('lens', dcLen, dcLenM)
		trainingSetSVM = dataCopy[0:dcLenM]
		testSetSVM = dataCopy[dcLenM:dcLen]
		lastColumnTrainingSet = lastColumn[0:dcLenM]
		lastColumnTestSet = lastColumn[dcLenM:dcLen]
		print('dataset len', len(trainingSetSVM), len(testSetSVM), len(lastColumnTrainingSet), len(lastColumnTestSet))

		# linear 85%
		# rbf 85%
		# poly 85%
		# sigmoid 85%  C=0.1 degree=1 gamma=0.7 coef0=1
		#for k in ['sigmoid']:
		#	for c in [0.1, 0.5, 1.0, 5, 10, 25, 50, 100, 250]:
		#		for d in [1, 3, 5, 6, 7]:
		#			for g in [0.1, 0.5, 0.7]:
		#				for c0 in [1, 5, 15]:
		#					print('pars', k, c, d, g, c0)
		#					testWithSVM(k, c, d, g, c0, trainingSetSVM,testSetSVM, lastColumnTrainingSet, lastColumnTestSet)

		svmPoly = tryTestWithSVM('poly', trainingSetSVM,testSetSVM, lastColumnTrainingSet,lastColumnTestSet)
		svmRbf = tryTestWithSVM('rbf', trainingSetSVM,testSetSVM, lastColumnTrainingSet,lastColumnTestSet)
		svmLinear = tryTestWithSVM('linear', trainingSetSVM,testSetSVM, lastColumnTrainingSet,lastColumnTestSet)
		svmSigmoid = tryTestWithSVM('sigmoid', trainingSetSVM,testSetSVM, lastColumnTrainingSet,lastColumnTestSet)

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
	
		# fp = falsipositivi
		hMAPsubsampling_fp=0
		# fn = falsinegativi
		hMAPsubsampling_fn=0
		# vp = veri positivi
		hMAPsubsampling_vp=0
		# vn = verinegativi
		hMAPsubsampling_vn=0

		# fp = falsipositivi
		MLsubsampling_fp=0
		# fn = falsinegativi
		MLsubsampling_fn=0
		# vp = veri positivi
		MLsubsampling_vp=0
		# vn = verinegativi
		MLsubsampling_vn=0

		MLsub_sensitivita=0
		MLsub_specificita=0

		hMAPsub_sensitivita=0
		hMAPsub_specificita=0			

		#splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69]
		#split dataSet to training set and test set
		for index,splitRatio in enumerate(splitRatioS):
			if GLOBAL_verbose: print("Index {0} split ration {1}".format(index,splitRatio))
			trainingSet, testSet = SubSamplingSplit(dataSet,splitRatio)
			hMAPPrediction,MLPrediction,NAIVEPrediction,dictionaryTypeError = BayesianClassifier(trainingSet,testSet,attrValues)
			hMAPsubsampling+=hMAPPrediction
			MLsubsampling+=MLPrediction
			NAIVEsubsampling+=NAIVEPrediction

			hMAPsubsampling_fp += dictionaryTypeError["HMAP_falsipositivi"]
			hMAPsubsampling_fn += dictionaryTypeError["HMAP_falsinegativi"]
			hMAPsubsampling_vp += dictionaryTypeError["HMAP_veripositivi"]
			hMAPsubsampling_vn += dictionaryTypeError["HMAP_verinegativi"]	
			MLsubsampling_fp   += dictionaryTypeError["ML_falsipositivi"]
			MLsubsampling_fn   += dictionaryTypeError["ML_falsinegativi"]
			MLsubsampling_vp   += dictionaryTypeError["ML_veripositivi"]
			MLsubsampling_vn   += dictionaryTypeError["ML_verinegativi"]
			MLsub_sensitivita+=dictionaryTypeError["ML_sensitivita"]
			MLsub_specificita+=dictionaryTypeError["ML_specificita"]
			hMAPsub_sensitivita+=dictionaryTypeError["HMAP_sensitivita"]
			hMAPsub_specificita+=dictionaryTypeError["HMAP_specificita"]

			if GLOBAL_verbose: print("--------------")
		
		GLOBAL_asseY_MAP_SUB.append(hMAPsubsampling/float(len(splitRatioS)))
		GLOBAL_asseY_ML_SUB.append(MLsubsampling/float(len(splitRatioS)))
		
		GLOBAL_asseY_MAPsub_fp.append(hMAPsubsampling_fp/float(len(splitRatioS)))
		GLOBAL_asseY_MAPsub_fn.append(hMAPsubsampling_fn/float(len(splitRatioS)))
		GLOBAL_asseY_MAPsub_vp.append(hMAPsubsampling_vp/float(len(splitRatioS)))
		GLOBAL_asseY_MAPsub_vn.append(hMAPsubsampling_vn/float(len(splitRatioS)))

		GLOBAL_asseY_MLsub_fp.append(MLsubsampling_fp/float(len(splitRatioS)))
		GLOBAL_asseY_MLsub_fn.append(MLsubsampling_fn/float(len(splitRatioS)))
		GLOBAL_asseY_MLsub_vp.append(MLsubsampling_vp/float(len(splitRatioS)))
		GLOBAL_asseY_MLsub_vn.append(MLsubsampling_vn/float(len(splitRatioS)))

		GLOBAL_asseY_MLsub_Sensitivita.append(MLsub_sensitivita/float(kFoldSize))
		GLOBAL_asseY_MLsub_Specificita.append(MLsub_specificita/float(kFoldSize))
		GLOBAL_asseY_MAPsub_Sensitivita.append(hMAPsub_sensitivita/float(kFoldSize))
		GLOBAL_asseY_MAPsub_Specificita.append(hMAPsub_specificita/float(kFoldSize))	
	

		print("HMAP - subsampling accuracy avarage: {0} %".format(hMAPsubsampling/float(len(splitRatioS))))
		print("HMAP - subsampling falsi positivi: {0} %".format(hMAPsubsampling_fp/float(len(splitRatioS))))
		print("HMAP - subsampling falsi negativi: {0} %".format(hMAPsubsampling_fn/float(len(splitRatioS))))
		print("HMAP - subsampling veri positivi: {0} %".format(hMAPsubsampling_vp/float(len(splitRatioS))))
		print("HMAP - subsampling veri negativi: {0} %".format(hMAPsubsampling_vn/float(len(splitRatioS))))
		print("ML - subsampling falsi positivi: {0} %".format(MLsubsampling_fp/float(len(splitRatioS))))
		print("ML - subsampling falsi negativi: {0} %".format(MLsubsampling_fn/float(len(splitRatioS))))
		print("ML - subsampling veri positivi: {0} %".format(MLsubsampling_vp/float(len(splitRatioS))))
		print("ML - subsampling veri negativi: {0} %".format(MLsubsampling_vn/float(len(splitRatioS))))
		
		print("ML   - subsampling accuracy avarage: {0} %".format(MLsubsampling/float(len(splitRatioS))))
		#print("NAIVE- subsampling avarage: {0} %".format(NAIVEsubsampling/float(len(splitRatioS))))

	hMAPfold = 0
	MLfold = 0
	NAIVEfold = 0
	
	# fp = falsipositivi
	hMAPfold_fp=0
	# fn = falsinegativi
	hMAPfold_fn=0

	# vp = veri positivi
	hMAPfold_vp=0
	# vn = verinegativi
	hMAPfold_vn=0

	hMAPfold_sensitivita=0
	hMAPfold_specificita=0

	# fp = falsipositivi
	MLfold_fp=0
	# fn = falsinegativi
	MLfold_fn=0
	# vp = veri positivi
	MLfold_vp=0
	# vn = verinegativi
	MLfold_vn=0

	MLfold_sensitivita=0
	MLfold_specificita=0
	
	if doKFolding:
		folds=KFoldSplit(dataSet,kFoldSize)	
		for index,fold in enumerate(folds):
			if GLOBAL_verbose: print("Index {0} split K-Fold {1}".format(index,kFoldSize))
			testSet = fold
			copy = folds[:]
			del copy[index]
			trainingSet = list(itertools.chain(*copy))
			hMAPPrediction,MLPrediction,NAIVEPrediction,dictionaryTypeError = BayesianClassifier(trainingSet,testSet,attrValues)
			hMAPfold+=hMAPPrediction
			MLfold+=MLPrediction
			NAIVEfold+=NAIVEPrediction

			hMAPfold_fp += dictionaryTypeError["HMAP_falsipositivi"]
			hMAPfold_fn += dictionaryTypeError["HMAP_falsinegativi"]
			hMAPfold_vp += dictionaryTypeError["HMAP_veripositivi"]
			hMAPfold_vn += dictionaryTypeError["HMAP_verinegativi"]	
			hMAPfold_specificita += dictionaryTypeError["HMAP_specificita"]
			hMAPfold_sensitivita += dictionaryTypeError["HMAP_sensitivita"]
			MLfold_fp   += dictionaryTypeError["ML_falsipositivi"]
			MLfold_fn   += dictionaryTypeError["ML_falsinegativi"]
			MLfold_vp   += dictionaryTypeError["ML_veripositivi"]
			MLfold_vn   += dictionaryTypeError["ML_verinegativi"]
			MLfold_sensitivita += dictionaryTypeError["ML_sensitivita"]
			MLfold_specificita += dictionaryTypeError["ML_specificita"]
			
			if GLOBAL_verbose: print("--------------")

		GLOBAL_asseY_MAP_KFOL.append(hMAPfold/float(kFoldSize))
		GLOBAL_asseY_ML_KFOL.append(MLfold/float(kFoldSize))

		GLOBAL_asseY_MAPfold_fp.append(hMAPfold_fp/float(kFoldSize))
		GLOBAL_asseY_MAPfold_fn.append(hMAPfold_fn/float(kFoldSize))
		GLOBAL_asseY_MAPfold_vp.append(hMAPfold_vp/float(kFoldSize))
		GLOBAL_asseY_MAPfold_vn.append(hMAPfold_vn/float(kFoldSize))

		GLOBAL_asseY_MLfold_fp.append(MLfold_fp/float(kFoldSize))
		GLOBAL_asseY_MLfold_fn.append(MLfold_fn/float(kFoldSize))
		GLOBAL_asseY_MLfold_vp.append(MLfold_vp/float(kFoldSize))
		GLOBAL_asseY_MLfold_vn.append(MLfold_vn/float(kFoldSize))
		
		GLOBAL_asseY_MLfold_Sensitivita.append(MLfold_sensitivita/float(kFoldSize))
		GLOBAL_asseY_MLfold_Specificita.append(MLfold_specificita/float(kFoldSize))
		GLOBAL_asseY_MAPfold_Sensitivita.append(hMAPfold_sensitivita/float(kFoldSize))
		GLOBAL_asseY_MAPfold_Specificita.append(hMAPfold_specificita/float(kFoldSize))	
	
		print("HMAP - k-folding accuracy avarage: {0} %".format(hMAPfold/float(kFoldSize)))
		
		print("HMAP - fold falsi positivi: {0} %".format(hMAPfold_fp/float(len(splitRatioS))))
		print("HMAP - fold falsi negativi: {0} %".format(hMAPfold_fn/float(len(splitRatioS))))
		print("HMAP - fold veri positivi: {0} %".format(hMAPfold_vp/float(len(splitRatioS))))
		print("HMAP - fold veri negativi: {0} %".format(hMAPfold_vn/float(len(splitRatioS))))
		print("ML - fold falsi positivi: {0} %".format(MLfold_fp/float(len(splitRatioS))))
		print("ML - fold falsi negativi: {0} %".format(MLfold_fn/float(len(splitRatioS))))
		print("ML - fold veri positivi: {0} %".format(MLfold_vp/float(len(splitRatioS))))
		print("ML - fold veri negativi: {0} %".format(MLfold_vn/float(len(splitRatioS))))
		
		print("ML   - k-folding accuracy avarage: {0} %".format(MLfold/float(kFoldSize)))
		#print("NAIVE- k-folding avarage: {0} %".format(NAIVEfold/float(kFoldSize)))

	processSVM(svmPoly, GLOBAL_SVM_poly_data, 'poly', start_time)
	processSVM(svmRbf, GLOBAL_SVM_rbf_data, 'rbf ', start_time)
	processSVM(svmLinear, GLOBAL_SVM_lin_data, 'lin ', start_time)
	processSVM(svmSigmoid, GLOBAL_SVM_sigm_data, 'sigm', start_time)

class SVMGraphData:
	def __init__(self):
		self.asseY = []
		self.asseY_fp = []
		self.asseY_fn = []
		self.asseY_vp = []
		self.asseY_vn = []
		self.asseY_Sensitivita = []
		self.asseY_Specificita = []

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

global GLOBAL_SVM_poly_data
global GLOBAL_SVM_rbf_data
global GLOBAL_SVM_lin_data
global GLOBAL_SVM_sigm_data
GLOBAL_SVM_poly_data = SVMGraphData()
GLOBAL_SVM_rbf_data = SVMGraphData()
GLOBAL_SVM_lin_data = SVMGraphData()
GLOBAL_SVM_sigm_data = SVMGraphData()

global GLOBAL_asseY_MAPsub_fp
GLOBAL_asseY_MAPsub_fp =[]

global GLOBAL_asseY_MAPsub_fn
GLOBAL_asseY_MAPsub_fn =[]

global GLOBAL_asseY_MAPsub_vp
GLOBAL_asseY_MAPsub_vp =[]

global GLOBAL_asseY_MAPsub_vn
GLOBAL_asseY_MAPsub_vn =[]

global GLOBAL_asseY_MLsub_fp
GLOBAL_asseY_MLsub_fp =[]

global GLOBAL_asseY_MLsub_fn
GLOBAL_asseY_MLsub_fn =[]

global GLOBAL_asseY_MLsub_vp
GLOBAL_asseY_MLsub_vp =[]

global GLOBAL_asseY_MLsub_vn
GLOBAL_asseY_MLsub_vn =[]

global GLOBAL_asseY_MLfold_fp
GLOBAL_asseY_MLfold_fp =[]

global GLOBAL_asseY_MLfold_fn
GLOBAL_asseY_MLfold_fn =[]

global GLOBAL_asseY_MLfold_vp
GLOBAL_asseY_MLfold_vp =[]

global GLOBAL_asseY_MLfold_vn
GLOBAL_asseY_MLfold_vn =[]

global GLOBAL_asseY_MAPfold_fp
GLOBAL_asseY_MAPfold_fp =[]

global GLOBAL_asseY_MAPfold_fn
GLOBAL_asseY_MAPfold_fn =[]

global GLOBAL_asseY_MAPfold_vp
GLOBAL_asseY_MAPfold_vp =[]

global GLOBAL_asseY_MAPfold_vn
GLOBAL_asseY_MAPfold_vn =[]


global GLOBAL_asseY_MAPfold_Sensitivita
GLOBAL_asseY_MAPfold_Sensitivita =[]


global GLOBAL_asseY_MAPfold_Specificita
GLOBAL_asseY_MAPfold_Specificita =[]


global GLOBAL_asseY_MAPsub_Sensitivita
GLOBAL_asseY_MAPsub_Sensitivita =[]


global GLOBAL_asseY_MAPsub_Specificita
GLOBAL_asseY_MAPsub_Specificita =[]

global GLOBAL_asseY_MLfold_Sensitivita
GLOBAL_asseY_MLfold_Sensitivita =[]

global GLOBAL_asseY_MLfold_Specificita
GLOBAL_asseY_MLfold_Specificita =[]

global GLOBAL_asseY_MLsub_Sensitivita
GLOBAL_asseY_MLsub_Sensitivita =[]

global GLOBAL_asseY_MLsub_Specificita
GLOBAL_asseY_MLsub_Specificita =[]

## Parameters array

# number of tests! (30)
number_of_test = 10 # 30 di default
sizeColumnsToKeepArray = list(range(1, number_of_test + 1))
for x in range(0, number_of_test):
	main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=sizeColumnsToKeepArray[x],typeOfFeatureSelection="RecursiveFeatureElimination")

#main(filename='phi.arff',doShuffle=True,splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69],kFoldSize=10,sizeColumnsToKeep=30,typeOfFeatureSelection="RecursiveFeatureElimination")

fig1 = plt.figure(1)
fig1.suptitle('Accuracy', fontsize=20)
plt.plot(GLOBAL_asseX, GLOBAL_SVM_poly_data.asseY)
plt.plot(GLOBAL_asseX, GLOBAL_SVM_rbf_data.asseY)
plt.plot(GLOBAL_asseX, GLOBAL_SVM_lin_data.asseY)
plt.plot(GLOBAL_asseX, GLOBAL_SVM_sigm_data.asseY)
plt.plot(GLOBAL_asseX, GLOBAL_asseY_MAP_SUB)
plt.plot(GLOBAL_asseX, GLOBAL_asseY_ML_SUB)
plt.plot(GLOBAL_asseX, GLOBAL_asseY_MAP_KFOL)
plt.plot(GLOBAL_asseX, GLOBAL_asseY_ML_KFOL)
plt.legend(['SVM Polynomial', 'SVM Rbf', 'SVM Linear', 'SVM Sigmoid', 'HMAP Subsampling', 'ML Subsampling', 'HMAP Cross Validation','ML Cross Validation'], loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylabel('Accuracy %')
plt.xlabel('Number of features')
# set asse x interval ( 1 step )
plt.xticks(range(0, int(max(GLOBAL_asseX))+1, 2))
plt.yticks(range(0, 98, 2))
plt.subplots_adjust(left=None, bottom=None, right=0.75, top=None, wspace=None, hspace=None)

plt.savefig("Accuracy",bbox_inches='tight', pad_inches=0)

# TODO: da qui in poi si fa' riferimento alla SVM lineare
#       quando si richiede una SVM

fig2 = plt.figure(2)
fig2.suptitle('Subsampling - Error Type', fontsize=20)
plt.subplot(3,1,1)
plt.ylabel('%')
plt.xlabel('Number of features')
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAPsub_fp)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAPsub_fn)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAPsub_vp)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAPsub_vn)
plt.legend(['HMAP - Falsi positivi', 'HMAP - Falsi Negativi', 'HMAP - Veri positivi', 'HMAP - Veri negativi'], loc='center left', bbox_to_anchor=(1,0.5))
plt.subplots_adjust(left=None, bottom=None, right=0.75, top=None, wspace=None, hspace=None)


plt.subplot(3,1,2)
plt.ylabel('%')
plt.xlabel('Number of features')
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MLsub_fp)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MLsub_fn)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MLsub_vp)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MLsub_vn)
plt.legend(['ML - Falsi positivi', 'ML - Falsi Negativi', 'ML - Veri positivi', 'ML - Veri negativi'], loc='center left', bbox_to_anchor=(1,0.5))
plt.savefig("Subsambpling - ErrorType",bbox_inches='tight', pad_inches=0)

fig3=plt.figure(3)
fig3.suptitle('Cross Validation - Error Type', fontsize=20)
plt.subplot(3,1,1)
plt.ylabel('%')
plt.xlabel('Number of features')
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAPfold_fp)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAPfold_fn)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAPfold_vp)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAPfold_vn)
plt.legend(['HMAP - Falsi positivi', 'HMAP - Falsi Negativi', 'HMAP - Veri positivi', 'HMAP - Veri negativi'], loc='center left', bbox_to_anchor=(1,0.5))
plt.subplots_adjust(left=None, bottom=None, right=0.75, top=None, wspace=None, hspace=None)

plt.subplot(3,1,2)
plt.ylabel('%')
plt.xlabel('Number of features')
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MLfold_fp)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MLfold_fn)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MLfold_vp)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MLfold_vn)
plt.legend(['ML - Falsi positivi', 'ML - Falsi Negativi', 'ML - Veri positivi', 'ML - Veri negativi'], loc='center left', bbox_to_anchor=(1,0.5))
plt.subplots_adjust(left=None, bottom=None, right=0.75, top=None, wspace=None, hspace=None)

plt.savefig("Cross Validation - ErrorType",bbox_inches='tight', pad_inches=0)

fig4 = plt.figure(4)
fig4.suptitle('SVM - Error Type', fontsize=20)
plt.plot(GLOBAL_asseX, GLOBAL_SVM_lin_data.asseY_fp)
plt.plot(GLOBAL_asseX, GLOBAL_SVM_lin_data.asseY_fn)
plt.plot(GLOBAL_asseX, GLOBAL_SVM_lin_data.asseY_vp)
plt.plot(GLOBAL_asseX, GLOBAL_SVM_lin_data.asseY_vn)
plt.legend(['Falsi positivi', 'Falsi Negativi', 'Veri positivi', 'Veri negativi'], loc=9, bbox_to_anchor=(0.5, -0.1))
plt.ylabel('%')
plt.xlabel('Number of features')
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
plt.savefig("SVM - ErrorType",bbox_inches='tight', pad_inches=0)


fig5 = plt.figure(5)
fig5.suptitle('Sensitivity - Cross Validation', fontsize=20)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAPfold_Sensitivita)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MLfold_Sensitivita)
plt.plot(GLOBAL_asseX, GLOBAL_SVM_lin_data.asseY_Sensitivita)
plt.legend(['HMAP','ML','SVM'], loc=9, bbox_to_anchor=(0.5, -0.1))
plt.ylabel('%')
plt.xlabel('Number of features')
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
plt.savefig("Sensitivity CrossValidation")

fig6 = plt.figure(6)
fig6.suptitle('Sensitivity - Subsampling', fontsize=20)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAPsub_Sensitivita)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MLsub_Sensitivita)
plt.plot(GLOBAL_asseX, GLOBAL_SVM_lin_data.asseY_Sensitivita)
plt.legend(['HMAP','ML','SVM'], loc=9, bbox_to_anchor=(0.5, -0.1))
plt.ylabel('%')
plt.xlabel('Number of features')
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
plt.savefig("Sensitivity Subsampling",bbox_inches='tight', pad_inches=0)

fig7 = plt.figure(7)
fig7.suptitle('Specificity - Subsampling', fontsize=20)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAPsub_Specificita)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MLsub_Specificita)
plt.plot(GLOBAL_asseX, GLOBAL_SVM_lin_data.asseY_Specificita)
plt.legend(['HMAP','ML','SVM'], loc=9, bbox_to_anchor=(0.5, -0.1))
plt.ylabel('%')
plt.xlabel('Number of features')
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
plt.savefig("Specificity Subsampling",bbox_inches='tight', pad_inches=0)

fig8 = plt.figure(8)
fig8.suptitle('Specificity - Cross Validation', fontsize=20)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MAPfold_Specificita)
plt.plot(GLOBAL_asseX,GLOBAL_asseY_MLfold_Specificita)
plt.plot(GLOBAL_asseX, GLOBAL_SVM_lin_data.asseY_Specificita)
plt.legend(['HMAP','ML','SVM'], loc=9, bbox_to_anchor=(0.5, -0.1))
plt.ylabel('%')
plt.xlabel('Number of features')
plt.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
plt.savefig("Specificity CrossValidation",bbox_inches='tight', pad_inches=0)

plt.show()

# run: ipython3 BClassifier.py -pylab
if (input('Close all windows? [S/N]') == "S"):
	# close all the Figure Windows
	plt.close('all')
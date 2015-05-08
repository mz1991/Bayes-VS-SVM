# Example of Naive Bayes implemented from Scratch in Python
import csv
import random
import math
import itertools
import random

def loadCsv(filename):
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def subSamplingSplit(dataset,splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def kFoldSplit(dataset,nFold):
	return [list(b) for b in zip(*[iter(dataset)]*(len(dataset)//nFold))]

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if (vector[-1] not in separated):
			separated[vector[-1]] = [] # -1 l'ultimo attributo e quello su cui voglio fare previsione
		separated[vector[-1]].append(vector)
	return separated

def mean(numbers):
	#return sum(numbers)/float(len(numbers))
	return sum([float(b) for b in numbers])/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	#variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	toInt=([float(b) for b in numbers])
	variance = sum([pow(x-avg,2) for x in toInt])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.items():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	#exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	exponent = math.exp(-(math.pow(float(x)-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getAccuracy(testSet, predictions):
	correct = 0 # TODO: falsi positivi (ipotesi valida ma rifiutata) and falso negativo (ipotesi sbagliata ma accettata)
	falsiPositivi = 0
	falsiNegativi = 0
	veriPositivi=0 # ipotesi valida e accettata
	veriNegativi=0 # ipotesi non valida e rifiutata
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
			if predictions[i] == "0":
				veriNegativi+=1
			else:
				veriPositivi+=1
		else:
			if testSet[i][-1]=="1" and predictions[i]=="0":
				falsiPositivi+=1
			elif testSet[i][-1]=="0" and predictions[i]=="1":
				falsiNegativi+=1

	falsiPositivi/len(testSet) * 100

	return (correct/float(len(testSet))) * 100.0 ,falsiPositivi,falsiNegativi,veriPositivi,veriNegativi

def bayesClassification(dataset,trainingSet,testSet):
	
	print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	accuracy,falsiPositivi,falsiNegativi,veriPositivi,veriNegativi = getAccuracy(testSet, predictions)
	print('Accuracy: {0}%'.format(accuracy))
	print('Falsi Positivi: {0}'.format(falsiPositivi))
	print('Falsi Negativi: {0}'.format(falsiNegativi))
	print('Veri Positivi: {0}'.format(veriPositivi))
	print('Veri Negativi: {0}'.format(veriNegativi))
	#sensitivita: veripositivi / (veripositivi + falsi negativi)
	#specificita:  verinegativi / (falsi positivi + veri negativi)
	print('Sensitivita: {0}'.format(veriPositivi/float(veriPositivi+falsiNegativi)))
	print('Specificita: {0}'.format(veriNegativi/float(falsiPositivi+veriNegativi)))
	return accuracy


def main():
	filename="bigDataset.csv"
	kFoldSize=10
	doShuffle = True
	
	lines = csv.reader(open(filename, "r"))
	dataset = list(lines)
	if doShuffle:
		random.shuffle(dataset)
	

	# -- Start Sub Samplig Cross Validation
	totAccuracySubSampling=0
	splitRatioS=[0.50,0.60,0.70,0.80,0.75,0.85,0.90,0.65,0.77,0.69]
	for index,splitRatio in enumerate(splitRatioS):
		print("Index {0} split ration {1}".format(index,splitRatio))
		trainingSet, testSet = subSamplingSplit(dataset,splitRatio)
		totAccuracySubSampling+=bayesClassification(dataset,trainingSet,testSet)
		print("--------------")
	
	totAccuracyKFold=0
	folds=kFoldSplit(dataset,kFoldSize)	
	for index,fold in enumerate(folds):
		print("Index {0} split K-Fold {1}".format(index,kFoldSize))
		testSet = fold
		copy = folds[:]
		del copy[index]
		trainingSet = list(itertools.chain(*copy))
		totAccuracyKFold+=bayesClassification(dataset,trainingSet,testSet)
		print("--------------")

	print("*************************************")
	print("Accuracy Mean with subSampling {0}%".format(totAccuracySubSampling/float(len(splitRatioS))))
	print("*************************************")

	print("*************************************")
	print("Accuracy Mean With K-Fold {0}%".format(totAccuracyKFold/float(kFoldSize)))
	print("*************************************")
main()

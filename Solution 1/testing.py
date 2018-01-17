from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from utilities import *
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

with open('stopWords.txt') as f:
		stop_words = f.readlines()
for i in range(len(stop_words)):
	stop_words[i] = stop_words[i].replace('\\n','')

data = fetchDataFromExcel(0)
testingData = formatTestingSet(data[0:100], stop_words)

# print("LogisticRegression_classifier : ")
# LogisticRegression_classifier = joblib.load('LogisticRegression_classifier.pkl') 
# LogisticRegression_classifier_predict = LogisticRegression_classifier.predict(testingData[0])
# print("		Prediction : "+ str(LogisticRegression_classifier_predict))
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# print()

# print("SGDClassifier_classifier : ")
# SGDClassifier_classifier = joblib.load('SGDClassifier_classifier.pkl') 
# SGDClassifier_classifier_predict = SGDClassifier_classifier.predict(testingData[0])
# print("		Prediction : "+ str(SGDClassifier_classifier_predict))
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# print()

# print("LinearSVC_classifier : ")
# LinearSVC_classifier = joblib.load('LinearSVC_classifier.pkl') 
# LinearSVC_classifier_predict = LinearSVC_classifier.predict(testingData[0])
# print("		Prediction : "+ str(LinearSVC_classifier_predict))
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# print()

print("SVC_classifier : ")
SVC_classifier = joblib.load('SVC_classifier.pkl') 
SVC_classifier_predict = SVC_classifier.predict(testingData[0])
print("		Prediction : "+ str(SVC_classifier_predict))
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print()

finalPrediction = []
for i in range(len(testingData[0])):
	rightOrWrong = 'WRONG'
	if (SVC_classifier_predict[i] == testingData[1][i]):
		rightOrWrong = 'RIGHT'
	finalPrediction.append({'ID': i, 'PREDICTED CATEGORY': SVC_classifier_predict[i], 'ACTUAL CATEGORY': testingData[1][i], 'RESULT': rightOrWrong })

print()
print()
print("FINAL PREDICTION:-")
print()
for i in finalPrediction:
	print(i)
print()
print()
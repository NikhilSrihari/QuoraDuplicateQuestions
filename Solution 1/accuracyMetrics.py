from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from utilities import *
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.externals import joblib

with open('stopWords.txt') as f:
		stop_words = f.readlines()
for i in range(len(stop_words)):
	stop_words[i] = stop_words[i].replace('\\n','')

data = fetchDataFromExcel(1)
testingSet = formatTestingSet(data[220000:230000], stop_words)

print("SVC_classifier : ")
SVC_classifier = joblib.load('SVC_classifier.pkl') 
SVC_classifier_Result = SVC_classifier.predict(testingSet[0])
SVC_classifier_Accuracy = accuracy_score(testingSet[1], SVC_classifier_Result) * 100
SVC_classifier_Precision = precision_score(testingSet[1], SVC_classifier_Result) * 100
SVC_classifier_Recall = recall_score(testingSet[1], SVC_classifier_Result) * 100
print("		Accuracy % : "+ str(SVC_classifier_Accuracy))
print("		Precision % : "+ str(SVC_classifier_Precision))
print("		Recall % : "+ str(SVC_classifier_Recall))
print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print()

# print("LogisticRegression_classifier : ")
# LogisticRegression_classifier = joblib.load('LogisticRegression_classifier.pkl') 
# LogisticRegression_classifier_Result = LogisticRegression_classifier.predict(testingSet[0])
# LogisticRegression_classifier_Accuracy = accuracy_score(testingSet[1], LogisticRegression_classifier_Result) * 100
# LogisticRegression_classifier_Precision = precision_score(testingSet[1], LogisticRegression_classifier_Result) * 100
# LogisticRegression_classifier_Recall = recall_score(testingSet[1], LogisticRegression_classifier_Result) * 100
# print("		Accuracy % : "+ str(LogisticRegression_classifier_Accuracy))
# print("		Precision % : "+ str(LogisticRegression_classifier_Precision))
# print("		Recall % : "+ str(LogisticRegression_classifier_Recall))
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# print()

# print("SGDClassifier_classifier : ")
# SGDClassifier_classifier = joblib.load('SGDClassifier_classifier.pkl') 
# SGDClassifier_classifier_Result = SGDClassifier_classifier.predict(testingSet[0])
# SGDClassifier_classifier_Accuracy = accuracy_score(testingSet[1], SGDClassifier_classifier_Result) * 100
# SGDClassifier_classifier_Precision = precision_score(testingSet[1], SGDClassifier_classifier_Result) * 100
# SGDClassifier_classifier_Recall = recall_score(testingSet[1], SGDClassifier_classifier_Result) * 100
# print("		Accuracy % : "+ str(SGDClassifier_classifier_Accuracy))
# print("		Precision % : "+ str(SGDClassifier_classifier_Precision))
# print("		Recall % : "+ str(SGDClassifier_classifier_Recall))
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# print()

# print("LinearSVC_classifier : ")
# LinearSVC_classifier = joblib.load('LinearSVC_classifier.pkl') 
# LinearSVC_classifier_Result = LinearSVC_classifier.predict(testingSet[0])
# LinearSVC_classifier_Accuracy = accuracy_score(testingSet[1], LinearSVC_classifier_Result) * 100
# LinearSVC_classifier_Precision = precision_score(testingSet[1], LinearSVC_classifier_Result) * 100
# LinearSVC_classifier_Recall = recall_score(testingSet[1], LinearSVC_classifier_Result) * 100
# print("		Accuracy % : "+ str(LinearSVC_classifier_Accuracy))
# print("		Precision % : "+ str(LinearSVC_classifier_Precision))
# print("		Recall % : "+ str(LinearSVC_classifier_Recall))
# print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
# print()
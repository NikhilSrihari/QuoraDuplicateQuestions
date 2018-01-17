from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from utilities import *
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.externals import joblib

with open('stopWords.txt') as f:
		stop_words = f.readlines()
for i in range(len(stop_words)):
	stop_words[i] = stop_words[i].replace('\\n','')

data = fetchDataFromExcel(0)
trainingSet = data[0:100000]

ps = PorterStemmer()
trainingInputData=[]
trainingTargetData=[]
for  i in trainingSet:
	w1 = preProcessing(i[3], stop_words)
	w2 = preProcessing(i[4], stop_words)

	uni1=my_unigram(w1)
	bi1=my_bigram(w1)
	feature1 = calcSentProb_Unigram(w2, uni1)
	feature3 = calcSentProb_Bigram(w2, bi1)

	uni2=my_unigram(w2)
	bi2=my_bigram(w2)
	feature2 = calcSentProb_Unigram(w1, uni2)
	feature4 = calcSentProb_Bigram(w1, bi2)

	trainingInputData.append([feature1, feature2, feature3, feature4])
	#trainingInputData.append([feature1, feature2])
	trainingTargetData.append(i[5]) 

# # LogisticRegression, SGDClassifier
# LogisticRegression_classifier = LogisticRegression()
# LogisticRegression_classifier.fit(trainingInputData, trainingTargetData)
# print("LogisticRegression_classifier Learning Complete ")
# joblib.dump(LogisticRegression_classifier, 'LogisticRegression_classifier.pkl') 

# SGDClassifier_classifier = SGDClassifier()
# SGDClassifier_classifier.fit(trainingInputData, trainingTargetData)
# print("SGDClassifier_classifier Learning Complete ")
# joblib.dump(SGDClassifier_classifier, 'SGDClassifier_classifier.pkl') 

# # SVC, LinearSVC
# LinearSVC_classifier = LinearSVC()
# LinearSVC_classifier.fit(trainingInputData, trainingTargetData)
# print("LinearSVC_classifier Learning Complete ")
# joblib.dump(LinearSVC_classifier, 'LinearSVC_classifier.pkl') 

SVC_classifier = SVC()
SVC_classifier.fit(trainingInputData, trainingTargetData)
print("SVC_classifier Learning Complete ")
joblib.dump(SVC_classifier, 'SVC_classifier.pkl')
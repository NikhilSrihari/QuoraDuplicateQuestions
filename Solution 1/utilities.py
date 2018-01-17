from nltk.probability import FreqDist
from nltk.util import ngrams
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import xlrd
import random
import numpy
from nltk.corpus import stopwords


def fetchDataFromExcel(shuffle):
	path = 'QuoraDataset.xlsx'
	workbook = xlrd.open_workbook(path)
	worksheet = workbook.sheet_by_index(0)
	data = []
	offset = 0
	for i, row in enumerate(range(worksheet.nrows)):
		if i <= offset:  # (Optionally) skip headers
			continue
		r = []
		for j, col in enumerate(range(worksheet.ncols)):
			r.append(worksheet.cell_value(i, j))
		data.append(r)
	if (shuffle==1):
		random.shuffle(data)
	return data;


def my_unigram(words):
	freq = FreqDist(words)
	freqDistr = freq.most_common(50)
	for i in range(len(freqDistr)):
		freqDistr[i]=list(freqDistr[i])
		freqDistr[i][1]=freqDistr[i][1]+1
	
	rowSum = 0		
	for i in range(len(freqDistr)):
		rowSum = rowSum + freqDistr[i][1]
	
	uni = []
	for i in range(len(freqDistr)):
		uni.append([freqDistr[i][0], (-1 * numpy.log(freqDistr[i][1]/rowSum))])
	# print(uni)
	return uni


def my_bigram(words):
	uniFreq = FreqDist(words)
	uniFreqDistr = uniFreq.most_common(50)
	for i in range(len(uniFreqDistr)):
		uniFreqDistr[i]=list(uniFreqDistr[i])
		uniFreqDistr[i][1]=uniFreqDistr[i][1]+1
	
	bigrams=list(ngrams(words,2))
	for i in range(len(bigrams)):
		bigrams[i]=bigrams[i][0]+' '+bigrams[i][1]
	
	biFreq = FreqDist(bigrams)
	biFreqDistr = biFreq.most_common(50)
	for i in range(len(biFreqDistr)):
		biFreqDistr[i]=list(biFreqDistr[i])
		biFreqDistr[i][1]=biFreqDistr[i][1]+1

	bi=[]
	for i in range(len(biFreqDistr)):
		tempWd = ((biFreqDistr[i][0]).split(' ', 1)[0]).strip()
		# print(tempWd)
		for j in range(len(uniFreqDistr)):
			if (uniFreqDistr[j][0].strip() == tempWd):
				unigramCnt=uniFreqDistr[j][1]

		bi.append([biFreqDistr[i][0], (-1 * numpy.log(biFreqDistr[i][1]/unigramCnt))]) 
	# print(bi)
	return bi


def calcSentProb_Unigram(words, uni):
	cnt = 0
	for i in range(len(words)):
		for j in range(len(uni)):
			if (words[i].strip()==uni[j][0].strip()):
				cnt = cnt + uni[j][1]
				break
	if (len(words)!=0):
		return (cnt/len(words))
	else:
		return 0


def calcSentProb_Bigram(words, bi):
	bigrams=list(ngrams(words,2))
	for i in range(len(bigrams)):
		bigrams[i]=bigrams[i][0]+' '+bigrams[i][1]

	cnt = 0
	for i in range(len(bigrams)):
		for j in range(len(bi)):
			if (bigrams[i].strip()==bi[j][0].strip()):
				cnt = cnt + bi[j][1]
	if (len(bigrams)!=0):
		return (cnt/len(bigrams))
	else:
		return 0


def formatTestingSet(testingSet, stop_words):
	ps = PorterStemmer()
	testingInputData=[]
	testingExpectedOutputData=[]
	for i in testingSet:
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

		testingInputData.append([feature1, feature2, feature3, feature4])
		#testingInputData.append([feature1, feature2])
		testingExpectedOutputData.append(i[5])

	return [testingInputData, testingExpectedOutputData]


def preProcessing(sent, stop_words):
	sent = sent.replace("(", " ").replace(")", " ").replace("?", " ").replace(",", " ").replace("'", "")
	for i in range(len(stop_words)):
		sent = sent.replace(stop_words[i], "")
	words = word_tokenize(sent)
	ps = PorterStemmer()
	for j in range(len(words)):
		words[j]=ps.stem(words[j])
	return words

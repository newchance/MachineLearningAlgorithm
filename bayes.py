#coding:utf-8
#=========================
#朴素贝叶斯分类算法
#author： zhang haibo
#time：2013-7-12
#=========================

from numpy import *


#词表到向量的转换函数
def loadDataSet():
    postingList = [ ['my', 'dog', 'has', 'flea', 'problems','help','please'],
                    ['maybe','not','take','him','to','dog','park','stupid'],
                    ['my','dalmation','is','so','cute','I','love','him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr','licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
                   ]
    classVec = [0,1,0,1,0,1] #1代表侮辱性文字， 0代表正常言论
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

#词集模型：每一个词是否出现，且每一个词只能出现一次
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)#全部设置成0的列表
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return returnVec

#词袋模型:每一个词在文档中不止出现一次
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] *len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


#朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords) ; p1Num = ones(numWords)#避免概率值为0
    p0Denom = 2.0 ; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
            
    p1Vect = log(p1Num/p1Denom)#避免向下溢出
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

#朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0 :
        return 1
    else:
        return 0

#测试朴素贝叶斯分类函数
def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as:', classifyNB(thisDoc, p0V,p1V,pAb)
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print testEntry, 'classified as:', classifyNB(thisDoc,p0V,p1V,pAb)

#文本解析
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

#垃圾邮件检测函数
def spamTest():
    docList = []; classList = []; fullText = []
    #输入
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    #划分训练集以及测试集
    trainingSet = range(50); testSet=[]
    for i in range(10) :
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    #训练
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    #测试
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print 'the error rate is : ', float(errorCount)/len(testSet)   
        
#==============测试代码======================
listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
print myVocabList
print setOfWords2Vec(myVocabList, listOPosts[0])
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
p0V,p1V,pAb = trainNB0(trainMat,listClasses) 
print p0V, p1V, pAb

testingNB()

spamTest()


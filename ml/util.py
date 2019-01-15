"""
Copyright 2017 Baris Akgun (baakgun@ku.edu.tr)

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this 
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this 
list of conditions and the following disclaimer in the documentation and/or other 
materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may 
be used to endorse or promote products derived from this software without specific 
prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT 
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.

This software is intended for educational purposes only. 
"""

import inspect
import numpy as np
import math
import sys
import plotting

def raiseNotDefined():
  fileName = inspect.stack()[1][1]
  line = inspect.stack()[1][2]
  method = inspect.stack()[1][3]

  print "*** Method not implemented: %s at line %s of %s" % (method, line, fileName)
  sys.exit(1)

def most_common(lst):
  """ Returns the most common element of the input container """
  return max(set(lst), key=lst.count)


def getRandomSplit(Data, Labels, ratio = 0.8): 
  """ 
  Randomly splits the data into training set and a testing set. 
  Ratio is the number of training examples over the entire dataset.

  This is provided here as a guidance.
  """
  n,d = Data.shape
  indices = np.arange(0,n)
  np.random.shuffle(indices)
  r = int(math.floor(len(indices)*ratio))
  trainIndices = indices[:r]
  testIndices = indices[r:]
  return getSplit(Data, Labels, trainIndices, testIndices)

def getCrossValSplit(Data, Labels, k): 
  """ 
  Creates a generator that returns the desired cross-validation split each time it is called.

  You should understand how this works.
  """
  n,d = Data.shape
  indices = np.arange(0,n)
  np.random.shuffle(indices)
  ratio = 1./k
  r = int(math.floor(n*ratio))

  for i in range(k):
    if(i == k-1):
      testIndices = indices[i*r:]
    else:
      testIndices = indices[i*r:(i+1)*r]
    trainIndices = np.setdiff1d(indices,testIndices)
    yield getSplit(Data, Labels, trainIndices, testIndices)

def getSplit(Data, Labels, trainIndices, testIndices):
  trainData = Data[trainIndices,:]
  trainLabels = Labels[trainIndices]
  testData = Data[testIndices,:]
  testLabels = Labels[testIndices]   
  return trainData, trainLabels, testData, testLabels 

def calculateClassificationError(predictedLabels, origLabels): 
  """
  Returns the percentage of wrongly predicted labels
  """
  error = 0.;
  for i in range(0,len(origLabels)):
    if(predictedLabels[i] != origLabels[i]):
      error += 1.
  error /= len(origLabels)
  return error

def calculateMeanL2Error(predictedLabels, origLabels): 
  """
  Returns the mean L2 error
  """
  return np.linalg.norm(predictedLabels-origLabels)/len(origLabels)

def testMultiParameters(testData, testLabels, ClassifierClass, parameterSet, trainData, trainLabels, errorFunc = calculateClassificationError, isError = True):
  """ parameterSet = set of parameters to try """

  results = []
  for parameters in parameterSet:
    classifier = ClassifierClass()
    classifier.setParams(parameters) 
    classifier.fit(trainData, trainLabels)
    testPredictions = classifier.predict(testData)
    e = errorFunc(testPredictions, testLabels)
    if not isError: #only makes sense for classification!
      e = 1-e
    results.append(e)
  if isError:
    best = np.min(results)
    selectedParams = parameterSet[np.argmin(results)]        
  else:
    best = np.max(results)
    selectedParams = parameterSet[np.argmax(results)]    
  return selectedParams, best, results 

def crossValidation(Data, Labels, ClassifierClass, paramSet, errorFunction, numCrossVal = 5):
  kcv = getCrossValSplit(Data, Labels, numCrossVal)  
  x = 0
  for  trainData, trainLabels, testData, testLabels in kcv:
    k,a,res = testMultiParameters(testData, testLabels, ClassifierClass, paramSet, trainData, trainLabels, errorFunc = errorFunction)
    kT,aT,resT = testMultiParameters(trainData, trainLabels, ClassifierClass, paramSet, trainData, trainLabels, errorFunc = errorFunction)    
    if x == 0:
      allRes  = np.array([res]).transpose()
      allResT = np.array([resT]).transpose()
    else:
      allRes = np.hstack((allRes,np.array([res]).transpose()))
      allResT = np.hstack((allResT,np.array([resT]).transpose())) 
    x += 1

  return allRes, allResT

def getParamAndRes(Data, Labels, ClassifierClass, paramSet, errorFunction, numCrossVal, extraString='', logTick = False, save=True):
  testRes, trainRes = crossValidation(Data, Labels, ClassifierClass, paramSet, errorFunction, numCrossVal)
  plotting.plotVsParam(paramSet, trainRes, testRes, title=ClassifierClass.name+extraString, fName=ClassifierClass.name+extraString+".png", logTick=logTick,save=save) 
  meanRes = np.mean(testRes,axis=1)
  stdsRes = np.std(testRes,axis=1)
  res = np.min(meanRes)
  param = paramSet[np.argmin(meanRes)]
  stdRes = stdsRes[np.argmin(meanRes)]
  return param, res, stdRes
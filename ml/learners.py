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

import numpy as np
import data
import util
from sklearn import linear_model
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from collections import Counter


class Learner(object):
  name = None

  def setParams(self):
    util.raiseNotDefined()

  def fit(self, trainingFeatures, trainingTargets):
    util.raiseNotDefined()

  def predict(self, testFeatures):
    util.raiseNotDefined()

#Classifiers
class knnClassifier(Learner):
  """
  WARNING: 
  If you use scikit-learn implementation here, you will not receive any credits.
  Only numpy methods are allowed in this class.
  """  
  
  name = "knn"
  def __init__(self, numNeighbors = 1):
    self.k = 1
    self.setParams(numNeighbors)

    """ Add anything else you want to initialize below"""
    self.data = None
    self.labels = None

    self._fitCalled = False

  def setParams(self,paramSet):
    if paramSet > 0:
      self.k = paramSet

  def fit(self, trainingFeatures, trainingTargets):
    """ 
    Method that trains the model. 
    Naming is chosen to mimick scikit-learn.

    Hint: This is trivial for kNN


    trainingFeatures: n x d 2D numpy array 
    trainingLabels: n dimensional 1D numpy array

    where d is the data dimension and n is the number of training points 
    """

    """ Implement kNN learning below. """

    self._fitCalled = True
    self.labels = trainingTargets
    self.data = trainingFeatures


  def predict(self, testFeatures):
    """ 
    Method that returns the predicted labels of the input features. 
    This is the main part of the kNN algorithm!

    testFeatures: l x d 2D numpy array 

    where d is the data dimension and l is the number of points to make predictions about

    returns an l dimensional 1D numpy array composed of the predictions
    """    

    if(not self._fitCalled):
      print('The fit method has not been called yet')
      return None

    l,d = testFeatures.shape
    n,d = self.data.shape    

    """ Fill and return this in your implementation. """
    predictions = np.empty(shape=(l,), dtype=self.labels.dtype)

    """ Implement kNN prediction here. """

    for i in range(0, l):
      distances = []
      for j in range(0, n):
        distances.append((np.sqrt(np.sum((testFeatures[i]-self.data[j])**2)), self.labels[j]))
      distances.sort()
      kNearestLabels = [x[1] for x in distances][0:self.k]
      most_common, num_most_common = Counter(kNearestLabels).most_common(1)[0]
      predictions[i] = most_common
    return predictions



class LogisticRegressionClassifier(Learner):
  """
  This class is a wrapper for the scikit-learn logistic regression implementation.
  No need to do anything in this class.
  """

  name = "logreg"
  def __init__(self, regularizer=100):    
    self.logistic = linear_model.LogisticRegression(C=regularizer) 
    self._fitCalled = False

  def setParams(self, paramSet):
    self.logistic.C = paramSet

  def fit(self, trainingFeatures, trainingTargets):
    self._fitCalled = True
    return self.logistic.fit(trainingFeatures, trainingTargets)

  def predict(self, testFeatures):
    if(not self._fitCalled):
      print('The fit method has not been called yet')
      return None      
    return self.logistic.predict(testFeatures)

def add_ones(x):
    return np.hstack((np.ones((x.shape[0], 1)), x))

#Regression
class LinearRegression(Learner):
  """
  WARNING: 
  If you use scikit-learn implementation here, you will not receive any credits.
  Only numpy methods are allowed in this class.
  """
  
  name = 'linreg'
  def __init__(self, PreprocessorClass = data.Normalizer):    
    self.w = None
    self._fitCalled = False
    self.ppC = PreprocessorClass


  def setParams(self, paramSet):
    """ There are no parameters to be set for the vanilla regression so we silently pass """
    pass

  def fit(self, trainingFeatures, trainingTargets):  
    """ 
    Method to train the linear regression approach.

    trainingFeatures: n x d 2D numpy array 
    trainingLabels: n dimensional 1D numpy array

    where d is the data dimension and n is the number of training points 
    """

    self._fitCalled = True
    self.pp = self.ppC(trainingFeatures)
    preProcTrainingFeatures = self.pp.preProc(trainingFeatures)


    """ 
    Implement the linear regression learning below.

    Hint:  w = X\b
    where w is the weight vector to be learned, 
    X is the matrix that should be built from the data and the bias terms 
    and b is the trainingTarget vector
    \ operation corresponds to multiplying the pseudo-inverse of X with b (very matlab-like)

    Look at numpy linalg methods!

    The preprocessing call has been handled for you.
    """

    X = add_ones(preProcTrainingFeatures)
    X_pinv = np.linalg.pinv(X)
    b = trainingTargets
    W = X_pinv.dot(b)
    self.w = W


  def predict(self, testFeatures):
    """ 
    Method that calculates the predicted outputs given the input features.

    testFeatures: l x d 2D numpy array 

    where d is the data dimension and l is the number of points to make predictions about

    returns an l dimensional 1D numpy array composed of the predictions
    """

    if(not self._fitCalled):
      print('The fit method has not been called yet')
      return None     

    preProcTestFeatures = self.pp.preProc(testFeatures)
    """ 
    Implement the prediction method for the linear regression below.

    Hint:  X*w = b
    where w is the learned weight vector, 
    X is the matrix that should be built from the data and the bias terms 
    and b is the prediction

    The preprocessing call has been handled for you
    """
    w = self.w
    X= add_ones(preProcTestFeatures)
    b = X.dot(w)
    return b



class RidgeRegression(Learner):
  """
  This class is a wrapper for the scikit-learn ridge regression implementation 
  plus custom-preprocessing.

  No need to do anything in this class.
  """  

  name = "ridgereg"
  def __init__(self, PreprocessorClass = data.Normalizer, regularizer=1.):    
    self.ridge = linear_model.Ridge(alpha=regularizer) 
    self.ppC = PreprocessorClass
    self._fitCalled = False


  def setParams(self, paramSet):
    self.ridge.alpha = paramSet

  def fit(self, trainingFeatures, trainingTargets):
    self._fitCalled = True
    self.pp = self.ppC(trainingFeatures)
    preProcTrainingFeatures = self.pp.preProc(trainingFeatures)
    return self.ridge.fit(preProcTrainingFeatures, trainingTargets)

  def predict(self, testFeatures):
    if(not self._fitCalled):
      print('The fit method has not been called yet')
      return None      
    preProcTestFeatures = self.pp.preProc(testFeatures)
    return self.ridge.predict(preProcTestFeatures)

  
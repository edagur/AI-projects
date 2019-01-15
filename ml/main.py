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
import learners
import util
import data

if __name__ == "__main__":
  """
  In this homework, you are going to:
  - Extract normalized saturation histograms from images. See the SaturationHistogramExtractor class in data.py
  - Implement kNN learning and prediction. See the knnClassifier class in learners.py
  - Implement linear regression learning and prediction. See the LinearRegression class in learners.py
  
  This file is provided for you to test your implementations. You can:
  - Work on your features by only calling getDataForClassification(). When you are confident with your features, you can 
  set pickleEnabled to True to save and load the features efficiently.
  - Work on your kNN method by setting classification to True (default). If image features are be too hard to get started, create your own dummy data set and work on kNN.
  - Work on your linear regression method by setting regression to True. If your ridge regression slows you down, modify the regressors dictionary to temporarily remove it.
  
  When you run this script, it prints to standard output. In addition, it saves performance figures from the methods. You are going to need to interpret these figures for your report.
  You can set the saveImages variables to false to view the figures instead of saving themm
  
  We are going to use a different file and potentially different datasets to test your implementations.
  As long as you only change the part of the files that are designated, you should be fine.
  We are going to grade 3 things for the coding part:
  - Your image features (data.py)
  - Your kNN implmentation (learners.py)
  - Your linear regression implementation (learners.py)
  
  Many other things that are implemented for you. These include data loading and cross-validation.
  I suggest you look at all the methods related to cross-validation implementation (util.py).
  Python pickle is also a nice feature to save and load arbitrary data structures (data.py).
  
  """

  #Set this to True when you are sure your feature extraction is working correctly
  pickleEnabled = True  

  #use these flags to work on a single problem
  classification = True
  regression = True

  #Save figures as images.
  saveImages = True

  #DO NOT MODIFY BELOW THIS LINE OTHER THAN TO DEBUG!
  rSeed = 1690061903 #1690061903 #301086, 1618270666, 570696321
  numCrossVal = 5

  if classification:
    #Classification
    print("--- Classification ---")
    print

    AllData, AllLabels = data.getDataForClassification(pickleEnabled=pickleEnabled)

    np.random.seed(rSeed) 
    kRange =  range(1,22,2)     
    knnParam, knnRes, knnStd = util.getParamAndRes(AllData, AllLabels, learners.knnClassifier, kRange, util.calculateClassificationError, numCrossVal, save=saveImages)
    print "Selected k for kNN: ", knnParam
    print "kNN cross-validation average error for the selected k:" ,knnRes
    print "kNN cross-validation standard deviation of the error for the selected k:" ,knnStd

    print
    np.random.seed(rSeed)
    lcRange = np.arange(-0.5, 4.01, 0.5)
    cRange = np.power(10,lcRange)
    lrParam, lrRes, lrStd = util.getParamAndRes(AllData, AllLabels, learners.LogisticRegressionClassifier, cRange, util.calculateClassificationError, numCrossVal, logTick = True, save=saveImages)
    print "Selected regularizer (C) for logistic regression: ", lrParam, " (larger values imply less regularization)"
    print "Logreg cross-validation average error for the selected C:" ,lrRes
    print "Logreg cross-validation standard deviation of the error for the selected C:" ,lrStd
    
    print
    if np.abs(lrRes-knnRes) < 1e-16:
      print "Both logreg and knn have the same cross-validation score"
    elif lrRes < knnRes:
      print "Logreg has a better cross-validation score than kNN with %.2f vs %.2f" % (lrRes, knnRes)
    else:
      print "kNN has a better cross-validation score than Logreg with %.2f vs %.2f" % (knnRes, lrRes)
    print

  if regression:
    #Regression
    print("--- Regression ---")
    print

    dataNames = ['variation1','variation2','airfoil']
    Data, Target = data.getDataForRegression(dataNames)

    regressors = [learners.LinearRegression, learners.RidgeRegression] 
    paramSets = {learners.LinearRegression.name:[1], \
                 learners.RidgeRegression.name:np.arange(0, 4.1, 0.1)}
    for name in dataNames:
      for method in regressors:
        np.random.seed(rSeed) 
        linParam, linRes, linStd = util.getParamAndRes(Data[name], Target[name],method, paramSets[method.name], util.calculateMeanL2Error, numCrossVal, "_"+name,save=saveImages)
        print "Selected parameter for " + method.name + " and " + name + " data: " + str(linParam)
        print method.name + " cross-validation average error for " + name + " data: " + str(linRes)
        print method.name + " cross-validation standard deviation of the error for " + name + " data: "  + str(linStd)
        print

    print "Selected parameter for Linear regression does not matter. Ridge regression with a regularizer equal to 0 is equivalent to linear regression."
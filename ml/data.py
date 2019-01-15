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

import cv2
import numpy as np
import os
import util
import pickle

class SaturationHistogramExtractor():
  def __init__(self, numBins):
    self.numBins = numBins

  def extract(self, img):
    """ 
    Implement the code to extract the saturation histogram of the image. Number of histogram bins should be taken as an input.
    Do not forget to normalize with the number of pixels!
    Return a 1D numpy array composed of the image features.


   -- Calculate the saturation value for each pixel of the image (opencv 1 line)
   -- Calculate the saturation histogram given these values (numpy 1 line)
   -- Normalize the histogram with the number of image pixels

    Some hints:
    - Saturation values are between 0 and 255
    - OpenCV  loads the images as BGR not RGB!
    - Your target color space is HSV 
    - Pixel number is an integer but beware of integer division for normalization!
    - 5 lines of code is enough
    - numpy has a function to calculate histograms

    """

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    a, b, c = hsv.shape
    hist,bins = np.histogram(hsv[:, :, 1].flatten(), bins = self.numBins)
    hist = hist/float(a*b)

    return hist

class ImageLoader(object):
  def __init__(self, classNames = ['wood','metal'], dataFolder = './', featureExtractor = SaturationHistogramExtractor(16)):
    """ 
    This class loads the images of each class and extract features. Images should be organized as:
      dataFolder/className1
      dataFolder/className2
      ...
      dataFolder/classNameN
    where classNames = ['className1', 'className2', ... , 'classNameN']
    For this project, everything is setup for you

    featureExtractor: This expects an object that has a method called extract which takes an image as an input and returns
    its features as a 1D numpy array
    """
    self.names = classNames
    self.fE = featureExtractor
    self.dF = dataFolder

  def loadData(self): 
    """ Loads the images and extracts their features with the supplied feature extractor. You do not need to modify this method. """
    Data = []
    Labels = []
    Files = []
    for name in self.names:
      directory = './' + name + '/';
      files = os.listdir(directory)
      files.sort()
      for f in files:
        if(f.endswith('.jpg')):
          img = cv2.imread(directory + f)
          feats = self.fE.extract(img)
          Data.append(feats)            
          Labels.append(name)
          Files.append(directory + f)
    return np.array(Data), np.array(Labels), Files

#You need to apply the same processing to the test so make sure to store it as well, maybe put it in the linear regression code?
class RegressionDataLoader(object):
  def __init__(self, dataFolder = 'regression'):
    """ 
    This class loads regression datasets without any featyure extraction or pre-processing
    Data should be organized as
      dataFolder/dataSetName.txt

    For this project, everything is setup for you

    featureExtractor: This expects an object that has a method called extract which takes all the data, organized
    as numPoints x dimensionality, as the input and returns its features as a 1D numpy array
    """
    self.dataFolder = dataFolder

  def loadData(self,dataName):
    """
    This method assumes that each line corresponds to a measurement and a target:
    x11 x12 ... x1d y1
    x21 x22 ... x2d y2
    ...
    xn1 xn2 ... xnd yn

    where n is the number of datapoints, 
    d is the data dimensionality,
    xi's correspond to variables 
    and yj correspond to the targets    
    """

    Data = []
    Targets = []
    fname = './' + self.dataFolder + '/' + dataName + '.txt'
    tmp = np.genfromtxt(fname,dtype=None)
    data=np.zeros([len(tmp),len(tmp[0])])
    for i in range(len(tmp)):
      for j in range(len(tmp[0])):
        data[i,j]=tmp[i][j]

    Data = data[:,:-1]
    Targets = data[:,-1]
    return Data, Targets        

class Normalizer(object):
  def __init__(self, Data):
    """
    Calculates the parameters to do z-normalization
    Data is a d x n numpy array 
    calculates the parameters to do preprocessing
    """
    self.means = np.mean(Data,axis=0)
    self.stds =  np.std(Data,axis=0)

  def preProc(self, Data):
    return (Data-self.means)/self.stds

def getDataForClassification(pickleEnabled = True, featureExtractor = SaturationHistogramExtractor):
  classNames = ['wood','metal']  
  fName = 'classifier.p'
  if( os.path.isfile(fName) and pickleEnabled):
    f = open(fName, "rb")
    tmp = pickle.load(f)
    AllData = tmp['Data']
    AllLabels = tmp['Labels']
  else:
    imL = ImageLoader(classNames)
    AllData, AllLabels, AllFiles = imL.loadData()
    if pickleEnabled:
      tmp = {}
      tmp['Data'] = AllData;
      tmp['Labels'] = AllLabels
      tmp['Files'] = AllFiles
      f = open(fName, "wb")
      pickle.dump(tmp, f)  
  return AllData, AllLabels

def getDataForRegression(dataNames):
  RegDataLoader = RegressionDataLoader('regression')
  Data = {}
  Target = {}
  for name in dataNames:
    Data[name], Target[name] = RegDataLoader.loadData(name) 
  return Data, Target

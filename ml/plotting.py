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

import matplotlib.pyplot as plt
import numpy as np

def plotVsParam(paramSet, trainRes, testRes, title = None, fName = None, logTick = False, save = True):
  """
  paramSet is assumed to be single dimensional.
  """
  
  mTest  = np.mean(testRes,axis=1)
  sTest  = np.std(testRes,axis=1)
  mTrain = np.mean(trainRes,axis=1)
  sTrain = np.std(trainRes,axis=1)
  
  upperLim = max([np.max(mTest+np.max(sTest)),np.max(mTrain+np.max(sTrain))])
  lowerLim = min([np.min(mTest-np.max(sTest)),np.min(mTrain-np.max(sTrain))])
  plt.clf()
  
  if(logTick):
    paramSet = np.log10(paramSet)
    if len(paramSet) < 10:
      tickLabels = ['$10^'+"{%.1f" % x + '}$' for x in paramSet]
    else:
      tickLabels = []
      k = int(len(paramSet)/10.)*2
      for i in range(0,len(paramSet)):
        if i%k==0:
          tickLabels.append('$10^'+"{%.1f" % paramSet[i] + '}$')
        else:
          tickLabels.append('')
    plt.xticks(paramSet, tickLabels)
  else:
    if len(paramSet) < 20:
      plt.xticks(paramSet)
    else:
      tickLabels = []
      k = int(len(paramSet)/20.)*2
      for i in range(0,len(paramSet)):
        if i%k==0:
          tickLabels.append("${%.1f" % paramSet[i] + "}$")
        else:
          tickLabels.append('') 
      plt.xticks(paramSet, tickLabels)
  plt.errorbar(paramSet, mTest,  yerr=sTest,  label = 'Testing')
  plt.errorbar(paramSet, mTrain, yerr=sTrain, label = 'Training')
  plt.axis([paramSet[0]-0.01, paramSet[-1]+0.01, lowerLim-0.01, upperLim+0.01])

  plt.legend()
  
  if title:
    plt.title(title)
    
  #shameless hack to prevent saving linear regression plots
  
  if fName[0:6] == 'linreg':
    return
  
  if fName and save:
    plt.savefig(fName)
  else:
    plt.show()
  

  
    
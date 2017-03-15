from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import pylab as pl

def MaxMinNormalization(x,Max,Min):  
    x = (x - Min) / (Max - Min);  
    return x; 
def themaxmin(datArr):
    Min = [1e10 for i in range(shape(datArr)[1]-1)]
    Max = [-1 for i in range(shape(datArr)[1]-1)]
    for line in datArr:
        for i in range(shape(datArr)[1]-1):
            if line[i] <= Min[i]:
                Min[i] = line[i]
            if line[i] > Max[i]:
                Max[i] = line[i]
    return Min,Max


def loadDataSet(fileName,delim = '\t'):
    fr = open(fileName)
    stringArr = [line.strip().split() for line in fr.readlines()]
    datArr =  [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat,topNfeat = 9999999):
    meanVals = mean(dataMat,axis = 0)
    meanRemoved = dataMat - meanVals
    covMat  = cov(meanRemoved,rowvar = 0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigVaIlnd = argsort(eigVals)
    print eigVaIlnd
    eigValInd = eigValInd[:-(topNfeat + 1) : -1]
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat,reconMat


dataMat = loadDataSet('25dian.txt','  ')
pca(dataMat,1)
# print shape(dataMat)[1]
# Min,Max = themaxmin(dataMat)

# for line in dataMat:
#     for i in range(shape(dataMat)[1]-1):
#         line[i] =  MaxMinNormalization(line[i],Max[i],Min[i])
# lowDMat, reconMat = pca(dataMat,1)

# fig = plt.figure()

# ax = fig.add_subplot(211)
# ax.scatter(dataMat[0:164, 0].flatten().A[0], dataMat[0:164, 1].flatten().A[0], marker='^', s=90,c = 'blue')
# ax.scatter(dataMat[165:241, 0].flatten().A[0], dataMat[165:241, 1].flatten().A[0], marker='^', s=90,c = 'red')

# ax = fig.add_subplot(212)
# ax.scatter(reconMat[0:164, 0].flatten().A[0], reconMat[0:164, 1].flatten().A[0], marker='o', s=50, c='blue')
# ax.scatter(reconMat[165:241, 0].flatten().A[0], reconMat[165:241, 1].flatten().A[0], marker='o', s=50, c='red')
# plt.show()




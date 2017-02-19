#-*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pylab as pl 
import math 

def creatdata2(dataSet):
    data = [line.strip().split() for line in dataSet]
    heightlist = []
    for i in data:
        i = [float(item) for item in i]
        heightlist.append(i)
    return np.array(heightlist)

def nuclear(x1,x2,var):
    #x2为均值，var为设置的协方差矩阵，
    # return 1.0 * 1/np.sqrt(2 * np.pi) * h * np.e ** (-(x1 - x2) ** 2 / (2 * (h**2)))
    return 1.0 / math.sqrt(2.0 * np.pi * np.linalg.det(var)) \
    * np.e ** (-1.0 / 2.0 * (x1 - x2).dot(np.linalg.inv(var))\
        .dot((x1 - x2).T))

def thesex(your_height,var,num):#,k): #绘制ROC曲线时需要添加参数k
    Posterior_probability = [] #概率
    for i in range(len(practice_array)):
        sum = 0.0
        for j in practice_array[i]:
            sum += nuclear(your_height[0:num],j[0:num],var[0:num,0:num])
        Posterior_probability.append(1.0/len(practice_array[i]) * sum) #分别为0,1,2,3,4,5的概率
    # return Posterior_probability[k] #记录每个样本的阳性决策值，用于绘制ROC曲线
    return Posterior_probability.index(max(Posterior_probability)) #直接输出分类结果，用于检查错误率
    # return Posterior_probability #用于绘制Parzen窗
def classification(a,b):#判断类别
    if a >= b:
        return 1
    else:
        return 0

#绘制Parzen窗口（每个特征一张图）
def drawwindows(traDataSet,num,sigma = None,Xmin = None,XMax = None):
    if num != 1:
        print "只能画出一维图像"
        return 
    if Xmin == None:
        rate = 1.6
        Xmin = min([min(dataSet[0:num]) for dataSet in traDataSet]) / rate
        Xmax = max([max(dataSet[0:num]) for dataSet in traDataSet]) * rate
    x = np.arange(Xmin,Xmax,(Xmax - Xmin) / 100)
    color = ['blue','red','green','yellow','black','pink']
    for i in range(len(traDataSet)):
        f = 0
        dataSet = traDataSet[i]
        for data in dataSet:
            mu = float(data)
            y = 1.0/math.sqrt(2 * np.pi * sigma) * np.e ** (-1.0/2 * (x - mu) ** 2 / sigma)
            f += y
            pl.plot(x,y,color = color[i])
        pl.plot(x,f,color = color[i],label = "Gesture"+str(i))
        pl.legend(loc = 'lower right')
    pl.show()

if __name__ == '__main__':

    #practice
    num = input("请输入特征的个数：")
    practice_array = []
    Rates = [[[] for i in range(6)] for j in range(5)]
    with open('HandPractice_t_0.txt', 'r') as zero:
        practice_array.append(creatdata2(zero))   
        Rates[0][0] = [line[0] for line in practice_array[0]]
        Rates[1][0] = [line[1] for line in practice_array[0]]
        Rates[2][0] = [line[2] for line in practice_array[0]]
        Rates[3][0] = [line[3] for line in practice_array[0]]
        Rates[4][0] = [line[4] for line in practice_array[0]]
    with open('HandPractice_t_1.txt', 'r') as one:
        practice_array.append(creatdata2(one))
        Rates[0][1] = [line[0] for line in practice_array[1]]
        Rates[1][1] = [line[1] for line in practice_array[1]]
        Rates[2][1] = [line[2] for line in practice_array[1]]
        Rates[3][1] = [line[3] for line in practice_array[1]]
        Rates[4][1] = [line[4] for line in practice_array[1]]
    with open('HandPractice_t_2.txt', 'r') as two:
        practice_array.append(creatdata2(two))
        Rates[0][2] = [line[0] for line in practice_array[2]]
        Rates[1][2] = [line[1] for line in practice_array[2]]
        Rates[2][2] = [line[2] for line in practice_array[2]]
        Rates[3][2] = [line[3] for line in practice_array[2]]
        Rates[4][2] = [line[4] for line in practice_array[2]]
    with open('HandPractice_t_3.txt', 'r') as three:
        practice_array.append(creatdata2(three))
        Rates[0][3] = [line[0] for line in practice_array[3]]
        Rates[1][3] = [line[1] for line in practice_array[3]]
        Rates[2][3] = [line[2] for line in practice_array[3]]
        Rates[3][3] = [line[3] for line in practice_array[3]]
        Rates[4][3] = [line[4] for line in practice_array[3]]
    with open('HandPractice_t_4.txt', 'r') as four:
        practice_array.append(creatdata2(four))
        Rates[0][4] = [line[0] for line in practice_array[4]]
        Rates[1][4] = [line[1] for line in practice_array[4]]
        Rates[2][4] = [line[2] for line in practice_array[4]]
        Rates[3][4] = [line[3] for line in practice_array[4]]
        Rates[4][4] = [line[4] for line in practice_array[4]]
    with open('HandPractice_t_5.txt', 'r') as five:
        practice_array.append(creatdata2(five))
        Rates[0][5] = [line[0] for line in practice_array[5]]
        Rates[1][5] = [line[1] for line in practice_array[5]]
        Rates[2][5] = [line[2] for line in practice_array[5]]
        Rates[3][5] = [line[3] for line in practice_array[5]]
        Rates[4][5] = [line[4] for line in practice_array[5]]

# #画出parzen窗
#     Var = [  1.41094664e-08, 2.28484331e-15, 1.32999557e-21, 3.39695969e-22, 5.27721941e-06]
#     pl.title("Pazen Windows")
#     pl.grid()
#     pl.ylabel("The Rate")
#     color = ['blue','red','green','yellow','black','pink']
#     for sample in range(5):
#         pl.xlabel("The Feature " + str(sample + 1))
#         drawwindows(Rates[sample][0:num],Var[sample])

    #test

    test_array = []
    # var = np.array([[  1.10726271e-09, 1.24604594e-12, 5.75481503e-16, 4.15500710e-16, 1.41094664e-08]\
    # ,[  1.24604594e-12, 2.28484331e-15, 5.03500420e-19, 3.93142524e-19, -1.35390010e-12]\
    # ,[  5.75481503e-16, 5.03500420e-19, 1.32999557e-21, 5.70314053e-22, 2.30608783e-15]\
    # ,[  4.15500710e-16, 3.93142524e-19, 5.70314053e-22, 3.39695969e-22, 4.64068120e-15]\
    # ,[  1.41094664e-08, -1.35390010e-12, 2.30608783e-15, 4.64068120e-15, 5.27721941e-06]])
    
    # var = np.array([[  1.41094664e-08, 0,0,0,0]\
    # ,[  0, 4.58484331e-15, 0,0,0]\
    # ,[  0,0, 0.32999557e-21, 0,0]\
    # ,[  0,0,0, 2.93142524e-22, 0]\
    # ,[  0,0,0,0, 5.27721941e-06]])

    var = np.array([[  1.10726271e-09, 0,0,0,0]\
    ,[  0, 2.28484331e-15, 0,0,0]\
    ,[  0,0, 1.32999557e-21, 0,0]\
    ,[  0,0,0, 3.39695969e-22, 0]\
    ,[  0,0,0,0, 5.27721941e-06]])


    with open('HandTest_t_0.txt', 'r') as zero:
        test_array.append(creatdata2(zero))
    with open('HandTest_t_1.txt', 'r') as one:
        test_array.append(creatdata2(one))
    with open('HandTest_t_2.txt', 'r') as two:
        test_array.append(creatdata2(two))
    with open('HandTest_t_3.txt', 'r') as three:
        test_array.append(creatdata2(three))
    with open('HandTest_t_4.txt', 'r') as four:
        test_array.append(creatdata2(four))
    with open('HandTest_t_5.txt', 'r') as five:
        test_array.append(creatdata2(five))

#检测错误率
    for i in range(0,len(test_array)):
        error = 0
        for j in test_array[i]:
            result = thesex(j,var,num)
            print j,result
            if result != i:
                error += 1
        print i,':',1.0 * error / test_array[i].shape[0]


# #画出ROC曲线
#     pl.title("The ROC Curve of  Bayes classifier in Hand Gesture Recognition")
#     pl.xticks(np.arange(-0.05,1.05,0.05))
#     pl.yticks(np.arange(-0.05,1.05,0.05))
#     pl.grid()
#     for sample in range(0,len(test_array)):
#         Rate = []
#         Sn = []
#         Sp = []
#         y = []
#         for i in range(0,len(test_array)):
#             for j in test_array[i]:
#                 rate = thesex(j,var,sample)
#                 Rate.append([rate,i])

#         Rate.sort(key=lambda x:x[0])
#         for threshold in Rate:
#             TP,FP,FN,TN = 0,0,0,0
#             for k in Rate:
#                 result = classification(k[0],threshold[0])
#                 if sample == k[1]:
#                     if result == 1:
#                         TP += 1
#                     else:
#                         FN += 1
#                 else:
#                     if result == 1:
#                         FP += 1
#                     else:
#                         TN += 1

#             Sn.append(1.0 * TP / (TP + FN))
#             Sp.append(1.0 * TN / (TN + FP))
#             y.append(1 - 1.0 * TN / (TN + FP))
#         auc = 0.
#         prev_x = 0.
#         for i in range(len(Sn)):
#             if Sn[i] != prev_x:
#                 auc += (Sn[i] - prev_x) * y[i]
#                 prev_x = Sn[i]
#         pl.xlabel("False Positive Rate")
#         pl.ylabel("True Positive Rate")
#         Sn.append(0.0)
#         y.append(0.0)
#         pl.plot(y, Sn,label = "Gesture"+str(sample)+"'s auc is " + str(auc))# use pylab to plot x and y
#         pl.legend(loc = 'lower right')
#     x = [-0.05,1.05]
#     y = x
#     pl.plot(x,y,'--')
#     pl.show()# show the plot on the screen

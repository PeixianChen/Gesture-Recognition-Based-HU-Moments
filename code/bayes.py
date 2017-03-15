#-*-coding:utf-8-*-
import numpy as np
import pylab as pl 
import math 

def creatdata2(dataSet):
    data = [line.strip().split() for line in dataSet]
    heightlist = []
    for i in data:
        i = [float(item) for item in i]
        heightlist.append(i)
    return np.array(heightlist)


# 求均值和协方差矩阵
def mulnormal(Array, num):
    avg_array = np.mean(Array[:,0:num], axis=0)
    array_row = Array[:,0:num] - avg_array
    array_col = (Array[:,0:num] - avg_array).T
    var_array = array_col.dot(array_row) / (Array.shape[0] - 1.0)
    return avg_array, var_array


def thesex(your, num):#, k): #绘制ROC曲线时需要添加参数k
    Prior_probability = [] #概率
    for i in range(0,len(practice_array)):
        # Prior_probability.append(1.0 / math.sqrt(2.0 * np.pi * np.linalg.det(practice_var_array[i])) * np.e ** (-1.0 / 2.0 * (your[0:num] - practice_avg_array[i]).dot(np.linalg.inv(practice_var_array[i])).dot((your[0:num] - practice_avg_array[i]).T)))
        x = -1.0 / 2 * math.log( 2.0 * np.pi *  np.linalg.det(practice_var_array[i])) - 1.0 / 2.0 * (your[0:num] - practice_avg_array[i]).dot(np.linalg.inv(practice_var_array[i])).dot((your[0:num] - practice_avg_array[i]).T)
        Prior_probability.append(x)
   

    Posterior_probability = [] #后验概率

    for i in range(0,len(practice_array)):
        # Posterior_probability.append(1.0 * p * Prior_probability[i]/ (1.0 * p * sum(Prior_probability)))
        Posterior_probability.append(math.log(p) + Prior_probability[i])

    # return Posterior_probability[k]#记录每个样本的阳性决策值，用于绘制ROC曲线
    return Posterior_probability.index(max(Posterior_probability)) #直接输出分类结果，用于检查错误率
def classification(a,b):#判断类别
    if a >= b:
        return 1
    else:
        return 0

if __name__ == '__main__':

    #practice
    num = input("请输入特征的个数：")
    practice_array, practice_avg_array, practice_var_array = [],[],[]
    with open('Handpractice_add_0.txt', 'r') as zero:
        practice_array.append(creatdata2(zero))
        practice_avg_array.append(mulnormal(practice_array[-1], num)[0])
        practice_var_array.append(mulnormal(practice_array[-1], num)[1])
    with open('Handpractice_add_1.txt', 'r') as one:
        practice_array.append(creatdata2(one))
        practice_avg_array.append(mulnormal(practice_array[-1], num)[0])
        practice_var_array.append(mulnormal(practice_array[-1], num)[1])
    with open('Handpractice_add_2.txt', 'r') as two:
        practice_array.append(creatdata2(two))
        practice_avg_array.append(mulnormal(practice_array[-1], num)[0])
        practice_var_array.append(mulnormal(practice_array[-1], num)[1])
    with open('Handpractice_add_3.txt', 'r') as three:
        practice_array.append(creatdata2(three))
        practice_avg_array.append(mulnormal(practice_array[-1], num)[0])
        practice_var_array.append(mulnormal(practice_array[-1], num)[1])
    with open('Handpractice_add_4.txt', 'r') as four:
        practice_array.append(creatdata2(four))
        practice_avg_array.append(mulnormal(practice_array[-1], num)[0])
        practice_var_array.append(mulnormal(practice_array[-1], num)[1])
    with open('Handpractice_add_5.txt', 'r') as five:
        practice_array.append(creatdata2(five))
        practice_avg_array.append(mulnormal(practice_array[-1], num)[0])
        practice_var_array.append(mulnormal(practice_array[-1], num)[1])

# # 柱状图
#     n = [[],[],[],[],[]]
#     for array in practice_array:
#         for line in array:
#             n[0].append(line[0])
#             n[1].append(line[1])
#             n[2].append(line[2])
#             n[3].append(line[3])
#             n[4].append(line[4])
#     for line in n: 
#         pl.hist(line, bins=100,color='lightblue',normed=False)
#         pl.show()

    p = 1.0 * len(practice_array)


    #test

    test_array = []
    
    with open('HandTest_add_0.txt', 'r') as zero:
        test_array.append(creatdata2(zero))
    with open('HandTest_add_1.txt', 'r') as one:
        test_array.append(creatdata2(one))
    with open('HandTest_add_2.txt', 'r') as two:
        test_array.append(creatdata2(two))
    with open('HandTest_add_3.txt', 'r') as three:
        test_array.append(creatdata2(three))
    with open('HandTest_add_4.txt', 'r') as four:
        test_array.append(creatdata2(four))
    with open('HandTest_add_5.txt', 'r') as five:
        test_array.append(creatdata2(five))

    print practice_var_array
# #输出错误率
#     for i in range(0,len(test_array)):
#         error = 0
#         for j in test_array[i]:
#             result = thesex(j,num)
#             print j,result
#             if result != i:
#                 error += 1
#         print i,':',1.0 * error / test_array[i].shape[0]

# #画ROC曲线
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
#                 rate = thesex(j,num,sample)
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
#             # print "sample:",sample
#             # print "k[1]:",k[1]
#             # print "threshould:",threshold[0]
#             # print "tp:",TP
#             # print "fn:",FN
#             # print "tn:",TN
#             # print "FP:",FP

#             Sn.append(1.0 * TP / (TP + FN))
#             Sp.append(1.0 * TN / (TN + FP))
#             y.append(1 - 1.0 * TN / (TN + FP))
#             # print "Sn:",1.0 * TP / (TP + FN)
#             # print "----------------------"
#         # print sample,Sn,y
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


# 最小风险贝叶斯。
# 设男生被误判为女生的损失为6，女生被误判为男生的损失为1.

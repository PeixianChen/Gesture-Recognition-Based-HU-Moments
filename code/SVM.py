#-*-coding:utf-8-*-
import numpy as np
import pylab as pl 
import math 
from sklearn import svm  


def creatdata2(dataSet):
    data = [line.strip().split() for line in dataSet]
    heightlist = []
    for i in data:
        i = [float(item) for item in i]
        heightlist.append(i)
    return heightlist


if __name__ == '__main__':

    #practice
    # num = input("请输入特征的个数：")
    practice_array, practice_target = [],[]
    with open('HandPractice_add_0.txt', 'r') as zero:
        practice_array += creatdata2(zero)
        practice_target += [0 for i in range(50)]
    with open('HandPractice_add_1.txt', 'r') as one:
        practice_array += creatdata2(one)
        practice_target += [1 for i in range(50)]
    with open('HandPractice_add_2.txt', 'r') as two:
        practice_array += creatdata2(two)
        practice_target += [2 for i in range(50)]
    with open('HandPractice_add_3.txt', 'r') as three:
        practice_array += creatdata2(three)
        practice_target += [3 for i in range(50)]
    with open('HandPractice_add_4.txt', 'r') as four:
        practice_array += creatdata2(four)
        practice_target += [4 for i in range(50)]
    with open('HandPractice_add_5.txt', 'r') as five:
        practice_array += creatdata2(five)
        practice_target += [5 for i in range(50)]

    x = np.array(practice_array)
    y = np.array(practice_target)

    clf = svm. SVC(kernel = 'poly',decision_function_shape='ovr',degree = 10,probability = True)
    # clf = svm.LinearSVC (multi_class = 'crammer_singer')
    print clf.fit(x, y)
    
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

    for i in range(len(test_array)):
        error = 0
        for j in test_array[i]:
            result = clf.predict([j])
            print j,result
            if result[0] != i:
                error += 1
        print i,':',1.0 * error / len(test_array[i])
   


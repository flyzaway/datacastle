# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:01:07 2017
gbdt(Gradient Boosting Decision Tree)是一种迭代的决策树算法，该算法由多棵决策树组成，
所有树的结论累加起来做最终答案。它在被提出之初就和SVM一起被认为是泛化能力较强的算法。
近些年更因为被用于搜索排序的机器学习模型而引起大家关注。
@author: flyaway
"""
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from load_data import load_data
#from sklearn import preprocessing
####load data################
datasets = load_data()
train_set_x, train_set_y = datasets[0]
valid_set_x, valid_set_y = datasets[1]
test_set_x,test_student_id= datasets[2]


model = SelectKBest(chi2,k = 400).fit(train_set_x,train_set_y)
train_set_x = model.transform(train_set_x)
valid_set_x = model.transform(valid_set_x)
test_set_x = model.transform(test_set_x)

#PCA

#==============================================================================
# pca = PCA(n_components=250)
# pca.fit(train_set_x)
# train_set_x = pca.fit_transform(train_set_x)
# valid_set_x = pca.fit_transform(valid_set_x)
# test_set_x = pca.fit_transform(test_set_x)
# 
#==============================================================================
######load data###############


######train model#############
clf = GradientBoostingClassifier(loss = 'deviance',n_estimators = 200,random_state=2016)
clf = clf.fit(train_set_x,train_set_y)
############
train_y_pred = clf.predict(train_set_x)

print("对训练数据处理后标签")
print np.sum(train_set_y == 0),np.sum(train_set_y == 1000),np.sum(train_set_y == 1500),np.sum(train_set_y == 2000)
print("对训练数据的预测")
print np.sum(train_y_pred == 0),np.sum(train_y_pred == 1000),np.sum(train_y_pred == 1500),np.sum(train_y_pred == 2000)

error = 0
for i in range(len(train_set_y)):
    if train_set_y[i] != train_y_pred[i]:
        error += 1
print("训练样本的预测错误的个数以预测准确率")
print error, accuracy_score(train_y_pred,train_set_y)
############

############
valid_y_pred = clf.predict(valid_set_x)
print("对验证数据的真实标签")
print np.sum(valid_set_y == 0),np.sum(valid_set_y == 1000),np.sum(valid_set_y == 1500),np.sum(valid_set_y == 2000)
print("对验证数据的预测")
print np.sum(valid_y_pred == 0),np.sum(valid_y_pred == 1000),np.sum(valid_y_pred == 1500),np.sum(valid_y_pred == 2000)

error = 0
for i in range(len(valid_y_pred)):
    if valid_set_y[i] != valid_y_pred[i]:
        error += 1
print("验证样本的预测错误的个数以预测准确率")
print error,accuracy_score(valid_y_pred,valid_set_y)
############

############
test_y_pred = clf.predict(test_set_x)
print("对测试数据的预测")
print np.sum(test_y_pred == 0),np.sum(test_y_pred == 1000),np.sum(test_y_pred == 1500),np.sum(test_y_pred == 2000)
############
######train model#############

#写入结果

wf = open('./GBDT/predict.csv','wb')
wf.write('%s,%s\n'%('studentid','subsidy'))
for i in range(len(test_student_id)):
         wf.write('%s,%s\n'%(test_student_id[i],test_y_pred[i]))
wf.close()

#写入结果

#  0:9315
#1000:744
#1500:470
#2000:356
"""
t10_5,t15_8,t20_8
475 featureselection 250,GBDT = 200,成绩0.02558,第四
对训练数据处理后标签
7460 3556 3348 2549
对训练数据的预测
7414 3552 3423 2524
训练样本的预测错误的个数以预测准确率
1082 0.936025542482
对验证数据的真实标签
1865 890 837 637
对验证数据的预测
1851 858 884 636
验证样本的预测错误的个数以预测准确率
436 0.896902340979
对测试数据的预测
8973 1128 585 275
"""
"""
t10_5,t15_7,t20_8
600 featureselection 600,GBDT = 200,成绩第五名
对训练数据处理后标签
7460 3556 2976 2549
对训练数据的预测
7386 3678 2944 2533
训练样本的预测错误的个数以预测准确率
857 0.948189347682
对验证数据的真实标签
1865 890 744 637
对验证数据的预测
1843 916 739 638
验证样本的预测错误的个数以预测准确率
406 0.901837524178
对测试数据的预测
9071 1135 486 269
"""
"""
t10_5,t15_8,t20_10,成绩第四,说明这种重采样的方式不行
600 featureselection 400,GBDT = 200
对训练数据处理后标签
7460 3557 3348 3115
对训练数据的预测
7306 3560 3404 3210
训练样本的预测错误的个数以预测准确率
969 0.944565217391
对验证数据的真实标签
1865 889 837 779
对验证数据的预测
1803 899 848 820
验证样本的预测错误的个数以预测准确率
428 0.902059496568
对测试数据的预测
8902 1037 626 396
"""
#未提交
"""
t10_5,t15_8,t20_10,不再测试了，之后按5,8,8来重采用
600 featureselection 600,GBDT = 200
对训练数据处理后标签
7460 3557 3348 3115
对训练数据的预测
7242 3640 3412 3186
训练样本的预测错误的个数以预测准确率
907 0.948112128146
对验证数据的真实标签
1865 889 837 779
对验证数据的预测
1773 927 864 806
验证样本的预测错误的个数以预测准确率
408 0.906636155606
对测试数据的预测
8875 1118 593 375
"""

#
"""
t10_5,t15_8,t20_8,成绩0.02645,成绩第二
600 featureselection 400,GBDT = 200
对训练数据处理后标签
7460 3556 3348 2549
对训练数据的预测
7346 3612 3413 2542
训练样本的预测错误的个数以预测准确率
898 0.946904747827
对验证数据的真实标签
1865 890 837 637
对验证数据的预测
1826 901 870 632
验证样本的预测错误的个数以预测准确率
386 0.908725467013
对测试数据的预测
9001 1076 615 269
"""
#未提交，如果明天这个效果好,说明不做特征选择也可
"""
t10_5,t15_8,t20_8,成绩排名第八名，说明需要做特征选择
600 featureselection 600,GBDT = 200
对训练数据处理后标签
7460 3556 3348 2549
对训练数据的预测
7290 3654 3441 2528
训练样本的预测错误的个数以预测准确率
846 0.949979305859
对验证数据的真实标签
1865 890 837 637
对验证数据的预测
1795 943 863 628
验证样本的预测错误的个数以预测准确率
369 0.912745329865
对测试数据的预测
8970 1163 590 238
"""
"""
t10_5,t15_8,t20_8,成绩0.02666,成绩第一
854 featureselection 400,GBDT = 200
对训练数据处理后标签
7460 3556 3348 2549
对训练数据的预测
7383 3595 3413 2522
训练样本的预测错误的个数以预测准确率
934 0.944776207651
对验证数据的真实标签
1865 890 837 637
对验证数据的预测
1828 903 868 630
验证样本的预测错误的个数以预测准确率
404 0.904469141641
对测试数据的预测
8981 1147 555 278
"""
#未提交
"""
t10_5,t15_8,t20_8,成绩第三
854 featureselection 400,GBDT = 170
对训练数据处理后标签
7460 3556 3348 2549
对训练数据的预测
7510 3541 3376 2486
训练样本的预测错误的个数以预测准确率
1327 0.921539644061
对验证数据的真实标签
1865 890 837 637
对验证数据的预测
1889 883 853 604
验证样本的预测错误的个数以预测准确率
492 0.88366043982
对测试数据的预测
8933 1166 585 277
"""
"""
t10_5,t15_8,t20_8,第六
1712 featureselection 400,GBDT = 200
对训练数据处理后标签
7460 3556 3348 2549
对训练数据的预测
7346 3698 3363 2506
训练样本的预测错误的个数以预测准确率
902 0.946668243363
对验证数据的真实标签
1865 890 837 637
对验证数据的预测
1803 943 858 625
验证样本的预测错误的个数以预测准确率
381 0.909907779617
对测试数据的预测
9049 1111 541 260
"""


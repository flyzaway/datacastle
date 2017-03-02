# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:52:31 2017

@author: flyaway
"""
import cPickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle
#load 所有数据
def load_data():
    f1 = open('./data/borrow_count.pkl','rb')
    borrow_count = cPickle.load(f1)
    f1.close()
    f2= open('./data/dorm_count.pkl','rb')
    dorm_count = cPickle.load(f2)
    f2.close()
    f3= open('./data/lib_count.pkl','rb')
    lib_count = cPickle.load(f3)
    f3.close()
    f4= open('./data/score.pkl','rb')
    score = cPickle.load(f4)
    f4.close()
    f5= open('./data/subsidy.pkl','rb')
    subsidy = cPickle.load(f5)
    f5.close()
    
    f6= open('./data/card_count.pkl','rb')
    card_count = cPickle.load(f6)
    f6.close()
    #合并所有维度数据 
    tall = pd.concat([borrow_count,dorm_count,lib_count,score,card_count,subsidy],axis = 1)
    
    test_data = tall[(tall['money'] != 0.0) & (tall['money'] != 1000.0) & (tall['money'] != 1500.0 )& (tall['money'] != 2000.0)] 
    test_data['id'] = test_data.index
    test_id = pd.read_csv('./test/subsidy_test.txt',names = ['id'])
    test_data = pd.merge(test_data,test_id,on = 'id')#过滤需要预测的学生id
    test_student_id = test_data['id'].values
    del test_data['id']
    test_data = test_data.fillna(0)
    del test_data['money']
    
    train_data = tall[(tall['money'] == 0.0) | (tall['money'] == 1000.0) | (tall['money'] == 1500.0 ) | (tall['money'] == 2000.0)] 
    train_data = train_data.fillna(0)
    #过采样
    t10 = train_data[train_data['money'] == 1000.0]
    t15 = train_data[train_data['money'] == 1500.0]
    t20 = train_data[train_data['money'] == 2000.0]
    t10_5 = pd.concat([t10,t10,t10,t10,t10],axis = 0)
    t15_8 = pd.concat([t15,t15,t15,t15,t15,t15,t15,t15],axis = 0)
    t20_8 = pd.concat([t20,t20,t20,t20,t20,t20,t20,t20,],axis = 0)
    train_data = pd.concat([train_data,t10_5,t15_8,t20_8],axis = 0)
    train_label = train_data['money'].values
    print type(train_label)
    train_label = np.array([int(i) for i in train_label])
    print type(train_label)
    train_data = train_data.fillna(0.)
    del train_data['money']
    train_data = train_data.values
    train_data,train_label = shuffle(train_data,train_label,random_state = 1111)
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=1000) 
    for train_index, valid_index in sss.split(train_data, train_label):
        train_set_x, valid_set_x = train_data[train_index], train_data[valid_index]
        train_set_y, valid_set_y = train_label[train_index], train_label[valid_index]
    test_set_x = test_data.values
    return[(train_set_x,train_set_y),(valid_set_x,valid_set_y),(test_set_x,test_student_id)]

if __name__ == '__main__':
    load_data()
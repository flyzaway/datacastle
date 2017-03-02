# -*- coding: utf-8 -*-
"""
Created on Thu Jan 05 19:41:47 2017

@author: flyaway
""" 
import numpy as np
#提取1维特征
def read_borrow_table(file_name1 = './train/borrow_train.txt',file_name2 ='./test/borrow_test.txt' ):
    f = open('file_name1','r')
    lines = f.readlines()
    f.close()
    train_borrow_data = []
    for line in lines:
        line = line.replace('\n','')
        line = line.split(',')
        if len(line) != 4:
            print line
        train_borrow_data.append(line)
    data_train = pd.DataFrame(data = train_borrow_data,columns = ['id','time','book_name','bianhao'])
    
    f = open('file_name2','r')
    lines = f.readlines()
    f.close()
    test_borrow_data = []
    for line in lines:
        line = line.replace('\n','')
        line = line.split(',')
        if len(line) != 4:
            print line
        test_borrow_data.append(line)
    data_test = pd.DataFrame(data = test_borrow_data,columns = ['id','time','book_name','bianhao'])
    print len(data_train)
    print len(data_test)
    data = pd.concat([data_train,data_test],axis = 0)
    print len(data)
    gt5 = data.groupby('id')
    bt_count = gt5['bianhao'].agg(np.size)
    bt_count = bt_count.reset_index()
    bt_count = bt_count.set_index('id')
    bt_count.columns = ['borrow_count']
    wf = open('./data/borrow_count.pkl','wb')
    cPickle.dump(bt_count,wf)
    wf.close()

#提前24+1+1 =26维特征
def read_dorm_table(file_name1 = './train/dorm_train.txt',file_name2 ='./test/dorm_test.txt' ):
    data_train = pd.read_csv(file_name1,names = ['id','time','isout'])
    data_test = pd.read_csv(file_name2,names = ['id','time','isout'])
    data = pd.concat([data_train,data_test],axis = 0)
    time = data['time']
    time = np.array(time)
    hour = []
    for i in range(len(time)):
        hour.append(int(time[i][11:13]))
    data['hour'] = hour
    #计算学生在一天24个小时进出宿舍的次数
    gt4 = data.groupby(['id','hour'])
    pc = gt4['isout'].agg([np.size])
    pc = pc.reset_index()
    pc = pc.set_index('id')
    gt = pc.groupby('hour')
    hour_id = list(set(pc['hour']))
    dt_io_count_24 = pd.DataFrame()
    for i in hour_id:
        temp = gt.get_group(i)
        del temp['hour']
        dt_io_count_24 = pd.concat([dt_io_count_24,temp],axis = 1)
    dt_io_count_24 = dt_io_count_24.fillna(0.)
    dt_io_count_24.columns = ['00','01','02','03','04','05','06','07','08','09','10','11','12','12','14','15',
                                '16','17','18','19','20','21','22','23']
    #dt_io_count,统计学生进出的总次数
    dt_io_count = (data.groupby(['id']))['isout'].agg([np.size])
    dt_io_count.columns = ['io_count']
    #dt_in_countn统计学生进宿舍的总次数
    dt_in = (data.groupby(['id','isout']))
    data_in_count = dt_in['time'].agg([np.size])
    data_in_count = data_in_count.reset_index()
    data_in_count = data_in_count.set_index('id')
    data_in_count = data_in_count.groupby('isout')
    data_in_count= data_in_count.get_group(0)#取进入宿舍的次数
    del data_in_count['isout']
    data_in_count.columns = ['in_count']
    dt_count =  pd.concat([dt_io_count_24,dt_io_count,data_in_count],axis = 1)
    wf = open('./data/dorm_count.pkl','wb')
    cPickle.dump(dt_count,wf)
    wf.close()
#提取1维特征    
def read_library_table(file_name1 = './train/library_train.txt',file_name2 ='./test/library_test.txt' ):
    data_train = pd.read_csv(file_name1,names = ['id','gate','time'])
    data_test = pd.read_csv(file_name2,names = ['id','gate','time'])
    data = pd.concat([data_train,data_test],axis = 0)
    lt= data.groupby('id')
    pc= lt['gate'].agg(np.size)
    pc = pc.reset_index()
    lib_count= pc.set_index('id')
    wf = open('./data/lib_count.pkl','wb')
    cPickle.dump(lib_count,wf)
    wf.close()

#提取1维特征
def read_score_table(file_name1 = './train/score_train.txt',file_name2 ='./test/score_test.txt' ):
    data_train = pd.read_csv(file_name1,names = ['id','college','order'])
    data_test = pd.read_csv(file_name2,names = ['id','college','order'])
    data = pd.concat([data_train,data_test],axis = 0)
    sgt = data.groupby('college')
    sgt_max = sgt['order'].agg(np.max)
    sgt_max = sgt_max.reset_index()
    sgt_max.columns = ['college','max_order']
    score = pd.merge(data,sgt_max, on ='college')
    score1 = score['order'] / score['max_order']
    score2 = pd.DataFrame()
    score2['id'] = score['id']
    score2['bili'] = score1
    score2 = score2.set_index('id')
    wf = open('./data/score.pkl','wb')
    cPickle.dump(score2,wf)
    wf.close()    
   
    
#提取1维特征
def read_subsidy_table(file_name1 = './train/subsidy_train.txt',file_name2= './test/subsidy_test.txt'):
    data_train = pd.read_csv(file_name1,names = ['id','money'])
    data_test = pd.read_csv(file_name2,names = ['id'])
    data = pd.concat([data_train,data_test],axis = 0)
    data = data.set_index('id')
    wf = open('./data/subsidy.pkl','wb')
    cPickle.dump(data,wf)
    wf.close()
#以上共30维特征    
#already rename card as (0:'id',1:'pos',2:'address',3:'catalog',4:'time',5:'cost',6:'have')     
#提取356维特征
def read_card_table(file_name1 = './train/card_train.txt',file_name2 = './test/card_test.txt'): 
    data_train = pd.read_csv(file_name1)
    data_test = pd.read_csv(file_name2)
    data = pd.concat([data_train,data_test],axis = 0)
    #rename card as (0:'id',1:'pos',2:'address',3:'catalog',4:'time',5:'cost',6:'have')     
    #以人为单位，计算消费的维度 
    if True:    
        #以人为单位，计算消费的维度 
        gt1 = data.groupby('id')
        card_count = gt1['pos'].agg([np.size])
        card_cost = gt1['cost'].agg([np.sum,np.max,np.min,np.mean,np.median])
        card_cost.columns = ['cost_sum','cost_max','cost_min','cost_mean','cost_median']
        card_have = gt1['have'].agg([np.sum,np.max,np.min,np.mean,np.median])
        card_have.columns = ['have_sum','have_max','have_min','have_mean','have_median']
        card_money = pd.concat([card_count,card_cost,card_have] ,axis = 1)
        del card_count,card_cost,card_have
        gc.collect()
        # 1 + 5 + 5 =11
        
        #计算pos 机上花去多少钱,计算卡充值多少钱,计算圈存转账多少钱
        card = data[(data['pos'] == "POS消费") | (data['pos'] == "卡充值") | (data['pos'] == "圈存转账")]
        gt = card.groupby(['id','pos'])
        pc = gt['cost'].agg([np.size,np.sum,np.max,np.min,np.mean,np.median])
        pc = pc.reset_index()
        pc = pc.set_index('id')
        pos = list(set(pc['pos']))
        gt = pc.groupby('pos')
        for i in pos:
            temp = gt.get_group(i)
            del temp['pos']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.)
        del pc,temp
        gc.collect()
      
        #计算分类消费的维度
        gt2 = data.groupby(['id','catalog'])
        pc = gt2['cost'].agg([np.size,np.sum,np.max,np.min,np.mean,np.median])
        pc = pc.reset_index()
        pc = pc.set_index('id')
        catalog = list(set(pc['catalog']))
        gt = pc.groupby('catalog')
        for i in catalog:
            temp = gt.get_group(i)
            del temp['catalog']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.)
        del pc,temp,catalog
        gc.collect()
        
        #计算分时消费的维度,计算每个人在24小时中每一小时的消费情况
        time = data['time']
        time = np.array(time)
        hour = []
        for i in range(len(time)):
            hour.append(time[i][11:13])
        data['hour'] = hour
        gt4 = data.groupby(['id','hour'])
        pc = gt4['cost'].agg([np.size,np.sum,np.max,np.min,np.mean,np.median])
        pc = pc.reset_index()
        pc = pc.set_index('id')
        gt = pc.groupby('hour')
        hour_id = list(set(pc['hour']))
        for i in hour_id:
            temp = gt.get_group(i)
            del temp['hour']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.)
        
        #计算是否周末的消费的维度
        import time as T
        week = []
        for i in range(len(time)):
            week.append(T.strptime(time[i][0:10],'%Y/%m/%d').tm_wday + 1)
        lam = lambda x:1 if x>=6 else 0
        is_week = [lam(x) for x in week]
        data['is_week'] = is_week
        gt4 = data.groupby(['id','is_week'])
        pc = gt4['cost'].agg([np.size,np.sum,np.max,np.min,np.mean,np.median])
        pc = pc.reset_index()
        pc = pc.set_index('id')
        gt = pc.groupby('is_week')
        is_week_id = [1]
        for i in is_week_id :
            temp = gt.get_group(i)
            del temp['is_week']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.)
        
        #计算暑假的消费的维度
        holiday = []
        for i in range(len(time)):
            holiday.append(time[i][5:10])
        lam = lambda x:'shujia' if x >= "07/01" and x <="08/31" else 'no'
        is_holiday = [lam(x) for x in holiday]
        data['is_holiday'] = is_holiday
        gt4 = data.groupby(['id','is_holiday'])
        pc = gt4['cost'].agg([np.size,np.sum,np.max,np.min,np.mean,np.median])
        pc = pc.reset_index()
        pc = pc.set_index('id')
        gt = pc.groupby('is_holiday')
        is_holiday_id = ['shujia']
        for i in is_holiday_id :
            temp = gt.get_group(i)
            del temp['is_holiday']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.)
        
    if True:    
        #计算学生在前10大食堂的pos消费，并非所有的食堂消费
        #以下10个地点为食堂,计算学生在食堂的花费
        #过滤数据
        card2 = data[(data['catalog'] == "食堂")]
        #统计学生去食堂的次数
        gt = card2.groupby(['id'])
        size = gt['pos'].agg([np.size])
        card_money = pd.concat([card_money,size] ,axis = 1)
        
        gt4 = card2.groupby(['id','address'])
        a_sum_count = gt4['cost'].agg([np.size,np.sum,np.min,np.max,np.mean,np.median])
        a_sum_count = a_sum_count.reset_index()
        a_sum_count = a_sum_count.set_index('id')
        top = a_sum_count.groupby(['address']).agg(np.sum)
        top = top.sort_values(by = ['sum'],ascending=False)
        top = top.head(50) #30排名前50的pos机位置(窗口)的消费
        add = list(top.index)
        #for训练求得是学生分别再12个地点消费的总次数和钱数
        for i in add:
            temp = a_sum_count[a_sum_count['address']== i]
            del temp['address']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.) 
        
        #计算学生何时进食堂的信息
      
        #计算许嵩一天内24小时内进出食堂的次数
        gt6 = card2.groupby(['id','hour'])
        pc = gt6['cost'].agg(np.size)
        pc = pc.reset_index()
        pc = pc.set_index('id')
        gt = pc.groupby('hour')    
        hour_id = list(set(pc['hour']))
        for i in hour_id:
            temp = gt.get_group(i)
            del temp['hour']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.)
        del pc,temp,card2
        gc.collect()
        
    if True:    
        #计算学生在前10超市的pos消费，并非所有的食堂消费
        #以下10个地点为超市,计算学生在食堂的花费
        #过滤数据
        card2 = data[(data['catalog'] == "超市")]
        gt4 = card2.groupby(['id','address'])
        a_sum_count = gt4['cost'].agg([np.size,np.sum,np.min,np.max,np.mean,np.median])
        a_sum_count = a_sum_count.reset_index()
        a_sum_count = a_sum_count.set_index('id')
        top = a_sum_count.groupby(['address']).agg(np.sum)
        top = top.sort_values(by = ['sum'],ascending=False)
        top = top.head(10) #10排名前10的pos机位置(窗口)的消费
        add = list(top.index)
        #for训练求得是学生分别再10个地点消费的总次数和钱数
        for i in add:
            temp = a_sum_count[a_sum_count['address']== i]
            del temp['address']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.) 
       
        #计算学生何时进超市的信息
        #计算许嵩一天内24小时内进出超市的次数
        gt6 = card2.groupby(['id','hour'])
        pc = gt6['cost'].agg(np.size)
        pc = pc.reset_index()
        pc = pc.set_index('id')
        gt = pc.groupby('hour')    
        hour_id = list(set(pc['hour']))
        for i in hour_id:
            temp = gt.get_group(i)
            del temp['hour']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.)
        del pc,temp,card2
        gc.collect()
        
    
    if True:
        #####图书馆
        card2 = data[(data['catalog'] == "图书馆")]
        gt4 = card2.groupby(['id','address'])
        a_sum_count = gt4['cost'].agg([np.size,np.sum,np.min,np.max,np.mean,np.median])
        a_sum_count = a_sum_count.reset_index()
        a_sum_count = a_sum_count.set_index('id')
        top = a_sum_count.groupby(['address']).agg(np.sum)
        top = top.sort_values(by = ['sum'],ascending=False)
        top = top.head(5) #10排名前10的pos机位置(窗口)的消费
        add = list(top.index)
        #for训练求得是学生分别再12个地点消费的总次数和钱数
        for i in add:
            temp = a_sum_count[a_sum_count['address']== i]
            del temp['address']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.) 
        #计算学生何时进图书馆的信息
        #计算许嵩一天内24小时内进出图书馆的次数
        gt6 = card2.groupby(['id','hour'])
        pc = gt6['cost'].agg(np.size)
        pc = pc.reset_index()
        pc = pc.set_index('id')
        gt = pc.groupby('hour')    
        hour_id = list(set(pc['hour']))
        for i in hour_id:
            temp = gt.get_group(i)
            del temp['hour']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.)
        del pc,temp,card2
        gc.collect()  
            
    if True:
        #计算学生在前10文印中心的pos消费，并非所有的食堂消费
        #以下10个地点为文印中心,计算学生在食堂的花费
        #过滤数据
        card2 = data[(data['catalog'] == "文印中心")]
        gt4 = card2.groupby(['id','address'])
        a_sum_count = gt4['cost'].agg([np.size,np.sum,np.min,np.max,np.mean,np.median])
        a_sum_count = a_sum_count.reset_index()
        a_sum_count = a_sum_count.set_index('id')
        top = a_sum_count.groupby(['address']).agg(np.sum)
        top = top.sort_values(by = ['sum'],ascending=False)
        add = list(top.index)
        #for训练求得是学生分别再12个地点消费的总次数和钱数
        for i in add:
            temp = a_sum_count[a_sum_count['address']== i]
            del temp['address']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.)     
        
        
        #(在超市、图书馆、其他)三种消费最高的方式种的刷卡机的地点，除食堂之外
        #过滤数据
        card2 = data[(data['catalog'] == "淋浴")| (data['catalog'] == "洗衣房") | (data['catalog'] == "其他") | (data['catalog'] == "开水")]
        gt4 = card2.groupby(['id','address'])
        a_sum_count = gt4['cost'].agg([np.size,np.sum,np.min])
        a_sum_count = a_sum_count.reset_index()
        a_sum_count = a_sum_count.set_index('id')
        top = a_sum_count.groupby(['address']).agg(np.sum)
        top = top.sort_values(by = ['sum'],ascending=False)
        top = top.head(5)
        #add是(在超市、图书馆、其他)三个地方消费最高的10个刷卡机的地点
        add = list(top.index)
        #for训练求得是学生分别再10个地点消费的总次数和钱数
        for i in add:
            temp = a_sum_count[a_sum_count['address']== i]
            del temp['address']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.) 
        
    if True:
        time = np.array(data['time'])
        month = []
        for i in range(len(time)):
           month.append(time[i][0:7])
        data['month'] = month    
        #计算学生月均消费,一共有2013/09~2014/09，2014/09~2015/09 24个月
    
        gt4 = data.groupby(['id','month'])
        pc = gt4['cost'].agg([np.size,np.sum,np.max,np.min,np.mean,np.median])
        pc = pc.reset_index()
        pc = pc.set_index('id')
        gt = pc.groupby('month')
        need_month =set(pc['month'])
        #not_need_month = set(['2014/01','2014/02','2014/07','2014/08','2015/01','2015/02','2015/07','2015/08'])
        #need_month = list(month_catalog - not_need_month)
        for i in list(need_month):
            temp = gt.get_group(i)
            del temp['month']
            card_money = pd.concat([card_money,temp],axis = 1)
        card_money = card_money.fillna(0.)
    
        #计算学生每个月在食堂、超市等各个地方的月均消费
        gt10 = data.groupby(['id','month','catalog'])
        pc = gt10['cost'].agg([np.size,np.sum,np.mean])
        pc = pc.reset_index()
        pc = pc.set_index('id')
        month = list(set(pc['month']))
        catalog = list(set(pc['catalog']))
        for m in range(len(month)):
            data1 = pc[pc['month'] == month[m]]
            for c in  range(len(catalog)):
                data2 = data1[data1['catalog'] == catalog[c]]
                del data2['month']
                del data2['catalog']
                card_money = pd.concat([card_money,data2],axis = 1)
        card_money = card_money.fillna(0)
    
    wf = open('./data/card_count.pkl','wb')
    cPickle.dump(card_money,wf)
    wf.close()
    return card_money
    
    
import pandas as pd
import cPickle
import gc
if __name__ == "__main__":
    card_money = read_card_table()
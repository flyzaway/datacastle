#datacastle
#datacaslt助学金预测,算法赛排名第14，使用单模型
#赛题地址:http://www.pkbigdata.com/common/cmpt/大学生助学金精准资助预测_竞赛信息.html
#数据可自行下载
#read_data.py包含对所有表的特征的提取，特征提取整合训练数据和测试数据，使用pandas数据提取很快，基本10几分钟左右可以将训练数据和测试数据的特征提取完，特征维数大概有2000多维，也可以根据需要自行注释掉不需要的特征，保证电脑内存大于6个G
#load_data.py是对测试和训练样本的特征进行分离，并对不均衡样本进行重采样
#GBDT.py使用GBDT对数据进行预测。
#整个过程需要两步：1）运行read_data.py各个函数。2）运行GBDT.py,该文件自行调用load_data.py,并写入结果

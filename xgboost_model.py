"""
程序名称:     xgboost_model.py
功能描述:     xgb模型
创建日期:     2019-03-28
版本说明:     v1.0
"""

import argparse
import datetime
import codecs
import pandas as pd
from itertools import groupby
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
import xgboost as xgb
import re
import operator
import random
import warnings

warnings.filterwarnings("ignore")

def ceate_feature_map(features):
    outfile = open('./xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()
    
class Model_XGBOOST(object):

    def __init__(self, train_file, model_src_file,model_sel_file,test_file,result_file,feature_src_num,feature_sel_num,is_default):
        
        self.train_file = train_file
        self.model_src_file = model_src_file
        self.model_sel_file = model_sel_file
        self.test_file = test_file
        self.result_file = result_file
        self.feature_src_num = feature_src_num
        self.feature_sel_num = feature_sel_num
        self.is_default = is_default
        
    def train_src_model(self):
        time1 = datetime.datetime.now()
        print("xgboost::train_src_model:start time:", time1.strftime('%Y-%m-%d %H:%M:%S %f'))
        print('xgboost::train_src_model:train file path=%s,feature count=%d' %(self.train_file,self.feature_src_num))

        var_Features = []
        for i in range(self.feature_src_num):
            var_Features.append('f'+str(i))
        var_Features.append('label')
        raw_data = pd.read_table(self.train_file, sep='\t', header=0)
        features = list(raw_data.columns)
        dict1 = dict(zip(var_Features, features))
        dict2 = dict(zip(features,var_Features))
        self.dict_n2c = dict1
        self.dict_c2n = dict2
        raw_data.columns = var_Features
        X_columns_label = [x for x in raw_data.columns if x not in ['label']]
        y_columns_label = [x for x in raw_data.columns if x in ['label']]
        X = raw_data[X_columns_label]
        y = raw_data[y_columns_label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
        #Dtrain = xgb.DMatrix(X_train, label=y_train)
        Dtrain = xgb.DMatrix(X, label=y)
        Dtest = xgb.DMatrix(X_test)
        time2 = datetime.datetime.now()
        print('xgboost::train_src_model:read train data done!cost time=%d'%(time2-time1).seconds)
        
        params = {'booster': 'gbtree',            # BOOST类型：GBDT
                  'objective': 'multi:softprob',  # 问题分类：四分类逻辑回归
                  'num_class':4,                  # 类别数：4类
                  'eval_metric': 'merror',        # 评价指标：merror
                  'max_depth': 6,                 # 树的深度
                  'gamma': 0,                     # 节点分裂的最小损失函数值
                  'lambda': 1,                    # 权重的L2正则化项
                  'subsample': 0.8,               # 随机选择80%样本建立决策树
                  'colsample_bytree': 0.8,        # 随机选择80%特征建立决策树
                  'min_child_weight': 1,          # 叶子节点最小权重
                  'eta': 0.1,                     # 学习率
                  'seed': 0,                      # 随机种子
                  'scale_pos_weight': 1,          # 样本平衡度
                  'nthread': 4,                   # 进程数
                  'silent': 1}                    # 打印信息
        
        watchlist = [(Dtrain, 'train')]
        xgboostModel = xgb.train(params, Dtrain, num_boost_round=80, evals=watchlist)
        time3 = datetime.datetime.now()
        print('xgboost::train_src_model:fit train model done!cost time=%d' % (time3 - time2).seconds)
        
        ypred = xgboostModel.predict(Dtest)
        ylabel = np.argmax(ypred, axis=1)
        
        pickle.dump(xgboostModel, open(self.model_src_file, 'wb'))
        feat_names = var_Features[0:-1]
        ceate_feature_map(feat_names)
        importance = xgboostModel.get_fscore(fmap='./xgb.fmap')
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        df = pd.DataFrame(importance, columns=['feature', 'fscore'])
        df['fscore'] = df['fscore'] / df['fscore'].sum()
        
        for key, val in self.dict_n2c.items():
            df = df.replace(key, val)
        
        df.to_csv("./feat_importance.csv", index=False)
        time4 = datetime.datetime.now()
        print("xgboost::train_src_model:cost time:%s,save model path:%s:"%(time4.strftime('%Y-%m-%d %H:%M:%S %f'),self.model_src_file))
        
    def train_sel_model(self):
        time1 = datetime.datetime.now()
        print("xgboost::train_sel_model:start time:", time1.strftime('%Y-%m-%d %H:%M:%S %f'))
        print('xgboost::train_sel_model:train file path=%s,feature count=%d' %(self.train_file,self.feature_sel_num))

        var_Features = []
        for i in range(self.feature_src_num):
            var_Features.append('f'+str(i))
        var_Features.append('label')
        raw_data = pd.read_table(self.train_file, names=var_Features, sep='\t', header=None)
        raw_data.drop(0, inplace=True)
        
        def sel_feature(feature_sel_num):

            feature = []
            with open('./feat_importance.csv', 'r') as fid:
                for line in fid.readlines():
                    feat, val = re.split(',', line.strip())
                    feature.append(feat)
            feature.reverse()
            feature.pop()
            if len(feature) <= feature_sel_num:
                return feature
            else:
                return feature[0:feature_sel_num]

        feature_list = sel_feature(self.feature_sel_num)
        feature = [self.dict_c2n[f] for f in feature_list]
        feature.append('label')
        self.feature = feature

        with open('./feature_sel.txt', 'w') as file:
            for f in feature_list:
                file.write(f+'\n')

        raw_data = raw_data.loc[:,feature]
        for col in raw_data.columns:
            raw_data[col] = raw_data[col].astype("float")
        X_columns_label = [x for x in raw_data.columns if x not in ['label']]
        y_columns_label = [x for x in raw_data.columns if x in ['label']]
        X = raw_data[X_columns_label]
        y = raw_data[y_columns_label]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        #Dtrain = xgb.DMatrix(X_train, label=y_train)
        Dtrain = xgb.DMatrix(X, label=y)
        Dtest = xgb.DMatrix(X_test)
        time2 = datetime.datetime.now()
        print('xgboost::train_sel_model:load data cost time=%d' % (time2 - time1).seconds)
        
        params = {'booster': 'gbtree',            # BOOST类型：GBDT
                  'objective': 'multi:softprob',  # 问题分类：四分类逻辑回归
                  'num_class':4,                  # 类别数：4类
                  'eval_metric': 'merror',        # 评价指标：merror
                  'max_depth': 5,                 # 树的深度
                  'gamma': 0,                     # 节点分裂的最小损失函数值
                  'lambda': 1,                    # 权重的L2正则化项
                  'subsample': 0.8,               # 随机选择80%样本建立决策树
                  'colsample_bytree': 0.8,        # 随机选择80%特征建立决策树
                  'min_child_weight': 1,          # 叶子节点最小权重
                  'eta': 0.1,                     # 学习率
                  'seed': 0,                      # 随机种子
                  'scale_pos_weight': 1,          # 样本平衡度
                  'nthread': 4,                   # 进程数
                  'silent': 1}                    # 打印信息
        
        watchlist = [(Dtrain, 'train')]
        xgboostModel = xgb.train(params, Dtrain, num_boost_round=200, evals=watchlist)
        time3 = datetime.datetime.now()
        print('xgboost::train_sel_model:fit train model done!cost time=%d' % (time3 - time2).seconds)
        
        ypred = xgboostModel.predict(Dtest)
        ylabel = np.argmax(ypred, axis=1)
        pickle.dump(xgboostModel, open(self.model_sel_file, 'wb'))
        time4 = datetime.datetime.now()
        print("xgboost::train_sel_model:cost time:%s,save model path:%s:"%(time4.strftime('%Y-%m-%d %H:%M:%S %f'),self.model_sel_file))
        
    def test_model(self):

        time1 = datetime.datetime.now()
        print("xgboost::test_model:start time:", time1.strftime('%Y-%m-%d %H:%M:%S %f'))
        print('xgboost::test_model:test data path=%s,model path=%s,test result path=%s,feature count=%d' %(self.test_file,self.model_sel_file,self.result_file,self.feature_sel_num))
        
        var_Features = []
        for i in range(self.feature_src_num):
            var_Features.append('f'+str(i))
        var_Features.append('label')
        raw_data = pd.read_table(self.test_file, names=var_Features, sep='\t', header=None)
        raw_data.drop(0, inplace=True)
        raw_data = raw_data.loc[:,self.feature]

        for col in raw_data.columns:
            raw_data[col] = raw_data[col].astype("float")

        X_columns_label = [x for x in raw_data.columns if x not in ['label']]
        y_columns_label = [x for x in raw_data.columns if x in ['label']]
        X = raw_data[X_columns_label]
        y = raw_data[y_columns_label]
        Ddata = xgb.DMatrix(X)
        time2 = datetime.datetime.now()
        print('xgboost::test_model:load data cost time=%d' % (time2 - time1).seconds)

        xgboostModel = pickle.load(open(self.model_sel_file, 'rb'))
        ypred = xgboostModel.predict(Ddata)
        ylabel = np.argmax(ypred, axis=1)

        print('ACC: %.4f' % metrics.accuracy_score(y, ylabel))
        print('Recall: %.4f' % metrics.recall_score(y, ylabel, average='weighted'))
        print('F1-score: %.4f' % metrics.f1_score(y, ylabel, average='weighted'))
        print('Precesion: %.4f' % metrics.precision_score(y, ylabel, average='weighted'))
        print('confusion_matrix:')
        print(confusion_matrix(y, ylabel))
        print('report:')
        print(classification_report(y, ylabel))
        time3 = datetime.datetime.now()
        print('xgboost::test_model:test cost time=%d' % (time3 - time2).seconds)
        
        idx = 0
        L = len(X)
        with codecs.open(self.result_file, 'wb+', encoding='utf-8') as f:
            while idx < L:
                f.write(str(idx)+'\t'+str(ylabel[idx])+'\t'+ str(max(ypred[idx]))+'\n')
                idx += 1
        time4 = datetime.datetime.now()
        print('xgboost::test_model:save result cost time=%d' % (time4 - time3).seconds)
        print('xgboost::test_model:test end',time4.strftime('%Y-%m-%d %H:%M:%S %f'))

def main_train():
    print('begin main!')
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", nargs='?', default='./sample.train',help="train file path")
    parser.add_argument("--model_src_file", nargs='?',default='./xgb_src.model', help="model file path")
    parser.add_argument("--model_sel_file", nargs='?',default='./xgb_sel.model', help="model file path")
    parser.add_argument("--test_file", nargs='?',default='./sample.test', help="test file path")
    parser.add_argument("--result_file", nargs='?', default='./data.res', help="result file path")
    parser.add_argument("--feature_src_num", nargs='?', default=119, help="feature src num")
    parser.add_argument("--feature_sel_num", nargs='?', default=60, help="feature sel num")
    parser.add_argument("--is_default", nargs='?', default=1, help="is default param")
    args = parser.parse_args()
    
    model = Model_XGBOOST(args.train_file,args.model_src_file,args.model_sel_file,args.test_file,args.result_file,args.feature_src_num,args.feature_sel_num,args.is_default)
    
    #训练原始模型
    model.train_src_model()
    #训练特征模型
    model.train_sel_model()
    #测试模型
    model.test_model()
    print('end main!')
    
if __name__ == "__main__":
    
    main_train()

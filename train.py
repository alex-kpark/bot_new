import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import itertools
import sklearn as sk
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

import get_data
import model
#import model_rmse
#import model_regression

#전처리한 데이터들 놓기
bot_dir_path = 'D:/AION_DATA/30_bots/' 
bot_file_list = os.listdir(bot_dir_path)

user_dir_path = 'D:/AION_DATA/30_user_sample/'
user_file_list = os.listdir(user_dir_path)

#hidden_dim : 80이 SOTA

#하이퍼파라미터

#output_dim = 1
#iterations = 1000

seq_length = 30 #뉴런의 개수 + 통계값 수
data_dim = 11 #피쳐 
n_class = 2 #BinaryS Classification
total_epochs = 10000

hidden_dim = 10 #hidden layer의 개수 #80
learning_rate = 0.001 #0.01
batch_size = 25000 #8000 #Batch 없이 하는게 효율적이지만, GPU 쓸 때 메모리 터짐 #현재 70

'''
데이터 불러오기
'''
bots = get_data.bot_generator(bot_dir_path, bot_file_list) 
print("Bots successfully Loaded")
print('\n')

users = get_data.user_generator(user_dir_path, user_file_list)
print("Users successfully Loaded")
print('\n')

#Extract Necessary Features : 필요없는 피쳐 삭제
def feature_exclude(arr):
    temp = []
    for i in arr:
        deleted = get_data.dataset_cleaner(i)
        temp.append(deleted)
    
    result = np.asarray(temp)
    return result

#Label Creation : 정답지 만드는 함수
def label_creation(label):
    
    temp = []
    if label == 1: #bot
        for i in range(len(bots)):
            temp.append([0,1])
            #temp.append(1)
        result = np.asarray(temp)
        #result = np.expand_dims(result, axis=1)
        print(result.shape)
        return result
    
    else: #user
        for j in range(len(users)):
            temp.append([1,0])
            #temp.append(0)
        result = np.asarray(temp)
        #result = np.expand_dims(result, axis=1)
        print(result.shape)
        return result      


### 통계값 추가?
#Data X
bots = feature_exclude(bots) #10703,48,11 
users = feature_exclude(users) #14481,48,11
     
#Label
bots_label = label_creation(1)
users_label = label_creation(0)

#Train, Test Split
bot_x_train, bot_x_test, bot_y_train, bot_y_test = train_test_split(bots, bots_label, test_size=0.1, shuffle=True)
user_x_train, user_x_test, user_y_train, user_y_test = train_test_split(users, users_label, test_size=0.1, shuffle=True)
train_x = np.concatenate((bot_x_train, user_x_train))
train_label = np.concatenate((bot_y_train, user_y_train))
test_x = np.concatenate((bot_x_test, user_x_test))
test_label = np.concatenate((bot_y_test, user_y_test))

train_num = len(train_x)

print("Dataset Information" + '\n')
print(train_x.shape) #(20146, 48, 11)
print(train_label.shape) #(20146, 2)
print(test_x.shape) #(5038, 48, 11)
print(test_label.shape) #(5038, 2)

#학습 및 Test 시작
model.lstm(train_x, train_label, test_x, test_label, seq_length, data_dim,
            hidden_dim, batch_size, n_class, learning_rate, total_epochs)

'''
교차검증 구성

#10-Fold 구성
KF = KFold(n_splits=10, random_state=None, shuffle=False)

train_bot, test_bot = train_test_split(bots, test_size=0.1, shuffle=False)
train_user, test_user = train_test_split(users, test_size=0.1, shuffle=False)

'''

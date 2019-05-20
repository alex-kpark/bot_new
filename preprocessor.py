import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
import math
import csv
from os import listdir
from os.path import isfile, join

data_path = 'D:/AION_DATA/weekly/'

filenames = [f for f in listdir(data_path) if isfile(join(data_path, f))]
bot_files = [] #weekly안의 봇 파일들 모음
user_files = [] #weekly안의 유저 파일들 모음

for filename in filenames:
    if 'bot' in filename:
        bot_files.append(filename)
    else:
        user_files.append(filename)

bot_save_path = 'D:/AION_DATA/30_bots/'
user_save_path = 'D:/AION_DATA/30_users/'

'''
필요한 함수들 정의
'''

def data_generator(identifier, dir_path, filename, window_size):
    master_path = dir_path + filename
    print('\n')
    print('##########################')
    print(master_path, ' has Started!')
    print('##########################')
    print('\n')
    
    #Dataframe으로 전체 봇 데이터 우선 읽고
    data = pd.read_csv(master_path)
    data = pd.DataFrame(data)

    #고유 계정 아이디 추출해서 List로 가지고 있다가
    account_list = account_extractor(data)
    
    print('\n')
    print('Number of Accounts are:', str(len(account_list)))
    print('\n')

    #시계열 Dataset 만들어서 List로 가지고 있다가
    result = timeseries(data, account_list, window_size)

    #정규화 된 값을 List로 생성
    result = norm_flat(result)

    if identifier == 'bot':
        save_file_path = bot_save_path + filename + '_processed.csv'
    else:
        save_file_path = user_save_path + filename + '_processed.csv'

    with open(save_file_path, 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(result)

    print('\n')
    print('##########################')
    print(filename + 'has Finished!')
    print('##########################')
    print('\n')


#고유한 계정 캐릭터 아이디들을 Return 해줌
def account_extractor(target_df):
    account_pool = target_df.sort_values(by='character_number', ascending=True)
    accounts = account_pool.drop_duplicates(['character_number'], keep='last')
    
    account_list = []
    for account in accounts['character_number']:
        account_list.append(account)
    
    return account_list     

#특정 계정 아이디에서 시계열로 잘라준 값을 생성해줌 (List로 Return)
def timeseries(weekly_df, account_list, window): #전체 주 데이터, 봇/유저 고유 캐릭터정보, 설정 time-window
    window_size = window
    dataset = []
    
    processed=0
    print('\n')
    print('Totally ', str(processed), '/', str(len(account_list)))

    for account in account_list:
        individual_df = weekly_df[weekly_df['character_number']==account]
        individual_df = individual_df.reset_index(drop=True)
        individual_df = individual_df.drop('log_time', axis=1)
        individual_df = individual_df.drop('character_number', axis=1)
        individual_df = individual_df.drop('log_number', axis=1)
        individual_df = individual_df.drop('account_id', axis=1)

        print(len(individual_df))
        for window in range(0, len(individual_df) - window_size):
    
            #Window 크기로 잘라주고
            sliced = individual_df[window : window + window_size]

            #통계값 추가
            sliced.loc[len(sliced)] = sliced.mean(axis=0)
            sliced.loc[len(sliced)] = sliced.std(axis=0) #하나 추가되어서 자동으로 +1 인덱스 늘어남

            sliced_np = sliced.values
            dataset.append(sliced_np)

            if window % 30 == 0:
                print(str(window), '/', str(len(individual_df) - window_size))
        processed = processed + 1

        print(str(account), ' has Finished!')
    
    return dataset #List 형태로 Return

#정규화
def MinMaxScaler(arr):
    numerator = arr - np.min(arr, 0)
    denominator = np.max(arr, 0) - np.min(arr, 0)
    return numerator / (denominator + 1e-7)

#정규화 및 Flatten 진행
def norm_flat(data_in_list):
    flatten_result = []
    for element in data_in_list:
        normalized = MinMaxScaler(element)
        flatten = normalized.flatten()
        flatten_result.append(flatten)
        
    return flatten_result


#봇 수집 -> 유저 할 때는 유저로 바꾸기만 하면 됨

window_size = 30
for bot_file in bot_files:
    data_generator('bot', data_path, bot_file, window_size)

for user_file in user_files:
    data_generator('user', data_path, user_file, window_size)
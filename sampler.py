import pandas as pd
import numpy as np
import os
import csv

path = 'D:/AION_DATA/stat_user/'
save_dir = 'D:/AION_DATA/weekly_user_sampled/'
filelist = os.listdir(path)


def sampler(dir_path, filename, sample_size):

    print(filename)
    master_path = dir_path + filename

    print('\n')
    print("###################")
    print(master_path, ' has started sampling process')
    print("###################")

    temp = []
    for i in range(0, 11*60):
        temp.append(str(i))

    data = pd.read_csv(master_path, names=temp)
    data = pd.DataFrame(data)

    sample = data.sample(sample_size)
    sample = sample.reset_index(drop=True)
    #sample = sample.values
    #sample_container.append(sample)

    return sample

def flat(data_in_list):
    flatten_result = []
    for element in data_in_list:
        flatten = element.flatten()
        flatten_result.append(flatten)
    return flatten_result


def sample_generator():
    for filename in filelist:
        print(filename)
        save_path = save_dir + filename + '_sample.csv'

        sample = sampler(path, filename, 40000)
        
        sample.to_csv(save_path, mode='w', header=False, index=False)

        '''
        flatten_list = flat(sample_container)

        with open(save_path, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(flatten_list)
        '''
        print(save_path + ' Has Sampled!')

sample_generator()
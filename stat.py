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

from collections import Counter

bot_dir_path = 'D:/AION_DATA/stat_bot/' 
bot_file_list = os.listdir(bot_dir_path)

user_dir_path = 'D:/AION_DATA/stat_user_sample/'
user_file_list = os.listdir(user_dir_path)

bots = get_data.bot_generator(bot_dir_path, bot_file_list) 
users = get_data.user_generator(user_dir_path, user_file_list)

container = []

print(len(bots))
for i in range(len(bots)):

    bot = bots[i]
    user = users[i]

    #bot_mean = np.mean(bot, axis=0)
    #user_mean = np.mean(user, axis=0)

    bot_mean = np.std(bot, axis=0)
    user_mean = np.std(user, axis=0)

    temp = []
    for j in range(len(bot_mean)):
        if bot_mean[j] == user_mean[j]:
            #if j > 11:
            #    pass
            #else:
            temp.append(j)
            container.append(j)
        else:
            pass

print(Counter(container))

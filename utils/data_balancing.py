import numpy as np
import pandas as pd
from tqdm import tqdm

"""
@author (Brian) JoonHo Lee

Our data has classes [0,4], yet there is significant class imbalance. To resolve
this, model is pretrained on 2019 data with classes balanced by data from 2015
competition. e.g. classes 1,3,4 that are low in frequency received more samples.
In this script data from 2015 are randomly selected to resolve the class
imbalance, and save the labels to a new CSV file.
"""

balancing_limit = 2000
old_df = pd.read_csv("/nas-homes/joonl4/blind_2015/trainLabels.csv")
old_df2 = pd.read_csv("/nas-homes/joonl4/blind_2015/retinopathy_solution.csv")
old_df2 = old_df2.drop(["Usage"], axis = 1)
old = pd.concat([old_df, old_df2], axis = 0, sort = False)
old_df = old_df.reset_index(drop = True)
old_df['image'] += '.jpeg'
new_df = pd.read_csv("/nas-homes/joonl4/blind/train.csv")
new_df['id_code'] += '.png'
print("2019 DATA:")
print("    class 0: %d" % len(new_df.loc[new_df['diagnosis'] == 0]))
print("    class 1: %d" % len(new_df.loc[new_df['diagnosis'] == 1]))
print("    class 2: %d" % len(new_df.loc[new_df['diagnosis'] == 2]))
print("    class 3: %d" % len(new_df.loc[new_df['diagnosis'] == 3]))
print("    class 4: %d" % len(new_df.loc[new_df['diagnosis'] == 4]))
print(new_df['diagnosis'].value_counts())
print("2015 DATA:")
print("    class 0: %d" % len(old_df.loc[old_df['level'] == 0]))
print("    class 1: %d" % len(old_df.loc[old_df['level'] == 1]))
print("    class 2: %d" % len(old_df.loc[old_df['level'] == 2]))
print("    class 3: %d" % len(old_df.loc[old_df['level'] == 3]))
print("    class 4: %d" % len(old_df.loc[old_df['level'] == 4]))
print(old_df['level'].value_counts())

print("balancing data to meet %d images per class" % balancing_limit)
balanced_df = new_df
for i in range(5):
    current_count = len(new_df[(new_df.diagnosis == i)].index)
    maximum = len(old_df[old_df.level == i].index)
    add = balancing_limit - current_count
    add = np.min([add, maximum])
    print("filling %d rows" % add)
    balancer = old_df[(old_df.level == i)]
    balancer = balancer.reset_index(drop = True)
    balancer = balancer.loc[:add - 1]
    balancer = balancer.rename(
        columns={"image": "id_code", "level": "diagnosis"})
    balanced_df = pd.concat([balanced_df, balancer], axis = 0, sort = False)
    balanced_df = balanced_df.reset_index(drop = True)

print("Balanced DATA")
print(balanced_df['diagnosis'].value_counts())
print(balanced_df.head())
balanced_df.to_csv("/nas-homes/joonl4/blind/train_balanced.csv")
print("balanced data list generated!")
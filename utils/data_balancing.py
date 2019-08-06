############################################
# Author Brian Lee

# Reads 2015 competition data and 2019 data
# Balances 2019 data with more 2,3,4 classes
############################################

import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
gc.enable()
gc.collect()

old_df = pd.read_csv("/nas-homes/joonl4/blind_2015/trainLabels.csv")
old_df2 = pd.read_csv("/nas-homes/joonl4/blind_2015/retinopathy_solution.csv")
old_df2 = old_df2.drop(["Usage"], axis = 1)
old = pd.concat([old_df, old_df2], axis = 0, sort = False)
old_df.reset_index(drop = True)

# class_2_idx = old_df.loc[old_df['level']==2]
# class_3_idx = old_df.loc[old_df['level']==3]
# class_4_idx = old_df.loc[old_df['level']==4]

new_df = pd.read_csv("/nas-homes/joonl4/blind/train.csv")
old_df = old_df.astype(str)
new_df = new_df.astype(str)

print("2019 DATA:")
# print("    class 0: %d" % len(new_df.loc[new_df['diagnosis'] == 0]))
# print("    class 1: %d" % len(new_df.loc[new_df['diagnosis'] == 1]))
# print("    class 2: %d" % len(new_df.loc[new_df['diagnosis'] == 2]))
# print("    class 3: %d" % len(new_df.loc[new_df['diagnosis'] == 3]))
# print("    class 4: %d" % len(new_df.loc[new_df['diagnosis'] == 4]))
print(new_df['diagnosis'].value_counts())
print("2015 DATA:")
# print("    class 0: %d" % len(old_df.loc[old_df['level'] == 0]))
# print("    class 1: %d" % len(old_df.loc[old_df['level'] == 1]))
# print("    class 2: %d" % len(old_df.loc[old_df['level'] == 2]))
# print("    class 3: %d" % len(old_df.loc[old_df['level'] == 3]))
# print("    class 4: %d" % len(old_df.loc[old_df['level'] == 4]))
print(old_df['level'].value_counts())
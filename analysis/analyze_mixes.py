import os
import numpy as np
import pandas as pd
import matplotlib as mpl 
mpl.use('agg')
import matplotlib.pyplot as plt

la = pd.read_csv("../data/level_analysis.csv")
train_la = la.loc[la['type'] == 'train']

train_br = train_la['bass ratio'].tolist()
train_dr = train_la['drums ratio'].tolist()
train_or = train_la['other ratio'].tolist()
train_vr = train_la['vocals ratio'].tolist()

train_data_ratio = [train_br, train_dr, train_or, train_vr]

print("Analyzed {} tracks".format(len(train_br)))
mean = np.mean(train_data_ratio, axis=1)
std = np.std(train_data_ratio, axis=1)
print("Bass   LR: {0:0<18} {1:0<18}".format(mean[0], std[0]))
print("Drums  LR: {0:0<18} {1:0<18}".format(mean[1], std[1]))
print("Other  LR: {0:0<18} {1:0<18}".format(mean[2], std[2]))
print("Vocals LR: {0:0<18} {1:0<18}".format(mean[3], std[3]))
import os
import sys
import glob
import numpy as np
import pandas as pd

# load level analysis - run analyze.py first
levels = pd.read_csv('level_analysis.csv')

# list to store mix ratios
mix_ratios = []

# load audio data from each tracks
for row_idx, row in levels.iterrows():
	sys.stdout.write("* Loading track {:0>3}\r".format(row['id']))
	sys.stdout.flush()
	mix_ratios.append(row['mix ratio'])

se = []
for idx, y in enumerate(mix_ratios):
	train_tracks = list(mix_ratios)
	del train_tracks[idx]

	y_hat = np.mean(train_tracks)
	sigma = np.std(train_tracks)
	#print(y_hat, sigma)
	se.append(np.square(y - y_hat))
	print(np.square(y - y_hat))

mse = np.mean(se)
print(mse)


import os
import sys
import glob
import numpy as np
import soundfile as sf
import pandas as pd
import time

def load(level_analysis_path='level_analysis.csv', framing=False, frame_length=30.0):

	# training data 
	X = []
	Y = []

	# load level analysis - run analyze.py first
	levels = pd.read_csv(level_analysis_path)

	# load audio data from each tracks
	for row_idx, row in levels.iterrows():
		sys.stdout.write("* Loading track {:0>3}\r".format(row['id']))
		sys.stdout.flush()

		accomp_wav = os.path.join(row['path'], 'accomp_m16k_norm.wav')
		vocals_wav = os.path.join(row['path'], 'vocals_m16k_norm.wav')
		accomp_data, rate = sf.read(accomp_wav)
		vocals_data, rate = sf.read(vocals_wav)

		track_data = []
		frame_size = int(np.floor(frame_length * rate))
		n_frames = int(np.floor(accomp_data.shape[0]/(frame_size)))

		if framing:
			for frame_idx in range(n_frames):
				l = int(frame_idx * (frame_size))
				u = int(l + frame_size)
				track_data.append(np.stack((accomp_data[l:u], vocals_data[l:u]), axis=0))			
		else:
			track_data.append(np.stack((accomp_data[:frame_size], vocals_data[:frame_size]), axis=0))

		X.append(np.array(track_data))
		Y.append(row['mix ratio'])

	sys.stdout.write("* {} tracks loaded    \n".format(row_idx+1))
	return X, Y

def profile():
	print("* No framing")
	start = time.time()
	X, Y = load(framing=False)
	end = time.time() - start
	print(end)
	for track, mix in zip(X, Y):
		print(track.shape)
		print(mix)

	print("* With framing")
	start = time.time()
	X, Y = load(framing=True)
	end = time.time() - start
	print(end)
	for track, mix in zip(X, Y):
		print(track.shape)
		print(mix)

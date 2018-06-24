import os
import sys
import numpy as np
import models
import util

#models.build_cfm_model((1, 349200), 160000, 256, 256, summary=True)
#models.build_sb_model((2, 8.15*16000), 160000, n_dft=1024, n_hop=1024, summary=True)
#models.test_model((128, 128, 1), 160000, summary=True)
#sys.exit()

# selected hyperparameters
batch_size = 1
epochs = 400
n_dft = 1024
n_hop = 1024
model_type = 'sb' 	# 'sb' or 'cfm'
train = 'single' 	# 'single' or 'k-fold'

# for cfm size use frame_length=21.825 - n_dft=256, n_hop=256
# for sb size use frame_length=8.15 - n_dft=1024, n_hop=1024

X, Y = util.load(frame_length=8.15, framing=True)

if   train == 'k-fold':
	for idx, val_track in enumerate(X):
		train_tracks = list(X)
		del train_tracks[idx]

		X_train = np.array([frame for track in train_tracks for frame in track])
		Y_train = np.array([Y[1+idx] for idx, track in enumerate(train_tracks) for frame in track])
		X_val   = val_track
		Y_val   = np.array([Y[0] for frame in val_track])

		print("* Fold {}".format(idx))
		print(X_train.shape)
		print(Y_train.shape)
		print(X_val.shape)
		print(Y_val.shape)

elif train == 'single':

	X_train = np.array([frame for track in X[1:] for frame in track])
	Y_train = np.array([Y[1+idx] for idx, track in enumerate(X[1:]) for frame in track])
	X_val   = X[0]
	Y_val   = np.array([Y[0] for frame in X[0]])

	del X, Y

	print(X_train.shape)
	print(Y_train.shape)
	print(X_val.shape)
	print(Y_val.shape)

	model = models.build_sb_model((X_train.shape[1], X_train.shape[2]), 
										16000, n_dft, n_hop, summary=True)

	history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=True,
                        validation_data=(X_val, Y_val), 
                        shuffle=True)
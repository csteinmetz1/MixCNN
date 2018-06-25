import os
import gc
import sys
import argparse
import datetime
import numpy as np
from keras import backend as K
# custom modules
import models
import util

# setup argpase
parser = argparse.ArgumentParser(description='MixCNN training parameters')
parser.add_argument('-m', type=str, help='message to save with report')
args = parser.parse_args()

# selected hyperparameters
batch_size = 1
epochs = 45
n_dft = 1024
n_hop = 1024
frame_length = 8.15
framing = False
model_type = 'sb' 	# 'sb' or 'cfm'
train = 'single' 	# 'single' or 'k-fold'

# create directory to store outputs
report_dir = util.setup_report_directory()

# for cfm size use frame_length=21.825 - n_dft=256, n_hop=256
# for sb size use frame_length=8.15 - n_dft=1024, n_hop=1024

X, Y, index_list = util.load(frame_length=frame_length, framing=framing)
n_tracks = len(X)

if   train == 'k-fold':
    start_time = datetime.datetime.today() # get time
    fold_history = [] # store training history objects
    for idx in range(n_tracks):

        X_train = np.array([frame for track in X[idx+1:] + X[:idx] for frame in track])
        Y_train = np.array([Y[1+idx] for idx, track in enumerate(X[idx+1:] + X[:idx]) for frame in track])
        X_val   = X[idx]
        Y_val   = np.array([Y[idx] for frame in X[idx]])

        print("* Fold {}".format(idx+1))
        print(X_train.shape)
        print(Y_train.shape)
        print(X_val.shape)
        print(Y_val.shape)

        input_shape = (X_train.shape[1], X_train.shape[2])

        # build the model
        model = models.build_sb_model(input_shape, 
                                    16000, 
                                    n_dft, 
                                    n_hop, 
                                    summary=False)

        # train the network
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=True,
                            validation_data=(X_val, Y_val), 
                            shuffle=True)

        fold_history.append(history)
        K.clear_session()
        del X_train, Y_train, X_val, Y_val

    end_time = datetime.datetime.today()
    
    r = ({'history' : fold_history,
          'index list' : index_list, 
          'start time' : start_time,
          'end time' : end_time,
          'elapsed time' : end_time-start_time,
          'batch size' : batch_size,
          'epochs' : epochs,
          'input shape' : input_shape,
          'train' : train,
          'model type' : model_type,
          'folds' : n_tracks,
          'learning rate' : 0.001,
          'n_dft' : n_dft,
          'n_hop' : n_hop,
          'framing' : framing,
          'frame length' : frame_length,
          'model' : model})

    util.generate_report(report_dir, r, msg=args.m)

elif train == 'single':

    #X, Y, index_list = util.load(frame_length=frame_length, framing=framing)

    X_train = np.array([frame for track in X[1:] for frame in track])
    Y_train = np.array([Y[1+idx] for idx, track in enumerate(X[1:]) for frame in track])
    X_val   = X[0]
    Y_val   = np.array([Y[0] for frame in X[0]])

    del X, Y # clean up

    print(X_train.shape)
    print(Y_train.shape)
    print(X_val.shape)
    print(Y_val.shape)

    input_shape = (X_train.shape[1], X_train.shape[2])

    # build the model
    model = models.build_sb_model(input_shape, 
                                16000, 
                                n_dft, 
                                n_hop, 
                                summary=True)

    start_time = datetime.datetime.today()
    # train the network
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=True,
                        validation_data=(X_val, Y_val), 
                        shuffle=True)
    end_time = datetime.datetime.today()
    
    r = ({'history' : [history],
          'index list' : index_list, 
          'start time' : start_time,
          'end time' : end_time,
          'elapsed time' : end_time-start_time,
          'batch size' : batch_size,
          'epochs' : epochs,
          'input shape' : input_shape,
          'train' : train,
          'model type' : model_type,
          'folds' : 1,
          'learning rate' : 0.001,
          'n_dft' : n_dft,
          'n_hop' : n_hop,
          'framing' : framing,
          'frame length' : frame_length,
          'model' : model})

    util.generate_report(report_dir, r, msg=args.m)
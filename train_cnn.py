from __future__ import print_function
import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import losses
from keras import optimizers
from keras.constraints import maxnorm
from sklearn.model_selection import KFold
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import os
import gc
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

def get_date_and_time():
    date_and_time = str(datetime.now()).split(' ')
    date = date_and_time[0]
    time = ('-').join(date_and_time[1].split(':')[0:2])
    return date, time

def load_data(spect_type='mel', spect_size='sm', framing=False, shuffle=True):

    key = "{0} {1}".format(spect_type, spect_size)
    size = 20 # length of input to analyze

    y_rows = []
    x_rows = []

    if framing:
        lim = -30
        discarded = 0
        for idx, song in enumerate(glob.glob("data/*.pkl")):
            row = pickle.load(open(song, "rb"))
            n_frames = np.floor((row['bass ' + key].shape[1])/size).astype('int')
            for frame in range(n_frames):
                start_idx = frame*size
                end_idx = start_idx+size

                bass_spect = row['bass ' + key][:, start_idx:end_idx]
                drums_spect = row['drums ' + key][:, start_idx:end_idx]
                other_spect = row['other ' + key][:, start_idx:end_idx]
                vocals_spect = row['vocals ' + key][:, start_idx:end_idx]

                b_mean = np.mean(bass_spect, axis=(0,1))
                d_mean = np.mean(drums_spect, axis=(0,1))
                o_mean = np.mean(other_spect, axis=(0,1))
                v_mean = np.mean(vocals_spect, axis=(0,1))

                if b_mean > lim and b_mean > lim and o_mean > lim and v_mean > lim:
                    x_rows.append(np.dstack((bass_spect, drums_spect, other_spect, vocals_spect)))
                    y_rows.append(np.array((row['drums ratio'], row['other ratio'], row['vocals ratio'])))
                else:
                    discarded += 1
        print("Discarded {0:d} frames with energy below the threshold.".format(discarded))
    else:
        for idx, song in enumerate(glob.glob("data/*.pkl")):
            row = pickle.load(open(song, "rb"))
            y_rows.append(np.array((row['drums ratio'], row['other ratio'], row['vocals ratio'])))
            bass_spect = row['bass ' + key][:, :size]
            drums_spect = row['drums ' + key][:, :size]
            other_spect = row['other ' + key][:, :size]
            vocals_spect = row['vocals ' + key][:, :size]
            x_rows.append(np.dstack((bass_spect, drums_spect, other_spect, vocals_spect)))

    # transform into numpy arrays
    Y = np.array([row for row in y_rows])
    X = np.array([row for row in x_rows])

    # remove nans
    X = np.nan_to_num(X)

    # standardize inputs
    X -= np.mean(X, axis = 0) # zero-center
    X /= np.std(X, axis = 0) # normalize

    if shuffle:
        np.random.seed(42)
        state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(state)
        np.random.shuffle(Y)

    input_shape = (X.shape[1], X.shape[2], 4) # four instruments - 1 per channel

    print("Loaded inputs with shape:", X.shape)
    print("Loaded outputs with shape:", Y.shape)

    return X, Y, input_shape

def build_model(input_shape, summary=False):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(3))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam())

    if summary:
        model.summary()

    return model

def build_model_exp(input_shape, lr, summary=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
    #model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model.add(Dropout(0.2))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(2056, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(3, activation='linear'))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(lr=lr))

    if summary:
        model.summary()

    return model

def build_model_biggest(input_shape, lr, summary=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
    #model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    #model.add(Dropout(0.2))
    model.add(Dense(2056, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.2))
    model.add(Dense(3, activation='linear'))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(lr=lr))

    if summary:
        model.summary()

    return model

def train_and_eval_model(model, X_train, Y_train, X_test, Y_test, batch_size, epochs, show_pred=True, save_weights=False):

    # checkpoint to save weights
    filepath='checkpoints/checkpoint-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    
    if save_weights:
        history = model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, Y_test),
            callbacks=[checkpoint])
    else:
        history = model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score)

    if show_pred:
        pred = model.predict(X_test)
        print("We expect:", Y_test)
        print("we predict:", pred)

    return history, score

if __name__ == "__main__":

    batch_size = 1000
    epochs = 100
    lr = 0.001
    train = 'single'
    n_folds = 0
    spect_type = 'mfcc'
    spect_size = 'lg'
    X, Y, input_shape = load_data(spect_type=spect_type, spect_size=spect_size, framing=True)

    # get the start date and time and format it
    date, time = get_date_and_time()
    start_days = date.split('-')[2]
    start_hrs = time.split('-')[0]
    start_mins = time.split('-')[1]

    training_history = {}
    saved_model = None

    if train == 'k fold':
        kf = KFold(n_splits=n_folds)
        split = int(X.shape[0] - X.shape[0]/n_folds)
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            print("Running fold", i+1, "of", n_folds)
            model = build_model_larger(input_shape, lr)
            history, score = train_and_eval_model(model, X[train_index], Y[train_index], X[test_index], Y[test_index], batch_size, epochs)
            training_history[i] = {'score' : score, 'loss': history.history['loss'], 'val_loss' : history.history['val_loss']}
            if i == 0:
                saved_model = model
            K.clear_session()
            del history
            del model
            gc.collect()
    if train == 'single':
        split = int(np.floor(X.shape[0]*0.80))
        model = build_model_exp(input_shape, lr)
        print(X[:split, :].shape, Y[:split, :].shape, X[split:, :].shape, Y[split:, :].shape)
        history, score = train_and_eval_model(model, X[:split, :], Y[:split, :], X[split:, :], Y[split:, :], batch_size, epochs)
        model.save('final_model_loss_{0:f}.hdf5'.format(score))
        training_history[0] = {'score' : score, 'loss': history.history['loss'], 'val_loss' : history.history['val_loss']}
        saved_model = model
        K.clear_session()
        del history
        del model
        gc.collect()

    # get the end date and time and format it
    end_date, end_time = get_date_and_time()
    end_days = end_date.split('-')[2]
    end_hrs = end_time.split('-')[0]
    end_mins = end_time.split('-')[1]
    
    elp_hrs = np.abs(int(end_hrs) - int(start_hrs))
    elp_mins = np.abs(int(end_mins) - int(start_mins))

    if not os.path.isdir(os.path.join("reports", "{0}--{1}".format(date, time))):
        os.makedirs(os.path.join("reports", "{0}--{1}".format(date, time)))

    # Save training results to file
    with open(os.path.join("reports", "{0}--{1}".format(date, time), "report_summary.txt"), 'w') as results:
        results.write("--- RUNTIME ---\n")
        results.write("Start time: {0} at {1}\n".format(date, time))
        results.write("End time:   {0} at {1}\n".format(end_date, end_time))
        results.write("Runtime:    {0:d} hrs {1:d} mins\n\n".format(elp_hrs, elp_mins))
        results.write("--- MSE RESULTS ---\n")
        for i, stats in training_history.items():
            results.write("For fold {0:d} - Test loss: {1:0.4f} MSE\n".format(i+1, stats['score']))
        mean = np.mean([fold['score'] for i, fold in training_history.items()])
        std = np.std([fold['score'] for i, fold in training_history.items()])
        results.write("Average test loss: {0:0.4f} ({1:0.4f}) MSE\n\n".format(mean, std))
        results.write("--- TRAINING DETAILS ---\n")
        results.write("Batch size:  {0}\n".format(batch_size))
        results.write("Epochs:      {0}\n".format(epochs))
        results.write("Input shape: {0}\n".format(input_shape))
        results.write("Training type:  {0}\n".format(train))
        results.write("Folds:          {0:d}\n".format(n_folds))
        results.write("Training split: {0:d}/{1:d}\n".format(split, X.shape[0]-split))
        results.write("Learning rate:  {0:f}\n".format(lr))
        results.write("Spectrogram type: {0}\n".format(spect_type))
        results.write("Spectrogram size: {0}\n".format(spect_size))
        results.write("\n--- NETWORK ARCHITECTURE ---\n")
        saved_model.summary(print_fn=lambda x: results.write(x + '\n'))

    pickle.dump(training_history, open(os.path.join("reports", "{0}--{1}".format(date, time), "training_history.pkl"), "wb"), protocol=2)
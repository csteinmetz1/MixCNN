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
import os
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

def load_data():
    # import mel spectrogram data
    spectral_analysis = pickle.load(open("spectral_analysis.pkl", "rb"))

    # sort out train and test data
    x_train_rows = [row for row in spectral_analysis if row['type'] == 'Dev']
    x_test_rows = [row for row in spectral_analysis if row['type'] == 'Test']

    # stack subgroup spectrograms as channels
    dstack = True 

    if dstack:
        x_train = np.array([np.dstack((row['bass spect'], row['drums spect'], row['other spect'], row['vocals spect'])) for row in x_train_rows])
        x_test = np.array([np.dstack((row['bass spect'], row['drums spect'], row['other spect'], row['vocals spect'])) for row in x_test_rows])
        input_shape = (x_train.shape[1], x_train.shape[2], 4)
    else:
        x_train = np.array([np.hstack((row['bass spect'], row['drums spect'], row['other spect'], row['vocals spect'])) for row in x_train_rows])
        x_test = np.array([np.hstack((row['bass spect'], row['drums spect'], row['other spect'], row['vocals spect'])) for row in x_test_rows])
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
        input_shape = (x_train.shape[1], x_train.shape[2], 1)

    # import output (level anaylsis) data
    level_analysis = pd.read_csv("level_analysis.csv")

    # sort out train and test data
    y_train_rows = level_analysis.loc[level_analysis['type'] == 'Dev']
    y_test_rows = level_analysis.loc[level_analysis['type'] == 'Test']

    # crate array of length 3 - exclude the bass loudness ratio
    y_train = np.array([np.array([row[1]['drums ratio'], row[1]['other ratio'], row[1]['vocals ratio']]) for row in y_train_rows.iterrows()])
    y_test = np.array([np.array([row[1]['drums ratio'], row[1]['other ratio'], row[1]['vocals ratio']]) for row in y_test_rows.iterrows()])

    # create lists of total dataset
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))

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

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adadelta())

    if summary:
        model.summary()

    return model

def build_model_larger(input_shape, summary=False):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.1))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(3))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam())

    if summary:
        model.summary()

    return model

def train_and_eval_model(model, X_train, Y_train, X_test, Y_test, show_pred=True, save_weights=False):

    batch_size = 10
    epochs = 10

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
        print("We expect:", Y_test[0])
        print("we predict:", pred[0])

    return history, score

if __name__ == "__main__":
    n_folds = 2
    X, Y, input_shape = load_data()
    kf = KFold(n_splits=n_folds)

    # get the start date and time and format it
    date, time = get_date_and_time()
    start_days = date.split('-')[2]
    start_hrs = time.split('-')[0]
    start_mins = time.split('-')[1]

    training_history = {}
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print("Running fold", i+1, "of", n_folds)
        model = build_model_larger(input_shape)
        history, score = train_and_eval_model(model, X[train_index], Y[train_index], X[test_index], Y[test_index])
        training_history[i] = {'score' : score, 'loss': history.history['loss'], 'val_loss' : history.history['val_loss']}
        del history
        del model
        gc.collect()

    # get the end date and time and format it
    end_date, end_time = get_date_and_time()
    end_days = end_date.split('-')[2]
    end_hrs = end_time.split('-')[0]
    end_mins = end_time.split('-')[1]
    
    elp_days = int(end_days) - int(start_days)
    elp_hrs = int(end_hrs) - int(start_hrs)
    elp_mins = int(end_mins) - int(start_mins)

    if not os.path.isdir(os.path.join("reports", "{0}--{1}".format(date, time))):
        os.makedirs(os.path.join("reports", "{0}--{1}".format(date, time)))

    # Save training results to file
    with open(os.path.join("reports", "{0}--{1}".format(date, time), "report_summary.txt"), 'w') as results:
        results.write("--- RUNTIME ---\n")
        results.write("Start time: {0} at {1}\n".format(date, time))
        results.write("End time:   {0} at {1}\n".format(end_date, end_time))
        results.write("Runtime:    {0:d} days {0:d} hrs {1:d} mins\n\n".format(elp_days, elp_hrs, elp_mins))
        results.write("--- MSE RESULTS ---\n")
        for i, stats in training_history.items():
            results.write("For fold {0:d} - Test loss: {1:0.4f} MSE\n".format(i+1, stats['score']))
        mean = np.mean([fold['score'] for i, fold in training_history.items()])
        std = np.std([fold['score'] for i, fold in training_history.items()])
        results.write("Average test loss: {0:0.4f} ({1:0.4f}) MSE\n".format(mean, std))
        results.write("\n--- NETWORK ARCHITECTURE ---\n")
        model.summary(print_fn=lambda x: results.write(x + '\n'))

    pickle.dump(training_history, open(os.path.join("reports", "{0}--{1}".format(date, time), "training_history.pkl"), "wb"), protocol=2)
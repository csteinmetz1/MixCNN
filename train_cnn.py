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
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import pickle

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

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # import output (level anaylsis) data
    level_analysis = pd.read_csv("level_analysis.csv")

    # sort out train and test data
    y_train_rows = level_analysis.loc[level_analysis['type'] == 'Dev']
    y_test_rows = level_analysis.loc[level_analysis['type'] == 'Test']

    # crate array of length 4 arrays
    y_train = np.array([np.array([row[1]['bass ratio'], row[1]['drums ratio'], row[1]['other ratio'], row[1]['vocals ratio']]) for row in y_train_rows.iterrows()])
    y_test = np.array([np.array([row[1]['bass ratio'], row[1]['drums ratio'], row[1]['other ratio'], row[1]['vocals ratio']]) for row in y_test_rows.iterrows()])

    # create lists of total dataset
    X = np.concatenate((x_train, x_test))
    Y = np.concatenate((y_train, y_test))

    return X, Y, input_shape

def build_model(input_shape, summary=False):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(4))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adadelta(), metrics=['accuracy'])

    if summary:
        model.summary()

    return model

def train_and_eval_model(model, X_train, Y_train, X_test, Y_test, save_weights=False):

    batch_size = 10
    epochs = 250

    # checkpoint to save weights
    filepath='checkpoints/checkpoint-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    
    if save_weights:
        model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, Y_test),
            callbacks=[checkpoint])
    else:
        model.fit(X_train, Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return score

if __name__ == "__main__":
    n_folds = 20
    X, Y, input_shape = load_data()
    kf = KFold(n_splits=n_folds)

    training_scores = {}
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print("Running fold", i+1, "of", n_folds)
        model = None # clear the model
        model = build_model(input_shape)
        score = train_and_eval_model(model, X[train_index], Y[train_index], X[test_index], Y[test_index])
        training_scores[i] = score

    total_loss = 0
    total_acc = 0
    for i, score in training_scores.items():
        print("For fold", i+1, "Test loss:", score[0], "and Test accuracy:", score[1])
        total_loss += score[0]
        total_acc += score[1] 
    
    print("Average test loss:", total_loss/n_folds)
    print("Average test accuracy:", total_acc/n_folds)


    



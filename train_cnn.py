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

batch_size = 2
epochs = 100

# stack subgroup spectrograms as channels
dstack = True 
# import mel spectrogram data
spectral_analysis = pickle.load(open("spectral_analysis.pkl", "rb"))

# sort out train and test data
x_train_rows = [row for row in spectral_analysis if row['type'] == 'Dev']
x_test_rows = [row for row in spectral_analysis if row['type'] == 'Test']

# stack spectrograms side by side
if dstack:
    x_train = np.array([np.dstack((row['bass spect'], row['drums spect'], row['other spect'], row['vocals spect'])) for row in x_train_rows])
    x_test = np.array([np.dstack((row['bass spect'], row['drums spect'], row['other spect'], row['vocals spect'])) for row in x_test_rows])
else:
    x_train = np.array([np.hstack((row['bass spect'], row['drums spect'], row['other spect'], row['vocals spect'])) for row in x_train_rows])
    x_test = np.array([np.hstack((row['bass spect'], row['drums spect'], row['other spect'], row['vocals spect'])) for row in x_test_rows])
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    
spect_rows, spect_cols = x_train.shape[1], x_train.shape[2]
input_shape = (spect_rows, spect_cols, 4)

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

# build network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(4))

model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adadelta(), metrics=['accuracy'])
model.summary()

# checkpoint to save weights
filepath='checkpoints/checkpoint-{epoch:02d}-{loss:.4f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

# create lists of total dataset
X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))

# K fold x-validation with 10 folds
kf = KFold(n_splits=10)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    print("Train:", train_index, "Test:", test_index)
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[checkpoint])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])



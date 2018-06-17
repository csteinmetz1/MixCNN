import keras
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras import losses
from keras import optimizers
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D

def build_kapre_model(input_shape, sr, lr, summary=False):

    model = Sequential()
    model.add(Melspectrogram(n_dft=256, n_hop=256, input_shape=input_shape,
                         padding='same', sr=sr, n_mels=96,
                         fmin=0.0, fmax=sr/2, power_melgram=1.0,
                         return_decibel_melgram=False, trainable_fb=False,
                         trainable_kernel=False,
                         name='mel'))
    model.add(SeparableConv2D(128, (3,3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 4)))
    model.add(Dropout(0.5))
    model.add(SeparableConv2D(384, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4,5)))
    model.add(Dropout(0.5))
    model.add(SeparableConv2D(768, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3,8)))
    model.add(Dropout(0.5))
    model.add(SeparableConv2D(2048, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4,8)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(3, activation='linear'))
    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(lr=lr))

    if summary:
        model.summary()

    return model

def build_model_SB(input_shape, lr, summary=False):
    model = Sequential()
    model.add(SeparableConv2D(24, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(SeparableConv2D(48, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(SeparableConv2D(48, (5, 5), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='linear'))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(lr=lr))

    if summary:
        model.summary()

    return model

def build_model(input_shape, lr, summary=False):
    model = Sequential()
    model.add(Conv2D(24, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(24, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='relu'))
    model.add(Conv2D(48, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(96, (5, 5), activation='relu'))
    model.add(Conv2D(96, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='linear'))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(lr=lr))

    if summary:
        model.summary()

    return model

def build_model_large(input_shape, lr, summary=False):
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

build_kapre_model((4, 480000), 16000, 0.001, summary=True)
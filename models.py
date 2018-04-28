import keras
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, SeparableConv2D
from keras import backend as K
from keras import losses
from keras import optimizers

def build_model_mfcc(input_shape, lr, summary=False):
    model = Sequential()
    model.add(SeparableConv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(SeparableConv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SeparableConv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(3, activation='linear'))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(lr=lr))

    if summary:
        model.summary()

    return model

def build_model_mel(input_shape, lr, summary=False):
    model = Sequential()
    model.add(SeparableConv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SeparableConv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SeparableConv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SeparableConv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(3, activation='linear'))

    model.compile(loss=losses.mean_squared_error, optimizer=optimizers.Adam(lr=lr))

    if summary:
        model.summary()

    return model

def build_model(input_shape, lr, summary=False):
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
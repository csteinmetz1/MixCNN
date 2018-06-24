from keras.models import Sequential
from keras.layers import SeparableConv2D, Activation, Dropout, BatchNormalization, MaxPooling2D, Dense, Flatten
from keras.layers import SeparableConv1D, MaxPooling1D
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from kapre.time_frequency import Melspectrogram
from kapre.utils import Normalization2D

def build_cfm_model(input_shape, rate, n_dft, n_hop, summary=False):
    
    model = Sequential()
    model.add(Melspectrogram(n_dft=n_dft, n_hop=n_hop, input_shape=input_shape,
                         padding='same', sr=rate, n_mels=96,
                         fmin=0.0, fmax=rate/2, power_melgram=1.0,
                         return_decibel_melgram=False, trainable_fb=False,
                         trainable_kernel=False,
                         name='mel'))
    model.add(SeparableConv2D(128, (3,3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,4)))
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
    model.add(Dense(1))
    model.compile(loss=mean_squared_error, optimizer=Adam())

    if summary:
        model.summary()

    return model

def build_ds_model(input_shape, rate, n_dft, n_hop, summary=False):
    model = Sequential()
    model.add(Melspectrogram(n_dft=n_dft, n_hop=n_hop, input_shape=input_shape,
                        padding='same', sr=rate, n_mels=96,
                        fmin=0.0, fmax=rate/2, power_melgram=1.0,
                        return_decibel_melgram=False, trainable_fb=False,
                        trainable_kernel=False,
                        name='mel'))
    model.add(SeparableConv1D(32, (8), activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=(4)))
    model.add(SeparableConv1D(32, (8), activation='relu'))
    model.add(MaxPooling1D(pool_size=(4)))
    model.add(Dense(100))
    model.add(Dense(1))
    model.compile(loss=mean_squared_error, optimizer=Adam())

    if summary:
        model.summary()

    return model

def build_sb_model(input_shape, rate, n_dft, n_hop, summary=False):
    model = Sequential()
    model.add(Melspectrogram(n_dft=n_dft, n_hop=n_hop, input_shape=input_shape,
                        padding='same', sr=rate, n_mels=128,
                        fmin=0.0, fmax=rate/2, power_melgram=1.0,
                        return_decibel_melgram=False, trainable_fb=False,
                        trainable_kernel=False,
                        name='mel'))
    model.add(SeparableConv2D(24, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(SeparableConv2D(48, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 2)))
    model.add(SeparableConv2D(48, (5, 5), activation='relu'))
    model.add(Flatten())
    #model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(loss=mean_squared_error, optimizer=Adam())

    if summary:
        model.summary()

    return model

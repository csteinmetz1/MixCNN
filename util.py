from datetime import datetime
import glob
import pickle
import numpy as np

def get_date_and_time():
    date_and_time = str(datetime.now()).split(' ')
    date = date_and_time[0]
    time = ('-').join(date_and_time[1].split(':')[0:2])
    return date, time

def load_data(spect_type='mel', spect_size='1024', hop_size='1024', framing=True, window_size=128):

    key = "{0} {1} {2}".format(spect_type, spect_size, hop_size)
    key = "mel 1024" # temporary fix

    # set silence thresholds
    if spect_type == 'mel':
        lim = 0.001
    if spect_type == 'mfcc':
        lim = -31.0
    
    y_rows = [] # input / output lists
    x_rows = []
    discarded = 0 # number of frames discarded

    if framing:
        for idx, song in enumerate(glob.glob("data/*.pkl")):
            row = pickle.load(open(song, "rb"))
            n_frames = np.floor((row['bass ' + key].shape[1])/window_size).astype('int')
            for frame in range(n_frames):
                start_idx = frame*window_size
                end_idx = start_idx+window_size

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
            bass_spect = row['bass ' + key][:, :window_size]
            drums_spect = row['drums ' + key][:, :window_size]
            other_spect = row['other ' + key][:, :window_size]
            vocals_spect = row['vocals ' + key][:, :window_size]
            x_rows.append(np.dstack((bass_spect, drums_spect, other_spect, vocals_spect)))

    # transform into numpy arrays
    Y = np.array([row for row in y_rows])
    X = np.array([row for row in x_rows])

    # remove nans
    X = np.nan_to_num(X)

    input_shape = (X.shape[1], X.shape[2], 4) # four instruments - 1 per channel

    print("Loaded inputs with shape:", X.shape)
    print("Loaded outputs with shape:", Y.shape)

    return X, Y, input_shape

def standardize(X_train, Y_train, X_test, Y_test, X=True, Y=True):
    if X:
        X_train_mean = np.mean(X_train, axis = 0)
        X_train_std  = np.std(X_train, axis = 0)

        X_train -= X_train_mean # zero-center
        X_train /= X_train_std  # normalize

        X_test  -= X_train_mean 
        X_test  /= X_train_std  

    if Y:
        Y_train_mean = np.mean(Y_train, axis = 0)
        Y_train_std  = np.std(Y_train, axis = 0)

        Y_train -= Y_train_mean 
        Y_train /= Y_train_std  

        Y_test  -= Y_train_mean 
        Y_test  /= Y_train_std  

    return X_train, Y_train, X_test, Y_test
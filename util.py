import os
import glob
import pickle
import numpy as np
from datetime import datetime

def load_data(spect_type='mel', spect_size='1024', hop_size='1024', framing=True, window_size=128):

    key = "{0} {1} {2}".format(spect_type, spect_size, hop_size)
    #key = "mel 1024" # temporary fix

    train_ids = np.arange(21, 100+1)
    val_ids = np.arange(11, 20+1)
    test_ids = np.arange(1, 10+1)

    # set silence threshold
    lim = 1e-6
    
    x_train = []
    y_train = []
    
    x_val = []
    y_val = []

    x_test = []
    y_test = []

    discarded = 0 # number of frames discarded

    if framing:
        for idx, song in enumerate(glob.glob("data/*.pkl")):
            row = pickle.load(open(song, "rb"))
            n_frames = np.floor((row['bass ' + key].shape[1])/window_size).astype('int')
            track_id = int(os.path.basename(song).split("_")[2].strip(".pkl"))
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
                    if   track_id in train_ids:
                        x_train.append(np.dstack((bass_spect, drums_spect, other_spect, vocals_spect)))
                        y_train.append(np.array((row['drums ratio'], row['other ratio'], row['vocals ratio'])))
                    elif track_id in val_ids:
                        x_val.append(np.dstack((bass_spect, drums_spect, other_spect, vocals_spect)))
                        y_val.append(np.array((row['drums ratio'], row['other ratio'], row['vocals ratio'])))
                    elif track_id in test_ids:
                        x_test.append(np.dstack((bass_spect, drums_spect, other_spect, vocals_spect)))
                        y_test.append(np.array((row['drums ratio'], row['other ratio'], row['vocals ratio'])))
                else:
                    discarded += 1
        print("\nDiscarded {0:d} frames with energy below the threshold.".format(discarded))
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
    X_train = np.array(x_train)
    Y_train = np.array(y_train)

    X_val = np.array(x_val)
    Y_val = np.array(y_val)

    X_test = np.array(x_test)
    Y_test = np.array(y_test)

    # remove nans
    #X = np.nan_to_num(X)

    input_shape = (X_train.shape[1], X_train.shape[2], 4) # four instruments - 1 per channel

    print("\n=============== Training Data ==============")
    print("Loaded inputs with shape:", X_train.shape)
    print("Loaded outputs with shape:", Y_train.shape)

    print("\n============== Validation Data =============")
    print("Loaded inputs with shape:", X_val.shape)
    print("Loaded outputs with shape:", Y_val.shape)

    print("\n=============== Testing Data ===============")
    print("Loaded inputs with shape:", X_test.shape)
    print("Loaded outputs with shape:", Y_test.shape)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, input_shape

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

def generate_report(report_dir, r):
    with open(os.path.join(report_dir, "report_summary.txt"), 'w') as results:
        results.write("--- RUNTIME ---\n")
        results.write("Start time: {}\n".format(r["start time"]))
        results.write("End time:   {}\n".format(r["end time"]))
        results.write("Runtime:    {}\n\n".format(r["elapsed time"]))
        results.write("--- MSE RESULTS ---\n")
        results.write("--- TRAINING DETAILS ---\n")
        results.write("Batch size:  {0}\n".format(r["batch size"]))
        results.write("Epochs:      {0}\n".format(r["epochs"]))
        results.write("Input shape: {0}\n".format(r["input shape"]))
        #results.write("Training type:  {0}\n".format(train))
        #results.write("Folds:          {0:d}\n".format(n_folds))
        #results.write("Training split: {0:d}/{1:d}\n".format(split, X.shape[0]-split))
        results.write("Learning rate:  {0:f}\n".format(r["learning rate"]))
        results.write("Spectrogram type: {0}\n".format(r["spect type"]))
        results.write("Spectrogram size: {0}\n".format(r["spect size"]))
        results.write("Standardize: {0}\n\n".format(r["standard"]))
        results.write("\n--- NETWORK ARCHITECTURE ---\n")
        r["model"].summary(print_fn=lambda x: results.write(x + '\n'))
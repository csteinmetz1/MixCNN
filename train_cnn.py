from __future__ import print_function
import keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import os
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from util import load_data, get_date_and_time, standardize
from models import *

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
                            shuffle=True,
                            callbacks=[checkpoint])
    else:   
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(X_test, Y_test), 
                            shuffle=True)

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score)

    if show_pred:
        pred = model.predict(X_test)
        print("We expect:", Y_test)
        print("We predict:", pred)

    return history, score

if __name__ == "__main__":

    batch_size = 1000
    epochs = 1
    lr = 0.001
    train = 'k fold'
    n_folds = 10
    spect_type = 'mel'
    spect_size = '1024'
    hop_size = '1024'
    standard = False
    wichern = True
    X, Y, input_shape = load_data(spect_type=spect_type, spect_size=spect_size, hop_size=hop_size, framing=True, window_size=128)

    # get the start date and time and format it
    date, time = get_date_and_time()
    start_days = date.split('-')[2]
    start_hrs = time.split('-')[0]
    start_mins = time.split('-')[1]

    report_dir = os.path.join("reports", "{0}--{1}".format(date, time))

    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)

    training_history = {}
    baseline_val_loss = []
    saved_model = None

    if train == 'k fold':
        kf = KFold(n_splits=n_folds)
        split = int(X.shape[0] - X.shape[0]/n_folds)

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            print("Running fold", i+1, "of", n_folds)

            # make train / test split
            X_train = X[train_index]
            Y_train = Y[train_index]
            X_test  = X[test_index]
            Y_test  = Y[test_index]

            # baseline predictions
            if wichern:
                bass_ratio = -10.0/-10.0
                drums_ratio = -10.0/-10.0
                other_ratio = -10.0/-14.0
                vocals_ratio = -10.0/-7.0

                Y_hat = np.array([drums_ratio, other_ratio, vocals_ratio])

                baseline_val_loss.append(np.sum([np.square((Y-Y_hat)) for Y in Y_test])/len(Y_test))

            # standardize inputs
            #if standard:
            #    X_train, Y_train, X_test, Y_test = standardize(X_train, Y_train, X_test, Y_test, Y=False)

            model = build_model_SB(input_shape, lr, summary=True)
            history, score = train_and_eval_model(model, X_train, Y_train, X_test, Y_test, batch_size, epochs, show_pred=False)
            training_history[i] = {'score' : score, 
                                   'loss': history.history['loss'], 
                                   'val_loss' : history.history['val_loss']}
            if i == 0:
                saved_model = model
            K.clear_session()
            del history
            del model
            gc.collect()
    if train == 'single':
        # calculate train / test indicies
        split = int(np.floor(X.shape[0]*0.80))

        # make train / test split
        X_train = X[:split, :]
        Y_train = Y[:split, :]
        X_test  = X[split:, :]
        Y_test  = Y[split:, :]

        # standardize inputs
        if standard:
            X_train, Y_train, X_test, Y_test = standardize(X_train, Y_train, X_test, Y_test, Y=False)

        model = build_model_SB(input_shape, lr, summary=True)
        history, score = train_and_eval_model(model, X_train, Y_train, X_test, Y_test, batch_size, epochs)
        model.save(os.path.join("reports", "{0}--{1}".format(date, time), 'final_model_loss_{0:f}.hdf5'.format(score)))
        training_history[0] = {'score' : score, 
                               'loss': history.history['loss'], 
                               'val_loss' : history.history['val_loss']}
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
        if wichern:
            for i, baseline_loss in enumerate(baseline_val_loss):
                results.write("For fold {0:d} - Test loss: {1:0.4f} MSE\n".format(i+1, baseline_loss))
            mean = np.mean([loss for loss in baseline_val_loss])
            std = np.std([loss for loss in baseline_val_loss])
        results.write("Average baseline test loss: {0:0.4f} ({1:0.4f}) MSE\n\n".format(mean, std))
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
        results.write("Standardize: {0}".format(standard))
        results.write("\n--- NETWORK ARCHITECTURE ---\n")
        saved_model.summary(print_fn=lambda x: results.write(x + '\n'))

    pickle.dump(training_history, open(os.path.join(report_dir, "training_history.pkl"), "wb"), protocol=2)
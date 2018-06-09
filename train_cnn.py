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
from util import load_data, standardize, generate_report
from models import *
import datetime
from pyalerts.py_alerts import email_alert

def train_and_eval_model(model, X_train, Y_train, X_val, Y_val, batch_size, epochs, save_weights=False):

    # checkpoint to save weights
    filepath='checkpoints/checkpoint-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    
    if save_weights:
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(X_val, Y_val),
                            shuffle=True,
                            callbacks=[checkpoint])
    else:   
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(X_val, Y_val), 
                            shuffle=True)

    score = model.evaluate(X_val, Y_val, verbose=0)
    print('Test loss:', score)

    return history, score

if __name__ == "__main__":

    # selected hyperparameters
    batch_size = 1000
    epochs = 100
    lr = 0.001
    spect_type = 'mel'
    spect_size = '1024'
    hop_size = '1024'
    standard = False

    # load data
    X_train, Y_train, X_val, Y_val, X_test, Y_test, input_shape = load_data(spect_type=spect_type, 
    spect_size=spect_size, hop_size=hop_size, framing=True, window_size=128)

    # get the start date and time and format it
    s = datetime.datetime.today()

    report_dir = os.path.join("reports", "{0}-{1:0>2}-{2:0>2}_{3:0>2}-{4:0>2}".format(s.year, s.month, s.day, s.hour, s.minute))

    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)

    training_history = {}
    baseline_val_loss = []
    saved_model = None

    # standardize inputs
    if standard:
        X_train, Y_train, X_test, Y_test = standardize(X_train, Y_train, X_test, Y_test, Y=False)

    model = build_model_SB(input_shape, lr, summary=True)
    history, score = train_and_eval_model(model, X_train, Y_train, X_test, Y_test, batch_size, epochs)
    model.save(os.path.join(report_dir, 'final_model_loss_{0:f}.hdf5'.format(score)))
    training_history[0] = {'score' : score, 
                            'loss': history.history['loss'], 
                            'val_loss' : history.history['val_loss']}
    saved_model = model
    K.clear_session()
    del history
    del model
    gc.collect()

    # get the end date and time and format it
    e = datetime.datetime.today()
    elp = e - s
    print("Training completed in {}".format(elp))

    # Save training results to file
    report_data = {"training history" : training_history,
                   "start time" : s,
                   "end time" : e,
                   "elapsed time" : elp,
                   "batch size" : batch_size,
                   "epochs" : epochs,
                   "input shape" : input_shape,
                   "learning rate" : lr,
                   "spect type" : spect_type,
                   "spect size" : spect_size,
                   "standard" : standard,
                   "model" : saved_model}

    generate_report(report_dir, report_data)
    pickle.dump(training_history, open(os.path.join(report_dir, "training_history.pkl"), "wb"), protocol=2)

    # email report
    with open(os.path.join(report_dir, "report_summary.txt"), 'r') as report_fp:
        report_details = report_fp.read()
        alert = email_alert()
        alert.send(subject="MixCNN Train Cycle {0}-{1:0>2}-{2:0>2} {3:0>2}:{4:0>2} [{5:0.4f}]".format(
            s.year, s.month, s.day, s.hour, s.minute, score), message=report_details)
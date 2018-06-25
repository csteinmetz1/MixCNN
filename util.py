import os
import sys
import glob
import pickle
import datetime
import numpy as np
import soundfile as sf
import pandas as pd

def load(level_analysis_path='level_analysis.csv', framing=False, frame_length=30.0):

    # training data 
    X = []
    Y = []

    # track id tracking
    index_list = []

    # load level analysis - run analyze.py first
    levels = pd.read_csv(level_analysis_path)

    # load audio data from each tracks
    for row_idx, row in levels.iterrows():
        index_list.append(row['id'])
        sys.stdout.write("* Loading track {:0>3}\r".format(row['id']))
        sys.stdout.flush()

        accomp_wav = os.path.join(row['path'], 'accomp_m16k_norm.wav')
        vocals_wav = os.path.join(row['path'], 'vocals_m16k_norm.wav')
        accomp_data, rate = sf.read(accomp_wav)
        vocals_data, rate = sf.read(vocals_wav)

        track_data = []
        frame_size = int(np.floor(frame_length * rate))
        n_frames = int(np.floor(accomp_data.shape[0]/(frame_size)))

        if framing:
            for frame_idx in range(n_frames):
                l = int(frame_idx * (frame_size))
                u = int(l + frame_size)
                track_data.append(np.stack((accomp_data[l:u], vocals_data[l:u]), axis=0))			
        else:
            track_data.append(np.stack((accomp_data[:frame_size], vocals_data[:frame_size]), axis=0))

        X.append(np.array(track_data))
        Y.append(row['mix ratio'])

    sys.stdout.write("* {} tracks loaded    \n".format(row_idx+1))
    return X, Y, index_list

def generate_report(report_dir, r, msg=''):
    with open(os.path.join(report_dir, "report_summary.txt"), 'w') as results:
        results.write("--- RUNTIME ---\n")
        results.write("Start time: {}\n".format(r["start time"]))
        results.write("End time:   {}\n".format(r["end time"]))
        results.write("Runtime:    {}\n\n".format(r["elapsed time"]))
        results.write("--- MESSAGE ---\n")
        results.write("{}\n\n".format(msg))
        results.write("--- MSE RESULTS ---\n")
        val_losses = []
        for fold, track_id in zip(r["history"], r["index list"]):
            results.write("* Track {}\n".format(track_id))
            results.write("    train   |  val\n")
            for epoch, (train_loss, val_loss) in enumerate(zip(fold.history["loss"], 
                                                        fold.history["val_loss"])):
                results.write("{0}: {1:0.6f}   {2:0.6f}\n".format(epoch+1, 
                                                train_loss, val_loss))
                val_losses.append(val_loss)
            results.write("\n")
        final_loss = np.mean(val_losses)
        results.write("Avg. val loss: {0:0.6f}\n".format(final_loss))
        results.write("\n--- TRAINING DETAILS ---\n")
        results.write("Batch size:  {0}\n".format(r["batch size"]))
        results.write("Epochs:      {0}\n".format(r["epochs"]))
        results.write("Input shape: {0}\n".format(r["input shape"]))
        results.write("Training:    {0}\n".format(r["train"]))
        results.write("Model type:  {0}\n".format(r["model type"]))
        results.write("Folds:       {0:d}\n".format(r["folds"]))
        results.write("Learning:    {0:f}\n".format(r["learning rate"]))
        results.write("FFT Size:    {0}\n".format(r["n_dft"]))
        results.write("Hop size:    {0}\n".format(r["n_hop"]))
        results.write("Framing:     {0}\n".format(r["framing"]))
        results.write("Frame:       {0} seconds\n\n".format(r["frame length"]))
        results.write("\n--- NETWORK ARCHITECTURE ---\n")
        r["model"].summary(print_fn=lambda x: results.write(x + '\n'))
        
        val_loss = [fold.history['val_loss'] for fold in r['history']]
        train_loss = [fold.history['loss'] for fold in r['history']]

        history = {'val loss' : val_loss,
                   'train loss' : train_loss,
                   'final loss' : final_loss}

        pickle.dump(history, open(os.path.join(report_dir, 
                    "history.pkl"), "wb"), protocol=2)
        return final_loss

def setup_report_directory():
    
    # ensure root reports directory exists
    if not os.path.isdir('reports'):
        os.makedirs('reports')

    # get current time
    s = datetime.datetime.today()

    # directory to save training details
    report_dir = os.path.join('reports', 
            "{0}-{1:0>2}-{2:0>2}_{3:0>2}-{4:0>2}".format(
            s.year, s.month, s.day, s.hour, s.minute))
            
    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)

    return report_dir

def profile():
    print("* No framing")
    start = time.time()
    X, Y = load(framing=False)
    end = time.time() - start
    print(end)
    for track, mix in zip(X, Y):
        print(track.shape)
        print(mix)

    print("* With framing")
    start = time.time()
    X, Y = load(framing=True)
    end = time.time() - start
    print(end)
    for track, mix in zip(X, Y):
        print(track.shape)
        print(mix)

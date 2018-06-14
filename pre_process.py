import glob
import sys
import os
from collections import OrderedDict
import pickle
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pyloudnorm
import soundfile as sf
import warnings

def augmentation():

    for idx, song in enumerate(glob.glob("DSD100/Sources/**/*")):
        track_id = song.split('/')[len(song.split('/'))-1][0:3]
        track_type = song.split('/')[len(song.split('/'))-2]

        if not os.path.isdir(os.path.join(song, "augmented")):
            os.makedirs(os.path.join(song, "augmented")) # create new dir to store augmented stems

        for stem in glob.glob(os.path.join(song, "*.wav")):
            stem_class = stem.split('/')[len(stem.split('/'))-1].split('.')[0]
            y, sr = librosa.load(stem, sr=44100, mono=False)
            y_left  = y[0,:]
            y_right = y[1,:]

            for factor in [0.81]: #[0.81, 0.93, 1.07, 1.23]:
                subdir = "stretch_{}".format(factor)
                if not os.path.isdir(os.path.join(song, "augmented", subdir)):
                    os.makedirs(os.path.join(song, "augmented", subdir))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    stretch_left = librosa.effects.time_stretch(y_left, factor)
                    stretch_right = librosa.effects.time_stretch(y_right, factor)
                stretch = np.stack((stretch_left, stretch_right), axis=0)
                #stretch = np.reshape(stretch, (stretch.shape[1], stretch.shape[0]))
                filename = "{}.wav".format(stem_class)
                librosa.output.write_wav(os.path.join(song, "augmented", subdir, filename), stretch, sr)
                sys.stdout.write(" Stretching by {: >4}     \r".format(factor))
                sys.stdout.flush()

            for semitones in [1]: #[-2, -1, 1, 2]:
                subdir = "shift_{}".format(semitones)
                if not os.path.isdir(os.path.join(song, "augmented", subdir)):
                    os.makedirs(os.path.join(song, "augmented", subdir))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    shift_left = librosa.effects.pitch_shift(y_left, sr, n_steps=semitones)
                    shift_right = librosa.effects.pitch_shift(y_right, sr, n_steps=semitones)
                shift = np.stack((shift_left, shift_right), axis=0)
                #shift = np.reshape(shift, (shift.shape[1], shift.shape[0]))
                filename = "{}.wav".format(stem_class)
                librosa.output.write_wav(os.path.join(song, "augmented", subdir, filename), shift, sr)
                sys.stdout.write(" Shifting by {: >2}      \r".format(semitones))
                sys.stdout.flush()

def level_analysis():
    meter = pyloudnorm.loudness.Meter(44100) # create loudness meter
    target = -24.0 # target loudness
    database = [] # dict to store columns
    for idx, song in enumerate(glob.glob("DSD100/Sources/**/*")):
        print(song)
        track_id = song.split('/')[len(song.split('/'))-1][0:3]
        track_type = song.split('/')[len(song.split('/'))-2]
        if not os.path.isdir(os.path.join(song, "normalized")):
            os.makedirs(os.path.join(song, "normalized")) # create new dir to store normalized stems
        print("Analyzing {}...".format(track_id))
        database.append(OrderedDict({'type' : track_type,
                                    'track id' : track_id, 
                                    'bass LUFS' : 0,
                                    'drums LUFS' : 0,
                                    'other LUFS' : 0,
                                    'vocals LUFS' : 0,
                                    'bass ratio' : 0,
                                    'drums ratio' : 0,
                                    'other ratio' : 0,
                                    'vocals ratio' : 0}))
        for stem in glob.glob(os.path.join(song,"*.wav")):

            stem_class = stem.split('/')[len(stem.split('/'))-1].split('.')[0]
            # read file and measure LUFS
            data, rate = sf.read(stem)

            # measure loudness
            stem_loudness = meter.integrated(data)

            # store results
            database[idx][stem_class + ' LUFS'] = stem_loudness
            database[idx][stem_class + ' ratio'] = database[idx]['bass LUFS'] / stem_loudness

            # normalize to target and save .wav
            norm_data = pyloudnorm.normalize.loudness(data, stem_loudness, target)
            sf.write(os.path.join(song, "normalized", stem_class + ".wav"), norm_data, rate)

    # create dataframe and save result to csv
    dataframe = pd.DataFrame(database)
    dataframe.to_csv("data/level_analysis.csv", sep=',')
    print("Saved level data for {0} tracks".format(len(database)))

    return database

def spectral_analysis(save_data=True, save_img=False):

    level_analysis = pd.read_csv("data/level_analysis.csv")

    for idx, song in enumerate(glob.glob("DSD100/Sources/**/*")):
        print(song)
        track_id = song.split('/')[len(song.split('/'))-1][0:3]
        track_type = song.split('/')[len(song.split('/'))-2]

        if not os.path.isdir(os.path.join(song, "img")):
            os.makedirs(os.path.join(song, "img"))

        bass_ratio = float(level_analysis.loc[level_analysis['track id'] == int(track_id)]['bass ratio'])
        drums_ratio = float(level_analysis.loc[level_analysis['track id'] == int(track_id)]['drums ratio'])
        other_ratio = float(level_analysis.loc[level_analysis['track id'] == int(track_id)]['other ratio'])
        vocals_ratio = float(level_analysis.loc[level_analysis['track id'] == int(track_id)]['vocals ratio'])

        database = (OrderedDict({'type' : track_type,
                                    'track id' : track_id,
                                    'bass ratio' : bass_ratio,
                                    'drums ratio' : drums_ratio,
                                    'other ratio' : other_ratio,
                                    'vocals ratio' : vocals_ratio,
                                    'bass mel 1024 1024' : [],
                                    'drums mel 1024 1024' : [],
                                    'other mel 1024 1024' : [],
                                    'vocals mel 1024 1024' : [],
                                    'bass mel 1024 512' : [],
                                    'drums mel 1024 512' : [],
                                    'other mel 1024 512' : [],
                                    'vocals mel 1024 512' : []}))

        for stem in glob.glob(os.path.join(song, "normalized", "*.wav")):
            stem_class = stem.split('/')[len(stem.split('/'))-1].split('.')[0]
            y, sr = librosa.load(stem, sr=44100, mono=True)
            #y = librosa.util.fix_length(y, sr*180)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=1024, n_mels=128)
            mel_hop = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
            database[stem_class + ' mel 1024 1024'] = mel
            database[stem_class + ' mel 1024 512'] = mel_hop

            if save_img:
                plt.figure(figsize=(5.5, 5))
                librosa.display.specshow(librosa.power_to_db(mel[:,128:2*128], ref=np.max), fmax=22050, y_axis='mel', x_axis='time')
                plt.tight_layout()
                plt.savefig(os.path.join(song, "img", "mel_" + stem_class + ".png")) 
                plt.close('all')

            sys.stdout.write("Saved Mel spectrogram data of track {1} - {0}   \r".format(stem_class, track_id))
            sys.stdout.flush()

        # save ordereddict to pickle
        if not os.path.isdir("data"):
            os.makedirs("data")
        pickle.dump(database, open(os.path.join("data", "spectral_analysis_{0}.pkl".format(track_id)), "wb"), protocol=2)
        sys.stdout.write("Spectral analysis complete for track {0}             \n".format(track_id)) 

if __name__ == "__main__":
    if len(sys.argv) == 2:
        pre_process_type = sys.argv[1]
        if pre_process_type == '--level':
            level_analysis()
        elif pre_process_type == '--spectral':
            spectral_analysis()
        elif pre_process_type == '--augment':
            augmentation()
        elif pre_process_type == '--all':
            level_analysis()
            spectral_analysis()
            augmentation()
        else:
            print("Usage: python pre_process.py  pre_process_type")
            print("Valid types: --level, --spectral, --augment --all")
            sys.exit(0)
    else:
        print("Usage: python pre_process.py  pre_process_type")
        print("Valid types: --level, --spectral, --augment --all")
        sys.exit(0)

import glob
import sys
import os
from collections import OrderedDict
import pickle
import scipy.io.wavfile as wavfile
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pyloudnorm.util
import pyloudnorm.loudness
import pyloudnorm.normalize

def level_analysis():
    # dict to store columns
    database = []
    for idx, song in enumerate(glob.glob("DSD100/Sources/**/*")):
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
            rate, data = wavfile.read(stem)
            data = pyloudnorm.util.validate_input_data(data, rate)
            stem_loudness = pyloudnorm.loudness.measure_gated_loudness(data, rate)
            # store results
            database[idx][stem_class + ' LUFS'] = stem_loudness
            database[idx][stem_class + ' ratio'] = database[idx]['bass LUFS'] / stem_loudness
            # normalize stem - NOTE: this currently produces a 32-bit floating point .wav 
            ln_audio = pyloudnorm.normalize.loudness(data, rate, -24.0)
            wavfile.write(os.path.join(song, "normalized", stem_class + ".wav"), rate, ln_audio)

    # create dataframe and save result to csv
    dataframe = pd.DataFrame(database)
    dataframe.to_csv("level_analysis.csv", sep=',')
    print("Saved level data for {0} tracks".format(len(database)))

    return database

def spectral_analysis(save_data=True, save_img=False):

    level_analysis = pd.read_csv("level_analysis.csv")

    for idx, song in enumerate(glob.glob("DSD100/Sources/**/*")):
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
                                    'bass mel u' : [],
                                    'drums mel u' : [],
                                    'other mel u' : [],
                                    'vocals mel u' : [],
                                    'bass mel tn' : [],
                                    'drums mel tn' : [],
                                    'other mel tn' : [],
                                    'vocals mel tn' : [],
                                    'bass mel sm' : [],
                                    'drums mel sm' : [],
                                    'other mel sm' : [],
                                    'vocals mel sm' : [],
                                    'bass mel lg' : [],
                                    'drums mel lg' : [],
                                    'other mel lg' : [],
                                    'vocals mel lg' : [],
                                    'bass mfcc sm' : [],
                                    'drums mfcc sm' : [],
                                    'other mfcc sm' : [],
                                    'vocals mfcc sm' : [],
                                    'bass mfcc lg' : [],
                                    'drums mfcc lg' : [],
                                    'other mfcc lg' : [],
                                    'vocals mfcc lg' : []}))

        for stem in glob.glob(os.path.join(song, "normalized", "*.wav")):
            stem_class = stem.split('/')[len(stem.split('/'))-1].split('.')[0]
            y, sr = librosa.load(stem, sr=44100, mono=True)
            y = librosa.util.fix_length(y, sr*180)
            y_22k = librosa.resample(y, sr, 22050)
            mel_u = librosa.feature.melspectrogram(y=y_22k, sr=22050, n_fft=16384, hop_length=8192, n_mels=128)
            mel_tn = librosa.feature.melspectrogram(y=y_22k, sr=22050, n_fft=8192, hop_length=4096, n_mels=128)
            mel_sm = librosa.feature.melspectrogram(y=y_22k, sr=22050, n_fft=4096, hop_length=2048, n_mels=128)
            mel_lg = librosa.feature.melspectrogram(y=y_22k, sr=22050, n_fft=2048, hop_length=1024, n_mels=128)
            mfcc_u = librosa.feature.mfcc(S=librosa.power_to_db(mel_u), n_mfcc=20)
            mfcc_tn = librosa.feature.mfcc(S=librosa.power_to_db(mel_tn), n_mfcc=20)
            mfcc_sm = librosa.feature.mfcc(S=librosa.power_to_db(mel_sm), n_mfcc=20)
            mfcc_lg = librosa.feature.mfcc(S=librosa.power_to_db(mel_lg), n_mfcc=20)
            database[stem_class + ' mel u'] = np.nan_to_num(mel_u)
            database[stem_class + ' mel tn'] = np.nan_to_num(mel_tn)
            database[stem_class + ' mel sm'] = mel_sm
            database[stem_class + ' mel lg'] = mel_lg
            database[stem_class + ' mfcc u'] = mfcc_u
            database[stem_class + ' mfcc tn'] = mfcc_tn
            database[stem_class + ' mfcc sm'] = mfcc_sm
            database[stem_class + ' mfcc lg'] = mfcc_lg

            if save_img:
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(librosa.power_to_db(mel_u, ref=np.max), y_axis='mel', x_axis='time')
                plt.colorbar(format='%+2.0f dB')
                plt.title('Mel spectrogram of normalized ' + stem_class)
                plt.tight_layout()
                plt.savefig(os.path.join(song, "img", "mel_" + stem_class + ".png")) 

                plt.figure(figsize=(10, 4))
                librosa.display.specshow(mfcc_sm, x_axis='time')
                plt.colorbar()
                plt.title('MFCC')
                plt.tight_layout() 
                plt.savefig(os.path.join(song, "img", "mfcc_" + stem_class + ".png")) 

                plt.close('all')

            sys.stdout.write("Saved Mel spectrogram data of track {1} - {0}   \r".format(stem_class, track_id))
            sys.stdout.flush()

        # save ordereddict to pickle
        if not os.path.isdir("data"):
            os.makedirs("data")
        pickle.dump(database, open(os.path.join("data", "spectral_analysis_{0}.pkl".format(track_id)), "wb"), protocol=2)
        sys.stdout.write("Spectral analysis complete for track {0}             \n".format(track_id)) 

#level_analysis()
spectral_analysis(save_img=True)
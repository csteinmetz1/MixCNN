import glob
import sys
import os
from collections import OrderedDict
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
        track_id = song.split('/')[3][0:3]
        track_type = song.split('/')[2]
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
            stem_class = stem.split('/')[4].split('.')[0]
            # read file and measure LUFS
            rate, data = wavfile.read(stem)
            data = pyloudnorm.util.validate_input_data(data, rate)
            stem_loudness = pyloudnorm.loudness.measure_gated_loudness(data, rate)
            # store results
            database[idx][stem_class + ' LUFS'] = stem_loudness
            database[idx][stem_class + ' ratio'] = database[idx]['bass LUFS'] / stem_loudness
            # normalize stem - NOTE: this currently produces a 32-bit floating point .wav 
            ln_audio = pyloudnorm.normalize.loudness(data, rate, -30.0)
            wavfile.write(os.path.join(song, "normalized", stem_class + ".wav"), rate, ln_audio)

    # create dataframe and save result to csv
    dataframe = pd.DataFrame(database)
    dataframe.to_csv("level_analysis.csv", sep=',')
    print("Saved level data from {0} tracks".format(len(database)))

def spectral_analysis(song_dir, save_data=True, save_img=False):
    if not os.path.isdir(os.path.join(song_dir, "img")):
        os.makedirs(os.path.join(song_dir, "img"))
    if not os.path.isdir(os.path.join(song_dir, "data")):
        os.makedirs(os.path.join(song_dir, "data"))
    for stem in glob.glob(os.path.join(song_dir, "normalized", "*.wav")):
        track_id = stem.split('/')[len(stem.split('/'))-3][0:3]
        stem_class = stem.split('/')[len(stem.split('/'))-1].split('.')[0]
        y, sr = librosa.load(stem, sr=22050, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

        if save_data:
            np.savetxt(os.path.join(song_dir, "data", stem_class + ".txt"), S, delimiter=',')
            sys.stdout.write("Saved Mel spectrogram data of track {1} - {0}   \r".format(stem_class, track_id))
            sys.stdout.flush()

        if save_img:
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel spectrogram of normalized ' + stem_class)
            plt.tight_layout()
            plt.savefig(os.path.join(song_dir, "img", stem_class + ".png")) 

    sys.stdout.write("Spectral analysis complete for track {0}             \n".format(track_id)) 

spectral_analysis("/Users/christiansteinmetz/Documents/MLGAN/DSD100/Sources/Test/049 - Young Griffo - Facade")
spectral_analysis("/Users/christiansteinmetz/Documents/MLGAN/DSD100/Sources/Test/005 - Angela Thomas Wade - Milk Cow Blues")
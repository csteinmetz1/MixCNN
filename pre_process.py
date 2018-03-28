import glob
import os
from collections import OrderedDict
import scipy.io.wavfile as wavfile
import pandas as pd
import pyloudnorm.util
import pyloudnorm.loudness
import pyloudnorm.normalize

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
		
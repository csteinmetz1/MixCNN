import os
import sys
import glob
import pandas as pd
import soundfile as sf
import pyloudnorm as pyln
from collections import OrderedDict

dataset_dir = "DSD100"

def measure_loudness(rate, target):
	""" Measure loudness of tracks, save to .csv, and normalize tracks to target. """

	meter = pyln.Meter(rate) # create loudness meter
	database = [] # dict to store columns

	for track in glob.glob(os.path.join(dataset_dir, "Sources", "**", "*")):
		track_details = os.path.basename(track)
		print("* Processing track {}...".format(track_details))
		track_id = track_details.split(' - ')[0]
		track_artist = track_details.split(' - ')[1]
		track_title = track_details.split(' - ')[2]
		track_stats = OrderedDict({'id' : track_id, 'artist' : track_artist,
								   'title' : track_title, 'augment type' : 'None',
								   'accomp LUFS'  : 0, 'vocals LUFS'  : 0,
								   'mix ratio' : 0, 'path' : track})

		# load audio data
		accomp_data, rate = sf.read(os.path.join(track, "accomp_m16k.wav"))
		vocals_data, rate = sf.read(os.path.join(track, "vocals_m16k.wav"))
 
		# measure loudness
		accomp_loudness = meter.integrated_loudness(accomp_data)
		vocals_loudness = meter.integrated_loudness(vocals_data)

		# store results
		track_stats['accomp LUFS'] = accomp_loudness
		track_stats['vocals LUFS'] = vocals_loudness
		track_stats['mix ratio']   = accomp_loudness / vocals_loudness

		# normalize to target and save .wav
		accomp_norm = pyln.normalize.loudness(accomp_data, accomp_loudness, target)
		vocals_norm = pyln.normalize.loudness(vocals_data, vocals_loudness, target)
		sf.write(os.path.join(track, "accomp_m16k_norm.wav"), accomp_norm, rate)
		sf.write(os.path.join(track, "vocals_m16k_norm.wav"), vocals_norm, rate)

		# load new track data into database
		database.append(track_stats) 

	# create dataframe and save result to csv
	dataframe = pd.DataFrame(database)
	dataframe = dataframe.set_index(['id'])
	dataframe.to_csv("level_analysis.csv", sep=',')
	print("Saved level data for {0} tracks".format(len(database)))

measure_loudness(16000, -24.0)
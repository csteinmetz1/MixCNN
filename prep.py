import os
import glob
import shutil
import subprocess

dataset_dir = "DSD100"

def remove_mixtures():
	# remove mixtures from the dataset
	mixtures = os.path.join(dataset_dir, "Mixtures")
	if os.path.isdir(mixtures):
		shutil.rmtree(mixtures)

def process_tracks(remove_orig=True):

	for track in glob.glob(os.path.join(dataset_dir, "Sources", "**", "*")):
		
		print("* Processing track {}...".format(os.path.basename(track)))

		# combine bass, drums, and other to create single accompaniment track
		print("Mixing accompaniment...")
		subprocess.call("sox bass.wav -p | sox - -m drums.wav -p | sox - -m other.wav accomp.wav",
			cwd=os.path.abspath(track), shell=True)

		# convert accompaniment track to mono and downsample to 16 kHz
		subprocess.call("sox accomp.wav -r 16k accomp_m16k.wav remix 1,2", 
			cwd=os.path.abspath(track), shell=True)
		print("Mixing accompaniment to mono and downsampling...")

		# convert vocal track to mono and downsample to 16 kHz
		subprocess.call("sox vocals.wav -r 16k -e floating-point -b 32 vocal_m16k.wav remix 1,2", 
			cwd=os.path.abspath(track), shell=True)
		print("Mixing vocals to mono and downsampling...")

		if remove_orig:
			# remove accompaniment stems
			for stem in ['bass.wav', 'drums.wav', 'other.wav']:
				stempath = os.path.join(track, stem)
				os.remove(stempath)
				print("Removed {}.".format(stem))

			# remove 44.1 kHz stereo accompaniment mix
			os.remove(os.path.join(track, "accomp.wav"))

			# remove 44.1 kHz stereo vocals
			os.remove(os.path.join(track, "vocals.wav"))


process_tracks(remove_orig=True)
remove_mixtures()
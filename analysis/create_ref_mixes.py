import os
import glob
import subprocess

test_dir = "DSD100/Sources/test/*"
val_dir = "DSD100/Sources/val/*"
train_dir = "DSD100/Sources/train/*"

paths = [test_dir, val_dir, train_dir]

if not os.path.exists("mixes"):
	os.makedirs("mixes")

def create_mix(song_dir):
	song_title = song_dir.split('/')[len(song_dir.split('/'))-1]
	bass = os.path.join(song_dir, "bass.wav")
	drums = os.path.join(song_dir, "drums.wav")
	other = os.path.join(song_dir, "other.wav")
	vocals = os.path.join(song_dir, "vocals.wav")
	output = os.path.join("mixes", song_title + ".wav")

	subprocess.call(["sox", "-m", bass, drums, other, vocals, output])

for path in paths:
	for song in glob.glob(path):
		song_title = song.split('/')[len(song.split('/'))-1]
		print("Mixing {}...".format(song_title))
		create_mix(song)
	
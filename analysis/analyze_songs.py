import glob
import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib as mpl 
mpl.use('agg')
import matplotlib.pyplot as plt


level_analysis = pd.read_csv("data/level_analysis.csv")

# Box plot for LUFS values
bass_LUFS = level_analysis['bass LUFS'].tolist()
drums_LUFS = level_analysis['drums LUFS'].tolist()
other_LUFS = level_analysis['other LUFS'].tolist()
vocals_LUFS = level_analysis['vocals LUFS'].tolist()
 
print(len(bass_LUFS))

data_LUFS = [bass_LUFS, drums_LUFS, other_LUFS, vocals_LUFS]

fig = plt.figure(1, figsize=(6,4))
ax = fig.add_subplot(111)
bp = ax.boxplot(data_LUFS, patch_artist=True)
ax.set_xticklabels(['Bass', 'Drums', 'Other', 'Vocals'])
ax.set_ylabel('Loudness (LUFS)')

# a e s t h e t i c s
gray = 'darkslategray'
lightgray = '#eaeaf2'
group_colors = ['steelblue', 'firebrick', 'seagreen', 'khaki']

ax.yaxis.grid(True,  linestyle="-", color='white', lw=2)
ax.set_facecolor((lightgray))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

for idx,box in enumerate(bp['boxes']):
    box.set(color = gray, linewidth=2)
    box.set(facecolor = group_colors[idx])
for whisker in bp['whiskers']:
    whisker.set(color=gray, linewidth=2)
for median in bp['medians']:
    median.set(color=gray, linewidth=2)
for cap in bp['caps']:
    cap.set(color=gray, linewidth=2)
for flier in bp['fliers']:
    flier.set(marker='D', fillstyle='full', color=gray, alpha=2)

fig.savefig('LUFS.png', bbox_inches='tight')
plt.close('all')

# Box plot for loudness ratios
bass_ratio = level_analysis['bass ratio'].tolist()
drums_ratio = level_analysis['drums ratio'].tolist()
other_ratio = level_analysis['other ratio'].tolist()
vocals_ratio = level_analysis['vocals ratio'].tolist()

data_ratio = [bass_ratio, drums_ratio, other_ratio, vocals_ratio]

fig = plt.figure(1, figsize=(6,4))
ax = fig.add_subplot(111)
bp = ax.boxplot(data_ratio, patch_artist=True)
ax.set_xticklabels(['Bass', 'Drums', 'Other', 'Vocals'])
ax.set_ylabel('Loudness ratio (w.r.t the bass)')

# a e s t h e t i c s
gray = 'darkslategray'
lightgray = '#eaeaf2'
group_colors = ['steelblue', 'firebrick', 'seagreen', 'khaki']

ax.yaxis.grid(True,  linestyle="-", color='white', lw=2)
ax.set_facecolor((lightgray))

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

for idx,box in enumerate(bp['boxes']):
    box.set(color = gray, linewidth=2)
    box.set(facecolor = group_colors[idx])
for whisker in bp['whiskers']:
    whisker.set(color=gray, linewidth=2)
for median in bp['medians']:
    median.set(color=gray, linewidth=2)
for cap in bp['caps']:
    cap.set(color=gray, linewidth=2)
for flier in bp['fliers']:
    flier.set(marker='D', fillstyle='full', color=gray, alpha=2)

fig.savefig('ratio.png', bbox_inches='tight')
plt.close('all')
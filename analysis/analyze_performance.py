import os
import numpy as np
import matplotlib as mpl 
mpl.use('agg')
import matplotlib.pyplot as plt

cnn = [0.0280, 0.0443, 0.0468, 0.0514, 0.0444, 0.0395, 0.0299, 0.0308, 0.0406, 0.0745]
baseline =[0.3619, 0.3198, 0.3432, 0.3760, 0.3433, 0.2905, 0.2230, 0.3647, 0.3477, 0.3697]

models = [baseline, cnn]

fig = plt.figure(1, figsize=(6,4))
ax = fig.add_subplot(111)
bp = ax.boxplot(models, patch_artist=True)
ax.set_xticklabels(['Baseline', 'CNN'])
ax.set_ylabel('Validation loss (MSE)')

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

fig.savefig('performance.png', bbox_inches='tight')
plt.close('all')
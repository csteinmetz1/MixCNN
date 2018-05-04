import sys
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# Set the default color cycle
#mpl.rcParams['axes.color_cycle'] = ['r', 'k', 'c']
sns.set_palette(sns.color_palette("GnBu_d"))
#"PuBuGn_d"

def analyze(filepath):
	# load training history 
	training_history = pickle.load(open(filepath, "rb"))
	date_and_time = filepath.split('/')[len(filepath.split('/'))-2]

	# create training loss plot - over all epochs
	plt.figure(1)
	for i, fold in training_history.items():
		loss = fold['loss']
		val_loss = fold['val_loss']

		n_epochs = len(loss)

		t = np.arange(1, n_epochs+1)
		plt.plot(t, loss, linewidth=0.5, color='#d73c49')
		plt.plot(t, val_loss, linewidth=0.5, color='#417e90')
		plt.ylabel('Training Loss (MSE)')
		plt.title('Training Loss (MSE)')
		plt.xlabel('Epoch')

	plt.savefig(os.path.join("reports", date_and_time, "train_and_val_loss_summary.png"))
	plt.close('all')

	# create training loss plot - over ending epochs
	plt.figure(1)
	for i, fold in training_history.items():
		loss = fold['loss']
		val_loss = fold['val_loss']

		n_epochs = len(loss)
		start = n_epochs-(int(np.floor(0.1*n_epochs)))
		end = n_epochs

		t = np.arange(start, end)
		plt.plot(t, loss[start:end], label='loss', linewidth=1, color='#d73c49')
		plt.plot(t, val_loss[start:end], label='val loss', linewidth=1, color='#417e90')
		plt.ylabel('Training Loss (MSE)')
		plt.gca().spines['top'].set_visible(False)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().grid(True)
		plt.legend(loc=1, borderaxespad=0.)

	plt.savefig(os.path.join("reports", date_and_time, "train_and_val_loss_end_epochs_summary.png"))
	plt.close('all')

	# create training loss plot with narrow y limits
	plt.figure(1)
	for i, fold in training_history.items():
		loss = fold['loss']
		val_loss = fold['val_loss']

		n_epochs = len(loss)
		start = 1
		end = n_epochs

		t = np.arange(start, end)
		plt.plot(t, loss[start:end], label='loss', linewidth=0.5, color='#d73c49')
		plt.plot(t, val_loss[start:end], label='val loss', linewidth=0.5, color='#417e90')
		plt.ylabel('Training Loss (MSE)')
		plt.xlabel('Epoch')
		plt.gca().spines['top'].set_visible(False)
		plt.gca().spines['right'].set_visible(False)
		plt.gca().grid(True)
		plt.ylim([0, 0.2])
		plt.legend(loc=1, borderaxespad=0.)
		plt.tight_layout()

	plt.savefig(os.path.join("reports", date_and_time, "detailed_loss.png"), )
	plt.close('all')

	# create training loss plots over ending epochs for each fold
	if not os.path.isdir(os.path.join("reports", date_and_time, "all_folds")):
		os.makedirs(os.path.join("reports", date_and_time, "all_folds"))

	for i, fold in training_history.items():
		loss = fold['loss']
		val_loss = fold['val_loss']

		plt.figure(1)
		n_epochs = len(loss)
		start = 1
		end = n_epochs

		t = np.arange(start, end)
		plt.plot(t, loss[start:end], label='loss', linewidth=0.5, color='#d73c49')
		plt.plot(t, val_loss[start:end], label='val loss', linewidth=0.5, color='#417e90')
		plt.ylabel('Training Loss (MSE)')
		plt.title('Training Loss (MSE)')
		plt.ylim([0, 0.1])
		plt.legend(loc=1, borderaxespad=0.)

		plt.savefig(os.path.join("reports", date_and_time, "all_folds", "train_and_val_loss_fold{0}_summary.png".format(i+1)))
		plt.close('all')
		
if __name__ == "__main__":
	if len(sys.argv) == 2:
		filepath = sys.argv[1]
		analyze(filepath)
	else:
		print("Invalid training history file.")
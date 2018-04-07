import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

def analyze(filepath):
	# load training history 
	training_history = pickle.load(open(filepath, "rb"))
	date_and_time = filepath.split('.')[0][len(filepath)-21:len(filepath)-1]
	if not os.path.isdir("reports/{0}".format(date_and_time)):
		os.makedirs("reports/{0}".format(date_and_time))

	# create training loss plots - over all epochs
	for i, fold in training_history.items():
		loss = fold['loss']
		val_loss = fold['val_loss']

		n_epochs = len(loss)

		t = np.arange(1, n_epochs+1)
		fig1 = plt.figure()
		ax1 = fig1.add_subplot(2, 1, 1)
		ax1.plot(t, loss)
		ax1.set_ylabel('Training Loss (MSE)')
		ax1.set_title('Training Loss (MSE)')

		ax2 = fig1.add_subplot(2, 1, 2)
		ax2.plot(t, val_loss)
		ax2.set_xlabel('Epoch')
		ax2.set_ylabel('Validation Loss (MSE)')

	fig1.savefig("reports/{0}/train_and_val_loss.png".format(date_and_time))

	# create training loss plots - over ending epochs
	for i, fold in training_history.items():
		loss = fold['loss']
		val_loss = fold['val_loss']

		n_epochs = len(loss)
		start = n_epochs-50
		end = n_epochs

		t = np.arange(start, end)
		fig2 = plt.figure()
		ax1 = fig2.add_subplot(2, 1, 1)
		ax1.plot(t, loss[start:end])
		ax1.set_ylabel('Training Loss (MSE)')
		ax1.set_title('Training Loss (MSE)')

		ax2 = fig2.add_subplot(2, 1, 2)
		ax2.plot(t, val_loss[start:end])
		ax2.set_xlabel('Epoch')
		ax2.set_ylabel('Validation Loss (MSE)')

	fig2.savefig("reports/{0}/train_and_val_loss_end_epochs.png".format(date_and_time))

if __name__ == "__main__":
	if len(sys.argv) == 2:
		filepath = sys.argv[1]
		analyze(filepath)
	else:
		print("Invalid training history file.")
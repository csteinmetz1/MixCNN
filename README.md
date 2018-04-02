# MLGAN
Mulitrack mix leveling with generative adversarial network.

## Setup

Install dependancies.

`$ pip install --upgrade -r requirements.txt`

Install python ITU-R BS.1770-4 loudness package.

```
$ git clone https://github.com/csteinmetz1/pyloudnorm.git
$ cd pyloudnorm
$ python setup.py install
```

## Dataset

Download and extract the DSD100 dataset: [http://liutkus.net/DSD100.zip](http://liutkus.net/DSD100.zip) (12 GB) 

Ensure that the extracted `DSD100` directory is placed in the top of the directory structure.

## Pre-process

To generate the input and output data run the `pre_process.py` script.

`$ python pre_process.py`

This will first measure the true mix loudness levels (and loudness ratios) which are saved to a .csv file. 
Then all of the stems are normalized to -30 LUFS.
Next spectrograms will be generated of the normalized stems and stored in a pickle file.

![bass](img/bass.png)
![bass](img/drums.png)
![bass](img/other.png)
![bass](img/vocals.png)

## Train

To train the CNN model run the `train_cnn.py` script.

`$ python train_cnn.py`
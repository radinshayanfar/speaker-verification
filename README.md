# Speaker Recognition & Verification Experiments with Common Voice.
This repository contains modified scripts of [SpeechBrain](https://github.com/speechbrain/speechbrain/) for running speaker identification and verification experiments with the Persian [Common Voice](https://commonvoice.mozilla.org/en/datasets) dataset. State-of-the-art [ECAPA-TDNN](https://arxiv.org/abs/2005.07143) model is utilized for this experiment.

The original scripts for training on [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) are [here](https://github.com/speechbrain/speechbrain/tree/9d56d50809d8745cc7dafa930aec554145be4028/recipes/VoxCeleb/SpeakerRec).


## Speaker verification using ECAPA-TDNN embeddings
The following notebooks used to train, test, and use ECAPA-TDNN model:

- [Train notebook](./notebooks/ECAPA_train.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/radinshayanfar/speaker-verification/blob/master/notebooks/ECAPA_train.ipynb)
- [Test notebook](<./notebooks/ECAPA-TDNN test.ipynb>) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<https://colab.research.google.com/github/radinshayanfar/speaker-verification/blob/master/notebooks/ECAPA-TDNN test.ipynb>)
- [Demo notebook](<./notebooks/ECAPA-TDNN demo.ipynb>) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](<https://colab.research.google.com/github/radinshayanfar/speaker-verification/blob/master/notebooks/ECAPA-TDNN demo.ipynb>)

Not all file paths used in the notebooks are available publicly.

## Performance summary

[Speaker Verification Results with Common Voice]

| System          | Dataset    | Accuracy<sup>1</sup> | Stress Test Accuracy<sup>2</sup> | Model Link<sup>3</sup> |
|-----------------|------------|------| -----| -----|
| ECAPA-TDNN      | Persian Common Voice | 97.5% | 86.8% | https://drive.google.com/drive/folders/1R_gvC_St56Atxfu8MLRb1PIlBnBahta2?usp=sharing |

<sup>1</sup> Tested on a private hand-made dataset, consisting of 8400 pair samples.

<sup>2</sup> Tested on German [BROTHERS](https://clarin.phonetik.uni-muenchen.de/BASRepository/index.php?target=Public/Corpora/BROTHERS/BROTHERS.2.php) dataset.

<sup>3</sup> Model is not temporarily available for public use.

## Pretrained Model + Easy-Inference

You can find the pre-trained model on [Google Drive](https://drive.google.com/drive/folders/1R_gvC_St56Atxfu8MLRb1PIlBnBahta2?usp=sharing). Moreover, an easy-inference interface is available [here](./verification.py).


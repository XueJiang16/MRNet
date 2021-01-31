# MRNet

This is the code implementation of paper  *MRNet: a Multi-scale Residual Network for EEG-based Sleep Staging*, which has been submitted in PAKDD 2021. The repository has been anonymized for paper review.

### Environment Setup 

We recommend to setup the environment through `conda`.

```shell
$ conda env create -f environment.yml
```

### Data Preparation

The dataset Sleep-edf can be downloaded [here](https://physionet.org/content/sleep-edfx/1.0.0/).

```shell
$ python data_preprocess.py
```

### Training

We use TensorFlow 2.3 to build MRNet, which is trained on the NVIDIA GTX 1080Ti with the batch size of 128. The network is trained for 70 epochs with random initialization of the weights. We use the SGD optimizer with the momentum $= 0.9$.  We take 0.1 as the initial learning rate and reduce it by 10 times every 20 epochs. 

For training the network, run

```shell
$ python train.py
```

### Test

For testing the network, run

```shell
$ python test.py
```


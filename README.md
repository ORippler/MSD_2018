# Medical Segmentation Decathlon Contribution 2018

This repository contains the code for our submission to the MSD 2018.
For any questions concerning the code or submission, feel free to open an issue.

The code utilizes the [batchgenerators framework](https://github.com/MIC-DKFZ/batchgenerators/) maintained by DKFZ, that was modified
slightly to work with `tf.estimator` and `tf.data.Dataset` and include new features. In detail, `spatial_transformations.py` and `noise_augmentations.py` were modified.

It is further in part based on the [3D-Unet implementation from ellisdg](https://github.com/ellisdg/3DUnetCNN).

## Prerequisites

Based on the dataset, up to 12 GB of GPU RAM may be required.
As the data is continuously loaded from Disk, no requirements exist for CPU RAM.

To use the code, please setup a conda environment from the `environment.yaml`.

## Training and predicting a dataset

To train and predict on a challenge task, execute:

 * `dataset.py`
 * `train.py`
 * `predict.py`

in sucession.
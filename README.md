# Kaggle Facial Keypoints Detection Challenge

## Introduction

### Singapore Kaggle Machine Challenge Meetup Group

The [Singapore Kaggle Machine Learning Challenge Meetup group](https://www.meetup.com/Singapore-Kaggle-Machine-Learning-Challenge) organized the [first Kaggle meetup](https://www.meetup.com/Singapore-Kaggle-Machine-Learning-Challenge/events/245657152/) in Singapore on Jan 9 2018.

In this meeting, attendees formed teams with like-minded data scientists on Kaggle challenges of interest. In the following six weeks, the team will discuss the challenge, form a strategy and implement it. The outcome would then be presented to the audience at the Data Science Evening (our second meeting).

### Our Team

We are Team 12 (BestFitting). Our team consists of:
- Puay Ni Yi (leader)
- Teh Guo Pei
- Cedric Chee

We are tackling the [facial keypoints detection](https://www.kaggle.com/c/facial-keypoints-detection) as our first Kaggle challenge.

### Project

This is a 6-weeks project.

#### Plan

We will use this repo as the central location to host all the tutorials and solutions for the challenge.

## The Challenge

Facial Keypoints Detection is a challenge focused on Computer Vision field. The techniques to solve this challenge is usually from Deep Learning and Convolutional Neural Networks (CNN).

### Overview

The objective of this task is to detect and predict keypoint positions (locations) on face images. To learn more, take a look [here](https://www.kaggle.com/c/facial-keypoints-detection).

## Tutorial

### Deep Learning Tutorial

We are basing our tutorial from [Daniel Nouri's blog post](https://www.kaggle.com/c/facial-keypoints-detection#deep-learning-tutorial).

As we are planning to use TensorFlow for implementing our solution, we will follow this [tutorial by Alex Staravoitau](https://navoshta.com/facial-with-tensorflow/). Alex's tutorial was based on the amazing tutorial by Daniel Nouri.

Dependencies/Libraries used:
- [nolearn](https://github.com/dnouri/nolearn), a scikit-learn wrapper for Lasagne.
- Theano
- scikit-learn
- TensorFlow
- matplotlib
- pandas
- jupyter
- numpy

#### Installation and Setup

- Step 1 - install all dependencies:
```bash
$ git clone https://github.com/cedrickchee/kaggle-facial-detection.git
$ cd kaggle-facial-detection
$ pip install -r requirements.txt
```

##### Problems/issues encountered:
- Theano
    - Error `ValueError: You are tring to use the old GPU back-end. It was removed from Theano. Use device=cuda* now ...`. Solution on how to [converting to the new gpu back end(gpuarray)](https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29).
        - Either set the environment variable, `THEANO_FLAGS='device=cuda'` or
        - edit Theano config file, `~/.theanorc`
        ```bash
        [global]
        device = cuda
        ```
    - Error `(theano.gpuarray): pygpu was configured but could not be imported or is too old (version 0.7 or higher required)`. To resolve this problem, [install `libgpuarray` Python library](http://deeplearning.net/software/libgpuarray/installation.html).
        - In the middle of this process, at the step when you install `pygpu` by running this command, you will encounter new error `ModuleNotFoundError: No module named 'Cython'`. Work around this by installing Cython with this command: `pip install Cython`
        ```bash
        $ python setup.py build
        ```
    - Error `ImportError: libgpuarray.so.3: cannot open shared object file: No such file or directory` when you try to `import pygpu`. GitHub [thread](https://github.com/Theano/libgpuarray/issues/89#issuecomment-144826220) discussing this problem. [How to fix shared object file error](https://codeyarns.com/2014/01/14/how-to-fix-shared-object-file-error/). Append `/usr/local/lib` path to `LD_LIBRARY_PATH` in `.bashrc`
    ```bash
    LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
    ```
- nolearn, a sckit-learn wrapper for Lasagne
    - Error `ImportError: cannot import name 'downsample'` when trying to `import lasagne`. This can be solved [this way](https://github.com/Lasagne/Lasagne/issues/867). The [cause](https://github.com/Theano/Theano/issues/4337#issuecomment-332041284) of the problem.
    ```bash
    $ pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
    
    ```

## Solution

Jupyter Notebook with Cedric's attempts to tackle the competition is in the [notebooks](notebooks) folder.

1. First model: [a single hidden layer](notebooks/1_single_hidden_layer.ipynb)
    - A very simple neural network (NN).
2. Second model: [convolutions](notebooks/2_convolutions.ipynb)
    - Convolutional neural network (CNN) with data augmentation, learning rate decay and dropout.
3. Third model: [training specialists](notebooks/3_training_specialists.ipynb)
    - A pipeline of specialist CNNs with early stopping and supervised pre-training.

## Results

Ranking on Leaderboard among 175 teams.

| Team Member   | Private Score        | Public Score         | Best Model |
| ------------- | -------------------- | -------------------- | ---------- |
| Cedric        | 1.96686 (26th place) | 2.15043 (16th place) | #3         |

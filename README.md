# SpeechCommAI
> SpeechCommAI is a console program for predicting up to 35 spoken words.

## Table of Contents
* [Introduction](#Introduction)
* [Requirements](#Requirements)
* [Setup](#setup)
* [Usage](#usage)
* [Technologies used](#technologies-used)
* [Screenshots](#screenshots)
* [Contact](#contact)


## Introduction
This project can be used to recognize speech commands, recorded in real-time. Prediction is made by a convolutional neural network based on Keras, learned on the [TensorFlow database](https://www.tensorflow.org/datasets/catalog/speech_commands).
Using special options, users can download a dataset, preprocess it and teach the program themselves.


### List of the speech commands
- backward
- bed
- bird
- cat
- dog
- down
- eight
- five
- follow
- forward
- four
- go
- happy
- house
- learn
- left
- marvin
- nine
- no
- off
- on
- one
- right
- seven
- sheila
- six
- stop
- three
- tree
- two
- up
- visual
- wow
- yes
- zero


## Requirements
- python (3.7+)
- pip
- required libraries listet in the `requirements.txt` file

## Setup
You can install requirements by the follwing command:
```
pip install -r requirements.txt
```

## Usage
To use this program run `main.py` in the command line:
```
python main.py <option>
```
As an `<option>` one of the following can be selected:
`D` – Download the raw dataset
`P` – Pre-process the dataset
`T` – Train the model
`L` – Live record

In addition, the `T` and `L` options can take one more optional argument specifying the type of set of words to be trained. The word sets are available in the `config.toml` file. If the name of the set is not specified, the program will accept the entire set of 35 words by default.


## Technologies used
Machine learning:
- TensorFlow - version 2.9.1
- Keras - version 2.9.0

Audio processing:
- librosa - version 0.9.1
- PyAudio - version 0.2.12

Plots:
- matplotlib - version 3.5.1
- scikit-learn - version 1.1.2

## Screenshots
Visualization of the learned network (confusion matrix):

![confusion matrix](https://gcdnb.pbrd.co/images/kkaaZCLfopNO.png?o=1)


The model's learning history:

![history](https://gcdnb.pbrd.co/images/Ee8q1CdoEGiI.png?o=1)


## Contact
Created by [@miolows](https://github.com/miolows) - feel free to contact me!


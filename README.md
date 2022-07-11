# SpeechCommAI
> This is a speech recognition program, written in Python, that can predict up to 35 spoken words.
> It can work on live recorded data.


## Table of Contents
* [Introduction](#Introduction)
* [Setup](#setup)
* [Usage](#usage)
* [Technologies used](#technologies-used)
* [Screenshots](#screenshots)
* [Project status](#project-status)
* [Contact](#contact)


## Introduction
The goal of the project was to create a convolutional neural network, teach it to recognize 35 words (speech commands) from the used [TensorFlow database](https://www.tensorflow.org/datasets/catalog/speech_commands), and implement live audio data acquisition to recognize speech recorded by the user.


## Setup
The project requirements are listed in the requirements.txt file. To install them type in the command line:
```
pip install -r requirements.txt
```


## Usage
To use this program run `main.py` in your IDE or by the command line:
```
python main.py
```
after the message appears, the user can say a word from the list of learned patterns, and the program will try to predict it and list it along with the percentage of matching. If no pattern has been selected with satisfactory certainty (less than 50%), the program will tell the user to ignore the signal.

List of patterns:
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




## Technologies used
Mchine learning:
- TensorFlow - version 2.3.0
- Keras - version 2.4.3

Audio processing:
- librosa - version 0.9.1
- PyAudio- version 0.2.11

Plots:
- matplotlib - version 3.5.1
- scikit-learn - version 1.0.2


## Screenshots
Visualization of the learned network (confusion matrix)
![confusion matrix](https://i.postimg.cc/G4cxybcp/Confusion-Matrix.png)

The model's learning history
![history](https://i.postimg.cc/SnD7LHsf/Model-history.png)


## Project status
Project is: _in progress_


## Contact
Created by [@miolows](https://github.com/miolows) - feel free to contact me!


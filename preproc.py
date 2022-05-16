import numpy as np
import librosa
import os

from dirmaker import DirMaker

data_dir = 'Dataset/'
labels = os.listdir(data_dir) #[:classes]
prep_data_dir = 'mfcc_Dataset/'

dm = DirMaker(labels, prep_data_dir)

nr_of_samples = 200
sr = 22050              #sampling rate of audio data
n_mfcc=44               #number of MFCCs to return

for l, label in enumerate(labels):
    
    data_path = os.path.join(data_dir, label)
    prep_path = os.path.join(prep_data_dir, label)
    audio_files = os.listdir(data_path)
    
    time_audio_files = []
    print("Klasa: " + str(l))
    for j, audio in enumerate(audio_files):
        #mfcc file name with the same name of the audio file (wav -> dat)
        audiofile_path = os.path.join(data_path, audio)
        prepfile_name = audio[:-3] + 'txt'
        prepfile_path = os.path.join(prep_path, prepfile_name)
        
        y, _ = librosa.load(audiofile_path, sr)
        
        #if the audio is shorter than 1s, fill up with 0
        y_reshaped = np.zeros(sr)
        y_reshaped[:y.shape[0]] = y
        
        # Compute MFCC features from the raw signal
        mfcc = librosa.feature.mfcc(y=y_reshaped, sr=sr, n_mfcc=n_mfcc)
        
        #Normalization of data (min=0.0; max=1.0)
        mfcc_min = mfcc.min()
        mfcc = np.array(list(map(lambda x: x-mfcc_min, mfcc)))
        mfcc_max = mfcc.max()
        mfcc = np.array(list(map(lambda x: x/mfcc_max, mfcc)))
        
        
        
        with open(prepfile_path,"w+") as f:
            np.savetxt(f, mfcc, fmt='%.8f')
import numpy as np
import librosa
from librosa.effects import time_stretch
import os
import json
from dataclasses import dataclass, asdict
import hashlib
import re
from tqdm import tqdm
import functools
import time

from config import Configurator

def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer

@dataclass
class PrepAudioData:
    ''' Data class of pre-processed and saved data '''
    class_name: str
    mfcc: []
    
@dataclass
class AIAudioData:
    ''' Data class of data loaded from pre-processed data files and passed to the AI module '''
    class_name: []
    # In order to give flexibility in the selection of types and number of 
    # analyzed classes, labels are created during data reading
    labels: []
    mfcc: []


''' *** Audio processing *** '''

def get_mfcc(audio, rate, duration, n_mfcc):
    #ensure that the audio sample has the specified duration. Otherwise, compress or stretch the signal.
    audio = adjust_audio(audio, rate, duration)
    mfcc = librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=n_mfcc)
    return mfcc

def file_mfcc(file, rate, duration, n_mfcc):
    audio, _ = librosa.load(file, sr=rate)
    mfcc = get_mfcc(audio, rate, duration, n_mfcc)
    return mfcc

def adjust_audio(y, sample_rate, output_duration):
    input_duration = len(y)/sample_rate
    if input_duration == output_duration:
        return y
    else:
        r = input_duration/output_duration
        y_adjusted = librosa.effects.time_stretch(y, rate=r)
        return y_adjusted

def get_delta_mfcc(mfcc, d_order):
    delta_mfcc = librosa.feature.delta(mfcc, order=d_order)
    return delta_mfcc

def norm(matrix):
    xmax, xmin = matrix.max(), matrix.min()
    return (matrix - xmin)/(xmax - xmin)


''' *** Data pre-processing *** '''
def which_set(filename, max_per_class, validation_percentage, testing_percentage):
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    
    h = hashlib.sha1()
    h.update(bytes(hash_name, encoding='utf-8'))
    hash_name_hashed = h.hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (max_per_class + 1))
                       * (100.0 / max_per_class))
    
    if percentage_hash < validation_percentage:
      result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
      result = 'testing'
    else:
      result = 'training'
    return result


def save_tofile(dirpath, filename, data):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
        
    path = os.path.join(dirpath, filename)
    with open(path ,"w+") as f:
        json.dump(asdict(data), f, indent=1)


def prep_class(prep_dir, class_path, audio_files,  max_per_class, validation_percentage, 
               testing_percentage,sample_rate, audio_dur, n_mfcc):
    class_name = os.path.basename(class_path)
    class_prep_dir = os.path.join(prep_dir, class_name)
    print("Processing {}".format(class_name))

    validation = PrepAudioData(class_name, [])
    testing = PrepAudioData(class_name, [])
    training = PrepAudioData(class_name, [])
    
    split_data = {'validation': validation,
                  'testing': testing,
                  'training': training}
                  
    for audio_file in tqdm(audio_files):
        audiofile_path = os.path.join(class_path, audio_file)
        mfcc = file_mfcc(audiofile_path, sample_rate, audio_dur, n_mfcc)
        which = which_set(audio_file, max_per_class, validation_percentage, testing_percentage)
        split_data[which].mfcc.append(mfcc.tolist())

    valid_fname, test_fname, train_fname = "validation.json", "testing.json", "training.json"
            
    save_tofile(class_prep_dir, valid_fname, validation)
    save_tofile(class_prep_dir, test_fname, testing)
    save_tofile(class_prep_dir, train_fname, training)
        
        
def prep_dataset(config):
    dataset = config.get('directories', 'Dataset')
    prep_dir = config.get('directories', 'Preprocessed data')
    sample_rate = config.get('audio', 'rate')
    duration = config.get('audio', 'prep duration')
    n_mfcc = config.get('audio', 'mfcc coefficients')
    
    s_max = config.get('preprocessing', 'sample max')
    v_perc = config.get('preprocessing', 'validation percentage')
    t_perc = config.get('preprocessing', 'testing percentage')
                             
    for class_path, subdirs, audio_files in os.walk(dataset):
        curr_dir = os.path.basename(class_path)
        if curr_dir != dataset:
            prep_class(prep_dir, class_path, audio_files, s_max, v_perc, t_perc, sample_rate, duration, n_mfcc)


''' *** Data loading from JSON files *** '''
def load_set(set_name, label_path, index, output):
    set_file = os.path.join(label_path, (set_name+".json"))
    
    with open(set_file, "r") as jsonFile:
        data = json.load(jsonFile)
        
        labels = [index]*len(data['mfcc'])
        output['labels'].append(labels)
        for key, value in data.items():
            output[key].append(value)

    return output
    

@timer
def load_data(data_dir, labels):
    validation = asdict(AIAudioData([], [], []))
    testing = asdict(AIAudioData([], [], []))
    training = asdict(AIAudioData([], [], []))
    
    for i, label in tqdm(enumerate(labels)):
        class_dir = os.path.join(data_dir, label)
        validation = load_set("validation", class_dir, i, validation)
        testing = load_set("testing", class_dir, i, testing)
        training = load_set("training", class_dir, i, training)

    return (validation, testing, training)
        


if __name__ == "__main__":
    c = Configurator()
    prep_dataset(c)
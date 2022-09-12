from dataclasses import dataclass, asdict, field
from tqdm import tqdm, asyncio
import os, re
import pathlib
import json
import hashlib
import tomli
import pandas as pd
from typing import List
import concurrent.futures
import random

import speechcommai.audio.audio as audio
from speechcommai.wrap import timer, mkdir


def which_set(filename, sets, weights):
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_number = int(hash_name, 16) #int from hex str
    random.seed(hash_number) #set the random number generator
    result = random.choices(sets, weights)[0] #1 element list
    return result


def sort_wav_files(dirpath, sets, weights):
    sets_list = [[] for _ in range(len(sets))]
    sorting = {s:sl for s, sl in zip(sets, sets_list)}
    for file in os.listdir(dirpath):
        which = which_set(file, sets, weights)
        sorting[which].append(file)
    return sets_list


def save_set(output_dir, class_name, set_name, set_data):
    save_data = {'class_name': class_name, 'mfcc': set_data}
    save_file = os.path.join(output_dir, f'{set_name}.json')
    with open(save_file ,"w+") as f:
        json.dump(save_data, f, indent=1)


def prep_set_files(input_dir, audio_file, *args):
    file_path = os.path.join(input_dir, audio_file)
    data = audio.get_file_mfcc(file_path, *args)
    return data.tolist()


@mkdir
def prep_class(output_dir, input_dir, set_names, set_values, *args):
    class_name = os.path.basename(input_dir)
    sorted_files = sort_wav_files(input_dir, set_names, set_values)
    for i, fset in enumerate(sorted_files):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(prep_set_files, input_dir, audio_file, *args)
                       for audio_file in fset]
            set_data = [f.result() for f in tqdm(concurrent.futures.as_completed(results), total=len(fset), desc=f'{class_name} ({set_names[i]})')]
        
        save_set(output_dir, class_name, set_names[i], set_data)

@mkdir
def prep_dataset(output_dir, input_dir, *audio_args, **sets):
    sets_n = list(sets.keys())
    sets_val = list(sets.values())
    for class_dataset in os.listdir(input_dir):
        output_d = os.path.join(output_dir, class_dataset)
        input_d = os.path.join(input_dir, class_dataset)
        prep_class(output_d, input_d, sets_n, sets_val, *audio_args)



        
# def which_set(filename, max_per_class, validation_percentage, testing_percentage):
#     base_name = os.path.basename(filename)
#     hash_name = re.sub(r'_nohash_.*$', '', base_name)
    
#     h = hashlib.sha1()
#     h.update(bytes(hash_name, encoding='utf-8'))
#     hash_name_hashed = h.hexdigest()
#     percentage_hash = ((int(hash_name_hashed, 16) % (max_per_class + 1)) * (100.0 / max_per_class))
    
#     if percentage_hash < validation_percentage:
#       result = 0
#     elif percentage_hash < (testing_percentage + validation_percentage):
#       result = 1
#     else:
#       result = 2
#     return result





"""
@dataclass
class PrepAudioData:
    ''' Data class of pre-processed and saved data '''
    class_name: str
    mfcc: List = field(default_factory=lambda: [])
    

''' *** Data pre-processing *** '''
@timer
def save(dirpath, filename, data):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    
    path = os.path.join(dirpath, filename)        
    with open(path ,"w+") as f:
        json.dump(asdict(data), f, indent=1)


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


def prep_class(prep_dir, class_path, audio_files, sample_rate, audio_dur, n_mfcc, hop_l,
               max_per_class=2**27, validation_percentage=15, testing_percentage=15):
    
    class_name = os.path.basename(class_path)
    class_prep_dir = os.path.join(prep_dir, class_name)
    print("Processing {}".format(class_name))
    
    validation = PrepAudioData(class_name)
    testing = PrepAudioData(class_name)
    training = PrepAudioData(class_name)    
    
    data_names = ["validation.json", "testing.json", "training.json"]
    data_holders = [PrepAudioData(class_name)]*3

    split_data = {'validation': validation,
                  'testing': testing,
                  'training': training}
                  
    for audio_file in tqdm(audio_files):
        audiofile_path = os.path.join(class_path, audio_file)
        mfcc = audio.get_file_mfcc(audiofile_path, sample_rate, audio_dur, n_mfcc, hop_l)
        which = which_set(audio_file, max_per_class, validation_percentage, testing_percentage)
        split_data[which].mfcc.append(mfcc.tolist())

    # for dn, dh in zip(data_names, data_holders):
        # save(class_prep_dir, dn, dh)
    
    save(class_prep_dir, data_names[0], validation)
    save(class_prep_dir, data_names[1], testing)
    save(class_prep_dir, data_names[2], training)

    

@timer 
def prep_dataset():
    with open("config.toml", mode="rb") as fp:
        config = tomli.load(fp)
    
    dir_d = config['directories']
    audio_d = config['audio']
    
    dataset = dir_d['dataset']
    prep_dir = dir_d['preprocessed_data']
    sample_rate = audio_d['rate']
    duration = audio_d['duration']
    n_mfcc = audio_d['mfcc']
    hop_l = audio_d['hop_length']
    
    if not os.path.exists(prep_dir):
        os.mkdir(prep_dir)

    for class_path, subdirs, audio_files in os.walk(dataset):
        curr_dir = os.path.basename(class_path)
        if curr_dir != dataset:
            prep_class(prep_dir, class_path, audio_files, sample_rate, duration, n_mfcc, hop_l)

if __name__ == "__main__":
    prep_dataset()
    
    
"""

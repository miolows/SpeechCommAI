from dataclasses import dataclass, asdict
from tqdm import tqdm
import os, re
import json
import hashlib
import tomli

import speechcommai.audio.audio as audio
from speechcommai.wrap import timer

@dataclass
class PrepAudioData:
    ''' Data class of pre-processed and saved data '''
    class_name: str
    mfcc: []
    

''' *** Data pre-processing *** '''
def save(dirpath, filename, data):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
        
    path = os.path.join(dirpath, filename)        
    with open(path ,"w+") as f:
        json.dump(asdict(data), f, indent=1)


def which_set(filename, max_per_class=2**27, validation_percentage=15, testing_percentage=15):
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

    validation = PrepAudioData(class_name, [])
    testing = PrepAudioData(class_name, [])
    training = PrepAudioData(class_name, [])
    
    split_data = {'validation': validation,
                  'testing': testing,
                  'training': training}
                  
    for audio_file in tqdm(audio_files):
        audiofile_path = os.path.join(class_path, audio_file)
        mfcc = audio.get_file_mfcc(audiofile_path, sample_rate, audio_dur, n_mfcc, hop_l)
        which = which_set(audio_file, max_per_class, validation_percentage, testing_percentage)
        split_data[which].mfcc.append(mfcc.tolist())

    valid_fname, test_fname, train_fname = "validation.json", "testing.json", "training.json"
            
    save(class_prep_dir, valid_fname, validation)
    save(class_prep_dir, test_fname, testing)
    save(class_prep_dir, train_fname, training)
        
        
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
import os, re
import random
import json
from tqdm import tqdm
import concurrent.futures

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
            set_data = [f.result() for f in tqdm(concurrent.futures.as_completed(results), 
                                                 total=len(fset), 
                                                 desc=f'{class_name} ({set_names[i]})')]
        save_set(output_dir, class_name, set_names[i], set_data)


@mkdir
def prep_dataset(output_dir, input_dir, *audio_args, **sets):
    sets_n = list(sets.keys())
    sets_val = list(sets.values())
    for class_dataset in os.listdir(input_dir):
        output_d = os.path.join(output_dir, class_dataset)
        input_d = os.path.join(input_dir, class_dataset)
        prep_class(output_d, input_d, sets_n, sets_val, *audio_args)
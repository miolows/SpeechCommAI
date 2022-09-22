import concurrent.futures
from dataclasses import dataclass
from tqdm import tqdm, asyncio
import json
import os

from speechcommai.wrap import timer


def load_data(data_dir, classes, set_name):
    set_data = []
    set_labels = []
    
    for idx, c in tqdm(enumerate(classes)):
        path = os.path.join(data_dir, c,  f'{set_name}.json')
        with open(path, "r") as json_file:
            file_data = json.load(json_file)
            data = file_data['mfcc']
            label = [idx]*len(data)
            
            set_data += data
            set_labels += label

    return set_data, set_labels

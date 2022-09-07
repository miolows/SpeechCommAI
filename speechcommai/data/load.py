import concurrent.futures
from dataclasses import dataclass
from tqdm import tqdm, asyncio
import json
import os

from speechcommai.wrap import timer


@dataclass
class AIAudioData:
    ''' Data class of data loaded from pre-processed data files and passed to the AI module '''
    class_name: []
    # In order to give flexibility in the selection of types and number of 
    # analyzed classes, labels are created during data loading
    labels: []
    mfcc: []
    
    def append(self, other):
        self.class_name.append(other.class_name)
        self.labels.append(other.labels)
        self.mfcc.append(other.mfcc)


''' *** Data loading from JSON files *** '''
def load_set(set_path, index):
    with open(set_path, "r") as json_file:
        data = json.load(json_file)
        class_name = data['class_name']
        mfcc = data['mfcc']
        labels = [index] * len(mfcc)
        set_data = AIAudioData(class_name, labels, mfcc)
        
        return set_data

@timer
def load_data(data_dir, labels):
    classes = list(map(lambda x: os.path.join(data_dir, x), labels))
    sets = ['validation', 'testing', 'training']
    output = [AIAudioData([],[],[]) for _ in range(len(sets))]
    
    for idx, set_name in enumerate(sets):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            set_data = [executor.submit(load_set, os.path.join(class_path, f'{set_name}.json'), index) 
                        for index, class_path in enumerate(asyncio.tqdm(classes, desc=f"Loading {set_name} data"))]
            
            for f in concurrent.futures.as_completed(tqdm(set_data, desc='data packing')):
                output[idx].append(f.result())

    return output
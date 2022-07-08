import concurrent.futures
from dataclasses import dataclass, asdict, fields
from tqdm.asyncio import tqdm
import json
import os

from wrap import timer


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
def load_set(class_path, set_name, index):
    set_file = os.path.join(class_path, f'{set_name}.json')
    with open(set_file, "r") as jsonFile:
        data = json.load(jsonFile)
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
        print(f"Loading {set_name} data")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            set_data = [executor.submit(load_set, class_path, set_name, index) 
                        for index, class_path in enumerate(tqdm(classes))]
            
            for f in concurrent.futures.as_completed(set_data):
                output[idx].append(f.result())
    return output
import concurrent.futures
from dataclasses import dataclass, asdict, fields
from tqdm import tqdm
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
        labels = [index]*len(mfcc)
        set_data = AIAudioData(class_name, labels, mfcc)
        
        return set_data
    

def load_class_data(index, class_path):
    sets = ['validation', 'testing', 'training']
    class_data = [load_set(class_path, s, index) for s in sets]
    return class_data

@timer
def load_data(data_dir, labels):
    # validation = AIAudioData([], [], [])
    # testing = AIAudioData([], [], [])
    # training = AIAudioData([], [], [])
    
    classes = list(map(lambda x: os.path.join(data_dir, x), labels))
    
    sets = ['validation', 'testing', 'training']
    output = [AIAudioData([],[],[]) for _ in range(len(sets))]
    
    for idx, data_set in tqdm(enumerate(sets)):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            set_list = [executor.submit(load_set, class_path, data_set, index) for index, class_path in enumerate(classes)]
            for f in concurrent.futures.as_completed(set_list):
                output[idx].append(f.result())
    return output

# @timer
# def load_data(data_dir, labels):
#     validation = AIAudioData([], [], [])
#     testing = AIAudioData([], [], [])
#     training = AIAudioData([], [], [])
    
#     classes = list(map(lambda x: os.path.join(data_dir, x), labels))
    
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = [executor.submit(load_class_data, index, class_path) for index, class_path in enumerate(classes)]
        
#         for f in concurrent.futures.as_completed(results):
#             print(len(f.result()))
#             pass
            
#     return (validation, testing, training)



# @timer
# def load_data(data_dir, labels):
#     validation = asdict(AIAudioData([], [], []))
#     testing = asdict(AIAudioData([], [], []))
#     training = asdict(AIAudioData([], [], []))
    
#     for i, label in tqdm(enumerate(labels)):
#         class_dir = os.path.join(data_dir, label)
#         validation = load_set("validation", class_dir, i, validation)
#         testing = load_set("testing", class_dir, i, testing)
#         training = load_set("training", class_dir, i, training)

#     return (validation, testing, training)
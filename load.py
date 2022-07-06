from dataclasses import dataclass, asdict
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
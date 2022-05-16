import re, os, hashlib
import random
import time

import functools
import numpy as np

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





MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage, testing_percentage):
    for i in range(100):
        """Determines which data partition the file should belong to.
        
        We want to keep files in the same training, validation, or testing sets even
        if new ones are added over time. This makes it less likely that testing
        samples will accidentally be reused in training when long runs are restarted
        for example. To keep this stability, a hash of the filename is taken and used
        to determine which set it should belong to. This determination only depends on
        the name and the set proportions, so it won't change as other files are added.
        
        It's also useful to associate particular files as related (for example words
        spoken by the same person), so anything after '_nohash_' in a filename is
        ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
        'bobby_nohash_1.wav' are always in the same set, for example.
        
        Args:
          filename: File path of the data sample.
          validation_percentage: How much of the data set to use for validation.
          testing_percentage: How much of the data set to use for testing.
        
        Returns:
          String, one of 'training', 'validation', or 'testing'.
        """
        base_name = os.path.basename(filename)
        # We want to ignore anything after '_nohash_' in the file name when
        # deciding which set to put a wav in, so the data set creator has a way of
        # grouping wavs that are close variations of each other.
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        
        # This looks a bit magical, but we need to decide whether this file should
        # go into the training, testing, or validation sets, and we want to keep
        # existing files in the same set even if more files are subsequently
        # added.
        # To do that, we need a stable way of deciding based on just the file name
        # itself, so we do a hash of that and then use that to generate a
        # probability value that we use to assign it.
        h = hashlib.sha1()
        h.update(bytes(hash_name, encoding='utf-8'))
        hash_name_hashed = h.hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) %
                            (MAX_NUM_WAVS_PER_CLASS + 1)) *
                            (100.0 / MAX_NUM_WAVS_PER_CLASS))
        if percentage_hash < validation_percentage:
          result = 'validation'
        elif percentage_hash < (testing_percentage + validation_percentage):
          result = 'testing'
        else:
          result = 'training'
    return result

def another_which_set(filename, validation_percentage, testing_percentage):
    for i in range(100):

        results = ['validation', 'testing', 'training']
        weights = [validation_percentage, testing_percentage, 100-(validation_percentage+testing_percentage)]
    
        base_name = os.path.basename(filename)
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        
        random.seed(bytes(hash_name, encoding='utf-8'))
          
        result = random.choices(results,weights)[0]

    return result


datadir = 'Dataset'
data_path = os.path.join(os.getcwd(), datadir)
labels = os.listdir(data_path)[:-1]
for i in labels:
    word_path = os.path.join(data_path, i)
    word_data = os.listdir(word_path)
    
    for w in word_data:
        print(which_set(w, 20, 60))



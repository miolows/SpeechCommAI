import numpy as np
import librosa


''' *** Audio processing *** '''
def get_mfcc(audio, rate, duration, n_mfcc):
    #ensure that the audio sample has the specified duration. Otherwise, compress or stretch the signal.
    audio = adjust_audio(audio, rate, duration)
    mfcc = librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=n_mfcc)
    return mfcc

def get_file_mfcc(file, rate, duration, n_mfcc):
    audio, _ = librosa.load(file, sr=rate)
    mfcc = get_mfcc(audio, rate, duration, n_mfcc)
    return mfcc

def adjust_audio(sample, sample_rate, output_duration):
    sample_len = len(sample)
    input_duration = sample_len/sample_rate
    
    if input_duration < output_duration:
        sample_long = np.zeros(output_duration*sample_rate)
        sample_long[:sample_len] = sample
        return sample_long
    
    elif input_duration > output_duration:
        r = input_duration/output_duration
        sample_short = librosa.effects.time_stretch(sample, rate=r)
        return sample_short
    else:
        return sample
    

def get_delta_mfcc(mfcc, d_order):
    delta_mfcc = librosa.feature.delta(mfcc, order=d_order)
    return delta_mfcc
import numpy as np
import pyaudio
import librosa
import librosa.display
import wave
import matplotlib.pyplot as plt
import os
import queue
from multiprocessing import Process, Queue
import threading
from prep import get_mfcc, timer
from config import Configurator
import soundfile as sf
import time
from prep import timer
class AudioRecord(object):
    def __init__(self, config, record_queue):
        self.format = pyaudio.paInt16
        self.chunk = 3024
        self.channels = 1       
        self.rate = config.get('audio', 'rate')
        self.mfcc_n = config.get('audio', 'mfcc coefficients')
        self.prep_duration = config.get('audio', 'prep duration')
        self.output = config.get('directories', 'Temporary files')
        self.threshold = 0.5
        rec_name = 'rec.wav'
        self.file_path = os.path.join(self.output, rec_name)
        
        self.record_q = record_queue
        # self.data_q = queue.Queue()
        # self.counter = 0
    
    
    ''' *** .wav file I/O methods *** '''
    def save_record(self, frame):
        wf = wave.open(self.file_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(frame)
        wf.close()
        

    def clear_temp_file(self):
        os.remove(self.file_path)    
    
    
    ''' *** Processed audio data queue  *** '''    
    # def get_audio_data(self):
    #     data = self.data_q.get()
    #     self.data_q.task_done()
    #     return data
    
    # def data_available(self):
    #     return not self.data_q.empty() 

    
    ''' *** Audio data processing  *** ''' 
    # @timer
    def process_signal(self, signal):
        #Split an audio signal into non-silent intervals
        non_silent = librosa.effects.split(signal, top_db=40)
        samples_num = non_silent.shape[0]
        feedback = []

        for s in range(samples_num):
            s_start = non_silent[s,0]
            s_stop = non_silent[s,1]
            
            if s_stop == len(signal):
                #if the sample ends with the end of a whole signal, assume that
                #it is truncated and pass it to feedback
                feedback = signal[s_start:]
            else:
                #if the sample ends within a signal, proceed processing
                feedback = []
                sample = signal[s_start:s_stop]
                
                if np.max(sample) > self.threshold:
                    mfcc = get_mfcc(sample, self.rate, self.prep_duration, self.mfcc_n)
                    self.data_q.put(mfcc)
            
        return feedback

                

    
    # @timer
    # def process_sample(self, buffer=[]):
    #     frame = self.record_q.get()
    #     #extend the signal with a buffer of a potentially truncated sample from the previous frame
    #     data = np.concatenate((buffer, frame))
        
    #     split = librosa.effects.split(data, top_db=40)
    #     samples_num = split.shape[0]
    #     samples = []
    #     for s in range(samples_num):
    #         s_start = split[s,0]
    #         s_stop = split[s,1]
    #         #determine whether the sample is truncated
    #         margin = s_stop/len(data)
    #         if margin == 1.0:
    #             self.process_sample(data[s_start:])

    #         else:
    #             # print('data:')
    #             # print(s_start, s_stop)
    #             sample = data[s_start:s_stop]
    #             #accept samples with significant amplitude
    #             # print(f'Max: {np.max(sample)}')
    #             if np.max(sample) >= self.threshold:
    #                 samples.append(sample)

    #     #if the list of accepted samples is not empty
    #     if samples:
    #         self.counter += 1
    #         #''' signal plotting for testing '''
    #         fig, ax = plt.subplots(nrows=(len(samples)+1), sharex=True)
    #         librosa.display.waveshow(data, sr=self.rate, ax=ax[0])
    #         for idx, s in enumerate(samples):
                
    #             # sf.write(f'{self.counter} - {idx}.wav', s, self.rate)
    #             librosa.display.waveshow(s, sr=self.rate, ax=ax[idx+1])
    #             mfcc = get_mfcc(sample, self.rate, self.prep_duration, self.mfcc_n)
    #             # print('self.record_q: ', self.record_q.qsize())
    #             # print('self.data_q:', self.data_q.qsize())

    #             self.data_q.put(mfcc)

    #     self.process_sample([])


    def live(self):
       feedback = []
       while True:
           frame = self.record_q.get()
           #extend the signal with a buffer of a potentially truncated sample from the previous frame
           frames = np.concatenate((feedback, frame))
           feedback = self.process_signal(frames)


    ''' *** Recording *** '''   
    def callback(self, in_data, frame_count, time_info, status):
        data = np.frombuffer(in_data, dtype=np.int16) / 32768.0
        self.record_q.put(data)

        return (in_data, pyaudio.paContinue)


    def start_recording(self):
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(format=self.format,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.chunk,
                                        stream_callback=self.callback)
        
        # threading.Thread(target=self.live, daemon=False).start()



    def stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()


    
if __name__ == '__main__':
    c = Configurator()
    audio = AudioRecord(c)
    audio.start_recording()
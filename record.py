import numpy as np
import pyaudio
import time
import librosa
import librosa.display
import wave
import matplotlib.pyplot as plt
import os
import queue
import threading
from prep import get_mfcc
from config import Configurator

class AudioRecord(object):
    def __init__(self, config, data_queue):
        self.format = pyaudio.paInt16
        self.chunk = 3024
        self.channels = 2        
        self.rate = config.get('audio', 'rate')
        self.rec_duration = config.get('audio', 'record duration')
        self.prep_duration = config.get('audio', 'prep duration')
        self.mfcc_n = config.get('audio', 'mfcc coefficients')
        self.output = config.get('directories', 'Temporary files')
        self.threshold = 0.6
        rec_name = 'rec.wav'
        self.file_path = os.path.join(self.output, rec_name)
        
        self.record_q = queue.Queue()
        self.data_q = data_queue
        
    def start_recording(self):
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(format=self.format,
                                        channels=self.channels,
                                        rate=self.rate,
                                        input=True,
                                        frames_per_buffer=self.chunk,
                                        stream_callback=self.callback)
        
        threading.Thread(target=self.live, daemon=True).start()


    
    def callback(self, in_data, frame_count, time_info, status):
        self.record_q.put(in_data)
        return (in_data, pyaudio.paContinue)
    
    
    def stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()
        
        self.record_q.join()


    def save_record(self, frame):
        wf = wave.open(self.file_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(frame)
        wf.close()

    def get_audio_data(self):
        data = self.data_q.get()
        self.data_q.task_done()
        return data

    def clear_temp_file(self):
        os.remove(self.file_path)


    def process_sample(self, buffer):
        y, _ = librosa.load(self.file_path, sr=self.rate)
        #extend the signal with a buffer of a potentially truncated sample from the previous frame
        ext_y = np.concatenate((buffer, y))
        #delete temporary record
        self.clear_temp_file()
        #Split an audio signal into non-silent intervals
        split = librosa.effects.split(ext_y)
        samples_num = split.shape[0]
        samples = []
        for s in range(samples_num):
            s_start = split[s,0]
            s_stop = split[s,1]
            
            margin = s_stop/len(ext_y)
            # print(margin)
            if margin > 0.98:
                return ext_y[s_start:]
                break
            else:
                sample = ext_y[s_start:s_stop]
                #accept samples with significant amplitude
                if np.max(sample) >= self.threshold:
                    samples.append(sample)

        #if the list of accepted samples is not empty
        if samples:
            # fig, ax = plt.subplots(nrows=(len(samples)+1), sharex=True)
            # librosa.display.waveshow(ext_y, sr=self.rate, ax=ax[0])
            ##########
            for idx, s in enumerate(samples):
                # librosa.display.waveshow(s, sr=self.rate, ax=ax[idx+1])
                mfcc = get_mfcc(sample, self.rate, self.prep_duration, self.mfcc_n)
                self.data_q.put(mfcc)

                
        return []


    def live(self):
       buffer = []
       while True:
           frame = self.record_q.get()
           self.save_record(frame)
           buffer = self.process_sample(buffer)
           self.record_q.task_done()
    


if __name__ == '__main__':
    c = Configurator()
    audio = AudioRecord(c)
    audio.start_recording()

    

# '''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# import numpy as np
# import pyaudio
# import time
# import librosa

# class AudioRecord(object):
#     def __init__(self):
#         self.FORMAT = pyaudio.paFloat32
#         self.CHANNELS = 1
#         self.RATE = 22050
#         self.CHUNK = 22050
#         self.p = None
#         self.stream = None
#         self.audio_data = None

#     def start(self):
#         self.p = pyaudio.PyAudio()
#         self.stream = self.p.open(format=self.FORMAT,
#                                   channels=self.CHANNELS,
#                                   rate=self.RATE,
#                                   input=True,
#                                   output=False,
#                                   stream_callback=self.callback,
#                                   frames_per_buffer=self.CHUNK)

#     def stop(self):
#         self.stream.close()
#         self.p.terminate()

#     def callback(self, in_data, frame_count, time_info, flag):
#         y = np.frombuffer(in_data, dtype=np.float32)
#         mfcc = librosa.feature.mfcc(y=y, n_mfcc=44)
        
#         mfcc_min = mfcc.min()
#         mfcc = np.array(list(map(lambda x: x-mfcc_min, mfcc)))
#         mfcc_max = mfcc.max()
#         mfcc = np.array(list(map(lambda x: x/mfcc_max, mfcc)))
#         self.audio_data = mfcc
#         return None, pyaudio.paContinue
    
    
    

#     def mainloop(self):
#         time.sleep(1.0)
#         # while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
#         #     time.sleep(2.0)


#     def record(self):
#         self.start()
#         print("Talk to me...")
#         time.sleep(1.5)    
#         self.stop()
#         print("Stop")

# if __name__ == '__main__':

#     audio = AudioRecord()
#     audio.start()     # open the the stream
#     audio.mainloop()
#     audio.stop()



# # import pyaudio
# # import struct
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import time

# # %matplotlib tk

# # CHUNK = 1024 * 4
# # FORMAT = pyaudio.paInt16
# # CHANNELS = 1
# # RATE = 44100


# # audio = pyaudio.PyAudio()

# # stream = audio.open(format = FORMAT,
# #                     channels = CHANNELS,
# #                     rate = RATE,
# #                     input = True,
# #                     output = True,
# #                     frames_per_buffer = CHUNK)



# # fig, ax = plt.subplots()
# # x = np.arange(0, 2*CHUNK, 2)
# # line, = ax.plot(x, np.zeros((CHUNK)))
# # ax.set_ylim(-5*255,5*255)
# # ax.set_xlim(0, CHUNK)
# # while True:
# #     data = stream.read(CHUNK)
# #     int_data = struct.unpack( str( CHUNK ) + 'h', data )

# #     line.set_ydata(int_data)
# #     fig.canvas.draw()
# #     fig.canvas.flush_events()
    


# # # import pyaudio
# # # import wave
# # # import sys


# # # audio = pyaudio.PyAudio()

# # # stream = audio.open(format = pyaudio.paInt16,
# # #                     channels = 1,
# # #                     rate = 44100,
# # #                     input = True,
# # #                     frames_per_buffer = 1024)

# # # frames = []

# # try:
# #     while True:
# #         data = stream.read(1024)
# #         frames.append(data)
# #         print(data)
# # except KeyboardInterrupt:
# #     print("Audio acquisition stopped.")
    
    
# # # stream.stop_stream()
# # # stream.close()
# # # audio.terminate()

# # # sound_file = wave.open("record.wav", "wb")
# # # sound_file.setnchannels(1)
# # # sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
# # # sound_file.setframerate(44100)
# # # sound_file.writeframes(b''.join(frames))
# # # sound_file.close()

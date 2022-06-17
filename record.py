import numpy as np
import pyaudio
import time
import librosa
import librosa.display
import wave
import matplotlib.pyplot as plt
import os

from prep import get_mfcc
from config import Configurator

class AudioRecord(object):
    def __init__(self, config):
        self.format = pyaudio.paInt16
        self.chunk = 1024
        self.channels = 2        
        self.rate = config.get('audio', 'rate')
        self.rec_duration = config.get('audio', 'record duration')
        self.prep_duration = config.get('audio', 'prep duration')
        self.mfcc_n = config.get('audio', 'mfcc coefficients')

        self.output = config.get('directories', 'Temporary files')
        self.rec_name = 'rec.wav'
        self.file_path = os.path.join(self.output, self.rec_name)


    def record(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                        channels=self.channels,
                        rate=self.rate,
                        input=True,
                        frames_per_buffer=self.chunk)
        
        print("*recording")
        frames = []
        frames_n = int(self.rate / self.chunk * self.rec_duration)
        for i in range(frames_n):
            frame = stream.read(self.chunk)
            frames.append(frame)
        print("*done*")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        self.save(p.get_sample_size(self.format), frames)


    def save(self, sample_size, frames):
        wf = wave.open(self.file_path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(sample_size)
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()


    def get_live_rec_data(self):
        y, _ = librosa.load(self.file_path, sr=self.rate)
        #Split an audio signal into non-silent intervals
        split = librosa.effects.split(y)
        samples_num = split.shape[0]
        ##########
        fig, ax = plt.subplots(nrows=(samples_num+1), sharex=True)
        librosa.display.waveshow(y, sr=self.rate, ax=ax[0])
        ##########
        for s in range(samples_num):
            s_start = split[s,0]
            s_stop = split[s,1]
            sample = y[s_start:s_stop]
            mfcc = get_mfcc(sample, self.rate, self.prep_duration, self.mfcc_n)
            ##########
            librosa.display.waveshow(sample, sr=self.rate, ax=ax[s+1])
            ##########
            #in fact it return mfcc of the first audio signal (s=0). It'll be changed in the future
            return mfcc



if __name__ == '__main__':
    c = Configurator()
    audio = AudioRecord(c)
    audio.record()
    g = audio.get_live_rec_data()

    

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

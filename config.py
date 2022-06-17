from configparser import ConfigParser
import os
import ast
import sys

class Configurator():
    def __init__(self):
        self.config_file = 'config.ini'
        self.config = ConfigParser()
        if not os.path.exists(self.config_file):
            self.config = self.write_config()
        else:
            self.config = self.read_config()


    def write_config(self):
        c = ConfigParser()
        c['directories'] = {# essential directories
                                  'Dataset': 'Dataset',
                                  'Preprocessed data': 'Prep_Dataset',
                                  'Saved models': 'Models',
                                  'Temporary files': 'Temp'}
        
        c['preprocessing'] = {# dataset splitting
                              'validation percentage': 15,
                              'testing percentage': 15,
                              'sample max': 2**27 - 1,
                              # data manipulation
                              'data normalization': False}
        
        c['audio'] = {# audio file processing
                      'rate': 22050,
                      'record duration': 2,
                      'prep duration': 1,
                      # audio processing
                      'mfcc coefficients': 13,
                      'sample shape': (13, 44, 1)}
        
        c['data']  = { #various data collections to train and use in project
                           'all': ['backward',  'bed',  'bird',  'cat',  'dog',  
                                   'down',  'eight',  'five',  'follow',  'forward',  
                                   'four',  'go',  'happy',  'house',  'learn',
                                   'left',  'marvin',  'nine',  'no',  'off',  
                                   'on',  'one',  'right',  'seven',  'sheila',  
                                   'six',  'stop',  'three',  'tree',  'two',  
                                   'up',  'visual',  'wow',  'yes',  'zero'],
                           
                            'first 2': ['backward',  'bed'],
                           
                            'first 10': ['backward',  'bed',  'bird',  'cat',  'dog',  
                                        'down',  'eight',  'five',  'follow',  'forward'],
            
                            'numbers': ['zero', 'one', 'two', 'three', 'four',
                                       'five', 'six', 'seven', 'eight', 'nine']}
        c['ai'] = {
                    'epochs': 40
                    }
        
        
        
        self.save_config(c)
        return c
        
    
    def save_config(self, config):
        with open(self.config_file, 'w+') as configfile:
            config.write(configfile)
            
    
    def read_config(self):
        c = ConfigParser()
        c.read(self.config_file)
        return c
    

    # def get(self, key):
    #     try:
    #         data = self.config[key]
    #         return data
            
    #     except KeyError:
    #         print("Error!")
    #         print("There is no '{}' section in the config file.".format(key))
    #         sys.exit()
        

    def get(self, key, subkey):
        try:
            data = self.config[key][subkey]
            try: #return string converted to the appropriate data type
                out = ast.literal_eval(data)  
                return out
            except ValueError: #in case that string is an appropriate data type, return it
                return data
            
        except KeyError:
            print("Error!")
            print("There is no '{}' section in the config file or '{}' in the '{}' section".format(key, subkey, key))
            sys.exit()
        
        
if __name__ == '__main__':
    c = Configurator()
    a = c.get('preprocessing', 'data normalization')
    print(a, type(a))
        
        
        
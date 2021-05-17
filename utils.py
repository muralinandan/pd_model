import enum
import json
import yaml
import os
from pydub.utils import mediainfo
from json.decoder import JSONDecodeError
import contextlib
import wave

class Status(enum.Enum):
    UNPROCESSED = 100
    VERIFIED_OK = 200
    VERIFIED_ADS = 201
    
    MARKED_FOR_DOWNLOAD = 400
    MARKED_FOR_DELETE = 401
    
    DOWNLOAD_PASS = 500
    DOWNLOAD_FAIL = 501
    
    CONVERSION_PASS = 600
    CONVERSION_FAIL = 601
    
    AUDIOSPLIT_PASS = 700
    AUDIOSPLIT_FAIL = 701

PUBLIC_ENUMS = {
    'Status': Status,
    # ...
}

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    config['base_path'] = os.path.dirname(os.path.abspath(__file__))
    return config

class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if type(obj) in PUBLIC_ENUMS.values():
            return {"__enum__": str(obj)}
        else:
            return json.JSONEncoder.default(self, obj)

def as_enum(obj):
    if "__enum__" in obj:
        name, member = obj["__enum__"].split(".")
        return getattr(PUBLIC_ENUMS[name], member)
    else:
        return obj
    
def open_files(file_path):
    if os.path.isfile(file_path):
        try:
            with open(file_path,'r') as fp:
                database = json.load(fp, object_hook=as_enum)
        except JSONDecodeError:
            database = None
    else:
        database = None
    return database

def garbage_collector(link_db):
    '''
        TODO: Create a garbage collector to remove the links that doesnt have downloadable content in the links_db.json
    '''
    pass
    
def get_video_info(video_filepath):
    """ this function returns number of channels, bit rate, and sample rate of the video"""
    video_data = mediainfo(video_filepath)
    channels = video_data["channels"]
    bit_rate = video_data["bit_rate"]
    sample_rate = video_data["sample_rate"]

    return channels, bit_rate, sample_rate

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate
    
def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)    

'''
TEST THE UTILITY FUNCTIONS
'''
if __name__ == '__main__':
    print ("Working Nice")

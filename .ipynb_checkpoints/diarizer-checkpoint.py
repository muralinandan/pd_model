import numpy as np
import sys, os, subprocess, re
import coloredlogs, logging, warnings
import yaml
import json


from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import enums
from google.cloud.speech_v1p1beta1 import types
from utils import Status, as_enum, EnumEncoder, load_config, get_video_info, garbage_collector, open_files

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger('DIARIZER')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Diarizer:
    def __init__(self,config,link_db):
        self.client = speech.SpeechClient()
    
    def run_speaker_diarization(self, audio_uri, audio_ch, audio_sr, max_speakers):
        logger.info('Performing Speaker Diarization for {}'.format(audio_uri))
        drzr_config = types.SpeakerDiarizationConfig(
                        enable_speaker_diarization=True,
                        max_speaker_count=max_speakers
                    )
        config = speech.types.RecognitionConfig(language_code = "en-US", sample_rate_hertz = int(audio_sr), encoding = enums.RecognitionConfig.AudioEncoding.MP3, audio_channel_count = int(audio_ch), enable_word_time_offsets = True, model = "video", enable_automatic_punctuation = False, diarization_config=drzr_config)
        
        audio_file = types.RecognitionAudio(uri=audio_uri)
        operation = self.client.long_running_recognize(config=config, audio=audio_file)
        res = operation.result()
        return res

'''
    Function to extract the start-end-speaker-word from the response of the API of form-
    .
    .
    start_time {
      nanos: 600000000
    }
    end_time {
      nanos: 900000000
    }
    word: "there\'s"
    speaker_tag: 3
    ,
    .
'''
def extract_speaker_offset(allwords):
    allwords_inline = []
    for a in allwords:
        word = a.word
        start_seco = a.start_time.seconds
        start_nano = a.start_time.nanos
        start = start_seco*1e9+start_nano

        end_seco = a.end_time.seconds
        end_nano = a.end_time.nanos
        end = end_seco*1e9+end_nano

        speaker = a.speaker_tag
        allwords_inline.append([start,end,speaker,word])
    return allwords_inline

'''
    Function to perform the grouping of the start-end-speaker-word objects in the following form-
    .
    .
    [3100000000.0, 3500000000.0, 3, 'question'],
    [3500000000.0, 3900000000.0, 3, 'everything'],
    [4100000000.0, 4300000000.0, 1, 'opening'],
    [4300000000.0, 4400000000.0, 1, 'up'],
    .
    .
'''
def perform_speaker_grouping(allwords):
    prev_speaker = -1
    final_speakers = []
    start = -1
    end = -1
    sentence = ""
    for i, value in enumerate(allwords):
        if i==0:
            start = value[0]
            end = value[1]
            prev_speaker = value[2]
            sentence += value[3]+' '
        else:
            if allwords[i][2] == allwords[i-1][2]:
                end = value[1]
                sentence += value[3]+' '
            else:
                final_speakers.append([start,end,prev_speaker,sentence])
                start = value[0]
                end = value[1]
                prev_speaker = value[2]
                sentence = value[3] + ' '
    if prev_speaker != final_speakers[-1][2]:
        final_speakers.append([start,end,prev_speaker,sentence])
    return final_speakers

def save_speaker_info(speakers,config):
    tagged_speakers_path = os.path.join(config['base_path'],config['tagged_speakers_path'])
    with open(tagged_speakers_path,'w+') as fp:
        fp.write(json.dumps(speakers, cls = EnumEncoder))

def make_link_db(updated_dict,config):
    logger.info('Updating the video database to {}'.format(config['link_db_path']))
    
    Path(config['link_db_path'].split('/')[0]).mkdir(parents=True,exist_ok=True)
    with open(os.path.join(config['base_path'],config['link_db_path']),'w+') as fp:
        fp.write(json.dumps(updated_dict, cls = EnumEncoder))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config_file', type=str, help='Configuration file', required=True)
    args = parser.parse_args()
    config = load_config(args.config_file)

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config['credentials']
    logger.info('Using configuration: {}'.format(config))
    with open(os.path.join(config['base_path'],config['link_db_path'])) as fp:
        link_db = json.load(fp, object_hook=as_enum)
    with open(os.path.join(config['base_path'],config['max_speakers_path'])) as fp:
        speaker_db = json.load(fp, object_hook=as_enum)
        
    drzr = Diarizer(config,link_db)
    tagged_speakers_db_path = os.path.join(config['base_path'],config['tagged_speakers_path'])
    tagged_speaker_db = open_files(tagged_speakers_db_path)
    for person,data in link_db.items():
        for index, video in tqdm(enumerate(data)):
            curr_data = {}
            if video['status'] == Status.UPLOAD_PASS:
                video_ch = video['channels']
                video_br = video['bit_rate']
                video_sr = video['sample_rate']
                video_uri= video['gcs_uri']
                video_title= re.sub('[^A-Za-z0-9]+', '_', video['title']) + ".mp3"
                try:
                    if video_title in speaker_db.keys():
                        max_speakers = speaker_db[video_title]
                    else:
                        max_speakers = 2
                    print ()
                    response = drzr.run_speaker_diarization(video_uri, video_ch, video_sr, max_speakers)
                    allwords = response.results[-1].alternatives[0].words
                    allwords = extract_speaker_offset(allwords)
                    speakers = perform_speaker_grouping(allwords)
                    video['status'] = Status.DIARIZATION_PASS
                    
                    curr_data['person'] = person
                    curr_data['title'] = video_title
                    curr_data['path'] = video['mp3_path']
                    curr_data['gcs_uri'] = video_uri
                    curr_data['url'] = video['url']
                    curr_data['speakers'] = speakers
                    curr_data['timestamp'] = video['date']
                    curr_data['channels'] = video_ch
                    curr_data['bit_rate'] = video_br
                    curr_data['sample_rate'] = video_sr
                except Exception as e:
                    logger.critical('Speaker Diarization Failed.')
                    logger.critical(e, exc_info=True)
                    video['status'] = Status.DIARIZATION_FAIL
                time.sleep(3)
            tagged_speaker_db.update({video_title:curr_data})
        data[index] = video
    link_db[person] = data
    
    make_link_db(link_db,config)
    save_speaker_info(tagged_speaker_db,config)
import coloredlogs, logging, warnings
import sys, re, os, subprocess, time
import yaml
import json

from pathlib import Path
from pydub import AudioSegment
from pydub.utils import make_chunks
from pprint import pprint

from argparse import ArgumentParser
from utils import Status, as_enum, EnumEncoder, load_config, get_video_info, garbage_collector

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger('AUDIO_SPLITS')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def make_link_db(updated_dict,config):
    logger.info('Updating the video database to {}'.format(config['link_db_path']))
    
    Path(config['link_db_path'].split('/')[0]).mkdir(parents=True,exist_ok=True)
    with open(os.path.join(config['base_path'],config['link_db_path']),'w+') as fp:
        fp.write(json.dumps(updated_dict, cls = EnumEncoder))

def cut_join(audio, times, video_title):
    
    extract = None
    chunk_length_ms = 10000
    audio_outs = []
    if len(times) == 0:
        return audio_outs
    for chunk in times:
        time = chunk.split('-')
        startMin,startSec = [int(x) for x in time[0].split(':')]
        endMin, endSec = [int(x) for x in time[1].split(':')]
        
        startTime = startMin*60*1000+startSec*1000
        endTime = endMin*60*1000+endSec*1000
        extract = audio[startTime:endTime] if extract==None else extract+audio[startTime:endTime]
    chunks = make_chunks(extract, chunk_length_ms)
    for i, chunk in enumerate(chunks):
        chunk_name = video_title+"!{0}.wav".format(i)
        audio_outs.append({'filename':chunk_name,'audio':chunk})
        
    return audio_outs
         
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config_file', type=str, help='Configuration file', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config_file)
    logger.info('Using configuration: {}'.format(config))
    
    with open(os.path.join(config['base_path'],config['link_db_path'])) as fp:
        link_db = json.load(fp, object_hook=as_enum)
        
    for person,data in link_db.items():
        for index, video in enumerate(data):
            video_title = re.sub('[^A-Za-z0-9]+', '_', video['title'])
            video_link = video['url']
            
            if video['status'] == Status.CONVERSION_PASS:
                audio = AudioSegment.from_wav(video['wav_path'])
                gold_times = video['timings']
                try:
                    print ("Starting FOR {}".format(video_title))
                    audios_outs = cut_join(audio,gold_times,video_title)
                    for audio in audios_outs:
                        chunk = audio['audio']
                        name = audio['filename']
                        chunk.export(os.path.join(config['gold_dir'],name), format="wav")
                    print ("DONE FOR {}".format(video_title))
                    video['status'] = Status.AUDIOSPLIT_PASS
                except Exception as e:
                    video['status'] = Status.AUDIOSPLIT_FAIL                
                    
        data[index] = video
    link_db[person] = data
    
    make_link_db(link_db,config)
                
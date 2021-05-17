import coloredlogs, logging, warnings
import sys, re, os, subprocess,time
import yaml
import json

# from google.cloud import storage
# from google.cloud import speech_v1 as speech
# from google.cloud.speech_v1 import enums
# from google.cloud.speech_v1 import types
from pathlib import Path
from pytube import YouTube
from argparse import ArgumentParser
from utils import Status, as_enum, EnumEncoder, load_config, get_video_info, garbage_collector

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger('DOWNLOADER')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def make_link_db(updated_dict,config):
    logger.info('Updating the video database to {}'.format(config['link_db_path']))
    
    Path(config['link_db_path'].split('/')[0]).mkdir(parents=True,exist_ok=True)
    with open(os.path.join(config['base_path'],config['link_db_path']),'w+') as fp:
        fp.write(json.dumps(updated_dict, cls = EnumEncoder))
    
class Downloader:
    def __init__(self,config,link_db):
        self.link_db_path = os.path.join(config['base_path'],config['link_db_path'])
        self.mp4_path = os.path.join(config['base_path'],config['mp4_dir'])
        self.wav_path = os.path.join(config['base_path'],config['wav_dir'])
        self.config = config
        self.link_db = link_db
    
    def download_links(self,yt,video):
        video_title = re.sub('[^A-Za-z0-9]+', '_', video['title'])
        video_link = video['url']
        video_upload_time = video['date']
        try:
            best_stream = yt.streams.get_audio_only()
        except Exception as e:
            print ("Issue with: {}".format(video_link))
            return video
        if best_stream != None:
            video['status'] = Status.MARKED_FOR_DOWNLOAD
            try:
                best_stream.download(self.mp4_path, filename=video_title, skip_existing=True)
                channels, bit_rate, sample_rate = get_video_info(os.path.join(self.mp4_path,video_title+".mp4"))
                video['status'] = Status.DOWNLOAD_PASS
                video['channels'] = channels
                video['bit_rate'] = bit_rate
                video['sample_rate'] = sample_rate
                logger.info('Download success for {} with Status- {}'.format(video_link,video['status']))
            except Exception as e:
                video['status'] = Status.DOWNLOAD_FAIL
                logging.critical(e, exc_info=True)
                logging.critical('Download failed for {} with Status- {}'.format(video_link,video['status']))
        else:
            video['status'] = Status.MARKED_FOR_DELETE
            logger.warning('Nothing Available to download in- {}. Thus, marking for delete with Status-{}'.format(video_link,video['status']))
        return video

def upload_file(video,bucket_name, source_file_path, blob_name):
    blob_name = blob_name+'/'+source_file_path.split('/')[-1];
    try:
        storage_cl = storage.Client()
        bucket = storage_cl.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(source_file_path)
        video['status'] = Status.UPLOAD_PASS
        video['gcs_uri'] = "gs://"+bucket_name+"/"+blob_name
    except Exception as e:
        video['status'] = Status.UPLOAD_FAIL
        logger.critical(e, exc_info=True)
    return video,blob_name
    
def video_convertor(video,wav_path,mp4_path):
    logger.info('MP4 to wav Conversion has begun')
    try:
        video_title = re.sub('[^A-Za-z0-9]+', '_', video['title'])
        wav_filename = video_title+".wav"
        mp4_filename = video_title+".mp4"
        video_channels = video['channels']
        video_bit_rate = video['bit_rate']
        video_sample_rate = video['sample_rate']
        subprocess.run(['ffmpeg','-i',os.path.join(mp4_path,mp4_filename),'-b:a',video_bit_rate,'-ac',video_channels,'-ar',video_sample_rate,os.path.join(wav_path,wav_filename),'-hide_banner','-nostats','-loglevel','0','-y'])
        video['status'] = Status.CONVERSION_PASS
        video['wav_path'] = os.path.join(wav_path,wav_filename)
    except Exception as e:
        video['status'] = Status.CONVERSION_FAIL
        logger.critical(e, exc_info=True)
    return video,os.path.join(wav_path,wav_filename)
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config_file', type=str, help='Configuration file', required=True)
    parser.add_argument('--testing', action='store_true', help='use for testing the downloader')
    args = parser.parse_args()
    
    config = load_config(args.config_file)
    config['testing'] = args.testing
    logger.info('Using configuration: {}'.format(config))
    
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = config['credentials']
    Path(config['mp4_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['wav_dir']).mkdir(parents=True, exist_ok=True)

    with open(os.path.join(config['base_path'],config['link_db_path'])) as fp:
        link_db = json.load(fp, object_hook=as_enum)
    dnwlr = Downloader(config,link_db)
        
    for person,data in link_db.items():
        for index, video in enumerate(data):
            video_title = re.sub('[^A-Za-z0-9]+', '_', video['title'])
            video_link = video['url']
            
            if video['status'] == Status.VERIFIED_OK:
                video_upload_time = video['date']
                video_gold_timings = video['timings']
                try:
                    yt = YouTube(video_link)
                    logger.info('Processing - {}, link - {}, Status - {}'.format(person,video_link,video['status']))
                except Exception as e:
                    video['status'] = Status.VERIFIED_ADS

                if video['status'] != Status.VERIFIED_ADS:
                    video = dnwlr.download_links(yt,video)
                    if video['status'] == Status.DOWNLOAD_PASS:
                        video,file_path = video_convertor(video,dnwlr.wav_path,dnwlr.mp4_path)
                        if video['status'] == Status.CONVERSION_PASS:
                            logger.info('MP4 to wav Conversion has passed with Status-{}'.format(video['status']))
                        else:
                            logger.critical('MP4 to wav Conversion has failed with Status-{}\n'.format(video['status']))
                else:
                    logger.warning('link consists of ADS. Thus skipping with Status - {}'.format(video['status']))
                print ()
            elif video['status'] == Status.CONVERSION_PASS or video['status'] == Status.AUDIOSPLIT_PASS:
                pass
            else:
                video['status'] = Status.UNPROCESSED
                
                
        data[index] = video
    link_db[person] = data

    make_link_db(link_db,config)

    
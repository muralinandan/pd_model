import coloredlogs, logging, warnings
import sys, re, os, subprocess, time
import yaml
import json
import numpy as np
import pandas as pd
import datetime

from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment
from pydub.utils import make_chunks
from pprint import pprint
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, plot_confusion_matrix

import surfboard
from surfboard.sound import Waveform
from surfboard.feature_extraction_multiprocessing import extract_features_from_paths as extractFeaturesMP
from surfboard.feature_extraction import extract_features

import parselmouth
from parselmouth.praat import call

from argparse import ArgumentParser
from utils import Status, as_enum, EnumEncoder, load_config, get_video_info, garbage_collector, getComponentsList, getStatusForDP

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger('FEATURE_EXTRACTION')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def make_link_db(updated_dict,config):
    logger.info('Updating the video database to {}'.format(config['link_db_path']))
    
    Path(config['link_db_path'].split('/')[0]).mkdir(parents=True,exist_ok=True)
    with open(os.path.join(config['base_path'],config['link_db_path']),'w+') as fp:
        fp.write(json.dumps(updated_dict, cls = EnumEncoder))

def preprocessFiles(path):
    outputFiles = {}
    allFiles = [f for f in os.listdir(config['gold_dir']) if os.path.isfile(os.path.join(config['gold_dir'], f))]
    for file in allFiles:
        titleName = file.split('!')[0]
        if titleName not in outputFiles.keys():
            outputFiles[titleName] = [os.path.join(config['gold_dir'], file)]
        else:
            outputFiles[titleName].append(os.path.join(config['gold_dir'], file))
    return outputFiles

def extractPitch(id,f0Min,f0Max,unit):
    sound = parselmouth.Sound(id)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0Min, f0Max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    jitters = {'localJitter':localJitter,'localabsoluteJitter':localabsoluteJitter,'rapJitter':rapJitter,'ppq5Jitter':ppq5Jitter,'ddpJitter':ddpJitter}
    shimmers = {'localShimmer':localShimmer,'localdbShimmer':localdbShimmer,'apq3Shimmer':apq3Shimmer,'apq5Shimmer':apq5Shimmer,'apq11Shimmer':apq11Shimmer,'ddaShimmer':ddaShimmer}
    return jitters , shimmers
    
def extractFeatures(paths,data,index,title,currdate,dob_year,diag_year):
    if len(paths)==0:
        return data
    records = []
    print ("Extracting features for {}".format(title))
    curr_year = int(currdate.split('-')[0])
    time = curr_year*52 + int(datetime.datetime.strftime(datetime.datetime.strptime(currdate, "%Y-%m-%d"),'%U'))
    person_age = curr_year - dob_year
    status = getStatusForDP(curr_year, diag_year)
    for path in tqdm(paths,total=len(paths)):
        wave = Waveform(path, sample_rate=44100)
        sound = parselmouth.Sound(path)
        jitters, shimmers = extractPitch(sound, 75, 500, "Hertz")
        ppe = wave.ppe()
        hnr = wave.hnr()
        dfa = wave.dfa()
        nhr = 1/(hnr+sys.float_info.epsilon)
        records.append([index,jitters['localJitter'],jitters['localabsoluteJitter'],jitters['rapJitter'],jitters['ppq5Jitter'],jitters['ddpJitter'],shimmers['localShimmer'],shimmers['localdbShimmer'],shimmers['apq3Shimmer'],shimmers['apq5Shimmer'],shimmers['apq11Shimmer'],shimmers['ddaShimmer'],nhr,hnr,dfa,ppe,status,person_age,time])
    data.extend(records)
    
    return data

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config_file', type=str, help='Configuration file', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config_file)
    logger.info('Using configuration: {}'.format(config))
    
    with open(os.path.join(config['base_path'],config['link_db_path'])) as fp:
        link_db = json.load(fp, object_hook=as_enum)
    with open(os.path.join(config['base_path'],config['dob_db_path'])) as fp:
        dob_db = json.load(fp)
    with open(os.path.join(config['base_path'],config['diagnosys_db_path'])) as fp:
        diag_db = json.load(fp)
   
    all_wav_files = preprocessFiles(config['gold_dir'])
    features = ['Subject#','Jitter(%)','Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP','Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA','NHR','HNR','DFA','PPE','status','age','time']
    df = pd.DataFrame(columns = features)
    df.to_csv(os.path.join(os.getcwd(),config['extracted_features_path']), index=False)
    
    for person,video_data in link_db.items():    
        data = []
        old_df = pd.read_csv(os.path.join(os.getcwd(),config['extracted_features_path']))
        for index, video in enumerate(video_data):
            paths = []
            video_title = re.sub('[^A-Za-z0-9]+', '_', video['title'])
            if video['status'] == Status.AUDIOSPLIT_PASS:
                allFiles = all_wav_files[video_title]
                paths.extend(allFiles)
            data = extractFeatures(paths,data,index,video_title,video['date'],dob_db[person],diag_db[person])
        new_df = pd.DataFrame(data,columns = features)
        old_df = old_df.append(new_df, ignore_index = True)
        old_df.to_csv(os.path.join(os.getcwd(),config['extracted_features_path']), index=False)
                    
    
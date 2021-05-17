import os, sys, re
import json
import coloredlogs, logging, warnings

from utils import load_config, Status, as_enum, garbage_collector
from argparse import ArgumentParser

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger('PREREQUISITES')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class Sanity:
    def __init__(self,link_db):
        self.max_speakers = {
            'FRONTLINE_Parkinson_s_Michael_J_Fox_Interview_PBS.mp3'                                         : 5,
            'Michael_J_Fox_Addresses_Darkest_Moments_Since_Parkinson_s_Diagnosis.mp3'                       : 2,
            'Michael_J_Fox_on_his_love_of_Canada_1987_CBC_Archives_CBC.mp3'                                 : 2,
            'Michael_J_Fox_Interview_Casualties_Of_War_1989_Reelin_In_The_Years_Archives_.mp3'              : 3,
            'Michael_J_Fox_Opens_Up_About_Health_and_Book_No_Time_Like_the_Future_The_View.mp3'             : 4,
            'TIME_Magazine_Interviews_Michael_J_Fox.mp3'                                                    : 2,
            'Michael_J_Fox_Opens_Up_About_His_Darkest_Moment_.mp3'                                          : 5,
            'Michael_J_Fox_says_his_toughest_year_tested_his_optimism_l_GMA.mp3'                            : 5,
            # ...
        }
        self.gold_speakers = {
            'FRONTLINE_Parkinson_s_Michael_J_Fox_Interview_PBS.mp3'                                         : [-1],
            'Michael_J_Fox_Addresses_Darkest_Moments_Since_Parkinson_s_Diagnosis.mp3'                       : [-1],
            'Michael_J_Fox_on_his_love_of_Canada_1987_CBC_Archives_CBC.mp3'                                 : [-1],
            'Michael_J_Fox_Interview_Casualties_Of_War_1989_Reelin_In_The_Years_Archives_.mp3'              : [-1],
            'Michael_J_Fox_Opens_Up_About_Health_and_Book_No_Time_Like_the_Future_The_View.mp3'             : [-1],
            'TIME_Magazine_Interviews_Michael_J_Fox.mp3'                                                    : [-1],
            'Michael_J_Fox_Opens_Up_About_His_Darkest_Moment_.mp3'                                          : [-1],
            'Michael_J_Fox_says_his_toughest_year_tested_his_optimism_l_GMA.mp3'                            : [-1],
            # ...
        }
        
        self.link_db = link_db
        
    def set_sanity(self,flag):
        logger.info('************Performing Sanity Check for Minimum Speakers************')
        response = self.print_files(flag)
        if response == True:
            if flag == "b_diarize":
                logger.info('Assigning the minimum speakers manually')
                with open(os.path.join(config['base_path'],config['max_speakers_path']),'w+') as fp:
                    json.dump(self.max_speakers,fp)
            elif flag == "b_consolidate":
                logger.info('Assigning the gold speakers manually')
                with open(os.path.join(config['base_path'],config['gold_speakers_path']),'w+') as fp:
                    json.dump(self.gold_speakers,fp)
        else:
            if flag == "b_diarize":
                logger.error('Provide the minimum speakers before saving the speakers to file')
            elif flag == "b_consolidate":
                logger.error('Provide the gold speakers before saving the speakers to file')
            sys.exit(100)
                 
    def get_sanity(self):
        return self.max_speakers,self.gold_speakers

    def print_files(self,flag):
        all_sane = True
        for person,data in link_db.items():
            for index, video in enumerate(data):
                if flag == "b_diarize":
                    if video['status'] == Status.UPLOAD_PASS:
                        video_title = re.sub('[^A-Za-z0-9]+', '_', video['title'])
                        video_title = video_title+'.mp3'
                        if video_title not in self.max_speakers.keys():
                            all_sane = False
                            logger.error('{} missing the minimum speakers'.format(video_title))
                elif flag == "b_consolidate":
                    if video['status'] == Status.DIARIZATION_PASS:
                        video_title = re.sub('[^A-Za-z0-9]+', '_', video['title'])
                        video_title = video_title+'.mp3'
                        if video_title not in self.gold_speakers.keys():
                            all_sane = False
                            logger.error('{} missing the gold speakers'.format(video_title))
        return all_sane
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config_file', type=str, help='Configuration file', required=True)
    parser.add_argument('--b_download', action='store_true', help='use for testing the downloader')
    parser.add_argument('--b_diarize', action='store_true', help='use for testing the downloader')
    parser.add_argument('--b_consolidate', action='store_true', help='use for testing the downloader')
    args = parser.parse_args()
    config = load_config(args.config_file)

    with open(os.path.join(config['base_path'],config['link_db_path'])) as fp:
        link_db = json.load(fp, object_hook=as_enum)
    snty = Sanity(link_db)
    
    if args.b_download == True:
        garbage_collector(link_db)
    elif args.b_diarize == True:
        snty.set_sanity("b_diarize")
    elif args.b_consolidate == True:
        snty.set_sanity("b_consolidate")
    
    
    
    
    
    
    
    


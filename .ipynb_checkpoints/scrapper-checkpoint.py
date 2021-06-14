import requests, urllib
import os,sys,time, re 
import coloredlogs, logging, warnings
import json, isodate

from bs4 import BeautifulSoup
from googlesearch import search
from googleapiclient.discovery import build
from argparse import ArgumentParser
from pathlib import Path


from utils import Status,EnumEncoder, load_config, as_enum, open_files

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger('GOOGLESCRAPPER')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def google_search(search_term, api_key, cse_id, **kwargs):    
    service = build("customsearch", "v1", developerKey=api_key)
    results = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return results['items']

def make_link_db(response_dict,config):
    logger.info('Saving Videos to {}'.format(config['link_db_path']))
    
    Path(os.path.join(config['base_path'],config['link_db_path'])).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(os.path.join(config['base_path'],config['link_db_path']),config['dest_file']),'w+') as fp:
        fp.write(json.dumps(response_dict, cls = EnumEncoder))
        
def make_custom_date(date):
    end_year = int(date)-1
    start_year = int(end_year)-10
    csdate = "date:r:"+str(start_year)+"0101"+":"+str(end_year)+"0101"
    logger.info("Scraping videos from {} to {}".format(start_year,end_year))
    return csdate
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config_file', type=str, help='Configuration file', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config_file)
    logger.info('Using configuration: {}'.format(config))
    
    google_api_key = config['google_api_key']
    google_cse_id = config['google_cse_id']
    person_list = [person for person in config['person_names']]
    response_dict = {}
    link_db_path = os.path.join(os.path.join(config['base_path'],config['link_db_path']),config['dest_file'])
    link_db = open_files(link_db_path)
                                
    for person in person_list:
        person_name = person.split('-')[0]
        person_date = person.split('-')[1]
        response_dict[person_name] = []
        
        query = person_name + ' interviews'
        results = [] if (link_db == None or person_name not in link_db) else link_db[person_name]
        available_urls = [r['url'] for r in results]
        csdate = make_custom_date(person_date)
        num = config['num_sr']
        
        
        for start in range(1, 31, 10):
            logger.info('Retrieving results for query: {}'.format(query))
            raw_results = google_search(query,google_api_key,google_cse_id,num=num,siteSearch='www.youtube.com',siteSearchFilter="i",start=start,sort=csdate)
            for result in raw_results:
                if 'videoobject' in result['pagemap']:
                    result = result['pagemap']['videoobject'][0]
                    video_duration = isodate.parse_duration(result['duration'])
                    video_duration_seconds = video_duration.total_seconds()
                                                            
                    if result['url'] not in available_urls and video_duration_seconds<=700:
                        results.extend([{
                                            'date':result['uploaddate'],
                                            'url':result['url'],
                                            'title':result['name'],
                                            'duration': video_duration_seconds,
                                            "status": Status.UNPROCESSED
                                        }])
                        available_urls.extend(result['url'])
            time.sleep(3)
        response_dict[person_name].extend(results)
            
    make_link_db(response_dict,config)
    logger.info('Scraping Task is Completed.')
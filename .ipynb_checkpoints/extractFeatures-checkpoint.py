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
logger = logging.getLogger('FEATURE_EXTRACTION')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def make_link_db(updated_dict,config):
    logger.info('Updating the video database to {}'.format(config['link_db_path']))
    
    Path(config['link_db_path'].split('/')[0]).mkdir(parents=True,exist_ok=True)
    with open(os.path.join(config['base_path'],config['link_db_path']),'w+') as fp:
        fp.write(json.dumps(updated_dict, cls = EnumEncoder))

def window_model(n_bands, n_frames, n_classes, hidden=32):
    from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D

    out_units = 1 if n_classes == 2 else n_classes
    out_activation = 'sigmoid' if n_classes == 2 else 'softmax'

    shape = (n_bands, n_frames, 1)

    # Basic CNN model
    # An MLP could also be used, but may need to reshape on input and output
    model = keras.Sequential([
       Conv2D(16, (3,3), input_shape=shape),
       MaxPooling2D((2,3)),
       Conv2D(16, (3,3)),
       MaxPooling2D((2,2)),
       Flatten(),
       Dense(hidden, activation='relu'),
       Dense(hidden, activation='relu'),
       Dense(out_units, activation=out_activation),
    ])
    return model

def song_model(n_bands, n_frames, n_windows, n_classes=3):
    from keras.layers import Input, TimeDistributed, GlobalAveragePooling1D

    # Create the frame-wise model, will be reused across all frames
    base = window_model(n_bands, n_frames, n_classes)
    # GlobalAveragePooling1D expects a 'channel' dimension at end
    shape = (n_windows, n_bands, n_frames, 1)

    print('Frame model')
    base.summary()

    model = keras.Sequential([
        TimeDistributed(base, input_shape=shape),
        GlobalAveragePooling1D(),
    ])

    print('Song model')
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['acc'])
    return model

def main():

    
    
    
    
    # Settings for our model
    n_bands = 13 # MFCCs
    sample_rate = 22050
    hop_length = 512
    window_length = 5.0
    song_length_max = 1.0*60
    n_frames = math.ceil(window_length / (hop_length/sample_rate))
    n_windows = math.floor(song_length_max / (window_length/2))-1

#     model = song_model(n_bands, n_frames, n_windows)

#     Generate some example data
    ex =  librosa.util.example_audio_file()
    print (type(ex))
    examples = 8
    numpy.random.seed(2)
    songs = pandas.DataFrame({
        'path': [ex] * examples,
        'genre': numpy.random.choice([ 'rock', 'metal', 'blues' ], size=examples),
    })
    print (songs.path)
#     assert len(songs.genre.unique() == 3) 

#     print('Song data')
#     print(songs)

#     def get_features(path):
#         f = extract_features(path, sample_rate, n_bands,
#                     hop_length, n_frames, window_length, song_length_max)
#         return f

#     from sklearn.preprocessing import LabelBinarizer

#     binarizer = LabelBinarizer()
#     y = binarizer.fit_transform(songs.genre.values)
#     print('y', y.shape, y)

#     features = numpy.stack([ get_features(p) for p in songs.path ])
#     print('features', features.shape)

#     model.fit(features, y) 

def preprocessFiles(path):
    outputFiles = {}
    allFiles = [f for f in os.listdir(config['gold_dir']) if os.path.isfile(os.path.join(config['gold_dir'], f))]
    for file in allFiles:
        titleName = file.split('!')[0]
        if titleName not in outputFiles.keys():
            outputFiles[titleName] = [file]
        else:
            outputFiles[titleName].append(file)
    return outputFiles
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config_file', type=str, help='Configuration file', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config_file)
    logger.info('Using configuration: {}'.format(config))
    
    with open(os.path.join(config['base_path'],config['link_db_path'])) as fp:
        link_db = json.load(fp, object_hook=as_enum)
        
    all_wav_files = preprocessFiles(config['gold_dir'])
    for person,data in link_db.items():
        for index, video in enumerate(data):           
            if video['status'] == Status.AUDIOSPLIT_PASS:
                video_title = re.sub('[^A-Za-z0-9]+', '_', video['title'])
                allFiles = all_wav_files[video_title]
                for file in allFiles:
                    
                    
    
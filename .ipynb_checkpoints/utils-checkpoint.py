import enum, json, yaml
import os, itertools
import contextlib, operator
import wave
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pydub.utils import mediainfo
from json.decoder import JSONDecodeError
from scipy import stats
from sklearn.metrics import confusion_matrix


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
    'Status': Status
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

def getComponentsList():
    return ['mfcc', 'spectral_flux', 'spectral_slope', 'spectral_centroid', 'spectral_spread', 'spectral_skewness',
        'spectral_kurtosis', 'spectral_rolloff', 'shannon_entropy_slidingwindow', 'rms']

def getStatusForDP(curr_year, diag_year):
    if diag_year == '0000':
        status = 0
    elif diag_year > curr_year:
        status = 2
    else:
        status = 1
    return status

def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=operator.itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(score.mean_validation_score,np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print()
        
def cross_val_RF(model, X, y):
    rs = RandomizedSearchCV(model, param_distributions={
        'n_estimators': stats.randint(30, 200),
        'max_features': ['auto', 'sqrt', 'log2'],
        "max_depth": [3, None],
        "max_features": stats.randint(1, 11),
        "min_samples_split": stats.randint(1, 11),
        "min_samples_leaf": stats.randint(1, 11),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
        })

    rs.fit(X, y)
    report(rs.grid_scores_)

def cross_val_NN(model, X, y):
    rs = RandomizedSearchCV(model, scoring='f1', param_distributions={
        'learning_rate': stats.uniform(0.001, 0.05),
        'hidden0__units': stats.randint(10, 200),
        'hidden1__units': stats.randint(20, 200),
        'hidden2__units': stats.randint(4, 100),
        'hidden3__units': stats.randint(4, 50),
        'hidden4__units': stats.randint(4, 50),
        'hidden0__type': ["Rectifier", "Sigmoid", "Tanh"],
        'hidden1__type': ["Rectifier", "Sigmoid", "Tanh"],
        'hidden2__type': ["Rectifier", "Sigmoid", "Tanh"],
        'hidden3__type': ["Rectifier", "Sigmoid", "Tanh"],
        'hidden4__type': ["Rectifier", "Sigmoid", "Tanh"]
        })

    rs.fit(X, y)
    report(rs.grid_scores_)

def cross_val_RBM(model, X, y):
    rs = RandomizedSearchCV(model, param_distributions={
            'n_components': stats.randint(1, 256),
            'learning_rate': stats.uniform(0.001, 0.05),
            'batch_size': stats.randint(1, 10),}, scoring='')
    
    rs.fit(X, y)
    report(rs.grid_scores_)
    
def plot_correlation(df,config,size=20):
    '''
    Below is the work to print out a correlation matrix to measure correlation between the principal features and the target vector. 
    Generally, if there is low to medium correlation between multiple vectors and the target, that's a sign that the vectors may be predictive of the target.
    '''
    df_correlation = df.copy()
    df_correlation.pop('subject#')
    df_correlation = df_correlation.drop_duplicates().dropna()
    
    corr = df_correlation.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(size, size))
        ax = sns.heatmap(corr, annot=True, fmt=".2f", mask=mask, linewidths=.1, linecolor='white',cbar=True, cbar_kws={"shrink": .5})
        ax.set_xticklabels(ax.get_xticklabels(), rotation=70, ha="center")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=40, ha="right")
        plt.tight_layout()
        f.savefig(os.path.join(config['base_path'],os.path.join(config['viz_dir'],'correlation_matrix.png')),bbox_inches='tight')
        plt.close()
        
def plot_feature_impact_chart(df,target,config):
    df_correlation = df.copy()
    df_correlation.pop('subject#')
    df_correlation = df_correlation.drop_duplicates().dropna()
    
    df_new = df_correlation.corr().sort_values([target], ascending = False)
    df_new.drop(df_new.head(1).index, inplace=True)
    df_new.drop(df_new.tail(1).index, inplace=True)
    df_new['Index'] = df_new.index
    chart_title = ("Feature Impact Chart - Correlation with Target (%s))" % target)
    with sns.axes_style("darkgrid"):
        fg = sns.catplot(x="Index", y=target, kind="bar", palette=sns.color_palette("rocket"), data=df_new)
        plt.xticks(rotation=70, horizontalalignment='center', fontweight='light', fontsize='large')
        plt.tight_layout()
        f = plt.gcf()
        f.set_size_inches(17, 10, forward=True)
        f.savefig(os.path.join(config['base_path'],os.path.join(config['viz_dir'],'feature_impact.png')), bbox_inches='tight')
        
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    tick_marks = np.arange(len(classes))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, loc='left')    
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.colorbar()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.round(cm, decimals=2)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.clim(0,1) # Reset the colorbar to reflect probabilities

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    
    plt.grid(False)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def call_confusion_matrix(y_test, y_pred, target_array, filedir, size=5):
    class_names = target_array
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print (cnf_matrix)
    np.set_printoptions(precision=2)
    filename  = 'confusion_matrix.png'
    cf_no_norm_file = ("1_%s" % filename)
    cf_norm_file = ("2_%s" % filename)

    # Plot non-normalized confusion matrix
    plt.figure(figsize=(size, size), dpi=200)
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    plt.savefig(os.path.join(filedir,cf_no_norm_file), bbox_inches='tight')

    # Plot normalized confusion matrix
    plt.figure(figsize=(size, size), dpi=200)
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix, with normalization')
    plt.savefig(os.path.join(filedir,cf_norm_file), bbox_inches='tight')

def heldout_score(clf, X_test, y_test):
    """ Evaluate deviance scores on X_test and y_test."""
    score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        score[i] = clf.loss_(y_test, y_pred)
    return score

def cv_estimate(n_splits=3):
    cv = KFold(n_splits=n_splits)
    cv_clf = ensemble.GradientBoostingClassifier(**params)
    val_scores = np.zeros((n_estimators,), dtype=np.float64)
    for train, test in cv.split(X_train, y_train):
        cv_clf.fit(X_train[train], y_train[train])
        val_scores += heldout_score(cv_clf, X_train[test], y_train[test])
    val_scores /= n_splits
    return val_scores
    
'''
TEST THE UTILITY FUNCTIONS
'''
if __name__ == '__main__':
    print ("Working Nice")

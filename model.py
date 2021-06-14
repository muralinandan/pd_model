import coloredlogs, logging, warnings
import sys, re, os, subprocess, time
import yaml
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math, matplotlib
import seaborn as sns

from sklearn import preprocessing, metrics, linear_model
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, cross_val_score, ShuffleSplit, validation_curve, train_test_split
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, hinge_loss
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM, MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from pprint import pprint
from pathlib import Path
from tqdm import tqdm

from argparse import ArgumentParser
from utils import *
from datasetLoader import ParkinsonData

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger('SUPERVISED_MODEL')
coloredlogs.install(logger=logger, level='DEBUG', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def make_link_db(updated_dict,config):
    logger.info('Updating the video database to {}'.format(config['link_db_path']))
    
    Path(config['link_db_path'].split('/')[0]).mkdir(parents=True,exist_ok=True)
    with open(os.path.join(config['base_path'],config['link_db_path']),'w+') as fp:
        fp.write(json.dumps(updated_dict, cls = EnumEncoder))
            
def runModel(input_file, target, model):
    
    class_names = ['Healthy', 'Diagnosed', 'Prospective']

#     Split data: 30% of data goes into the test set and predict_flag must remain false
    dataset = ParkinsonData(config, logger)
    if config['progression']:
        df = dataset.load_data(input_file)
    else:
        df = dataset.load_data(input_file, False)
    df_norm, norms = dataset.normalize_data(df, inplace=False)
    
#     Perform feature elimination and find the best features for this dataset
#     features, labels = df_norm[dataset.FEATURES].values, df_norm[dataset.STATUS].values
#     feat_mask, acc_log = dataset.perform_feature_elimination(model, features, labels)
#     df_norm = df_norm.drop(df_norm.columns[np.where(feat_mask == False)[0]], axis=1, inplace=True)
#     print (feat_mask, acc_log)

#     Perform the train-test split for further model fitting
    X_train, X_test, y_train, y_test = dataset.make_split_data(df_norm, 0.30, False)
    print ("Model hyperparameters:")
    print (model)
    
#     Do cross_validation for the MLP NN
#     cross_val_NN(model, X_train, y_train)  
#     cross_val_RF(model, X_train, y_train)

    ck_array = []
    
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    prob_in_class = model.predict_proba(X_test)
    ck_score = cohen_kappa_score(y_test, predicted)

    expected = y_test
    
#     ROC Curves for binary-class classification
#     roc(model, df_train, target)
    
    print ("Report that scores the ability of the model to predict the target on the test data:\n")
    print (metrics.classification_report(expected, predicted))
    print ()
    print (" * Precision Score = total positive results / sum(total positives, false positives)")
    print ("          The precision is intuitively the ability of the classifier not to label as") 
    print ("          positive a sample that is negative.")
    print ()
    print (" * Recall Score = total postivies / sum(total positives, false negatives)")
    print ("          The recall score is intuitively the ability of the classifier to find all") 
    print ("          the positive samples.")
    print ()
    print (" * F1-Score = (precision * recall)/(precision + recall)")
    print ("          The F1 score can be interpreted as a weighted average of the precision and") 
    print ("          recall")
    print ()
    print ("CONFUSION MATRIX")
    """
    Below is the work to plot out graphical confusion matricies. It plots CM's for both raw counts 
    as well as normalized values. This is useful as it allows us to understand how well the model is predicting (and mis-predicting) the results.
    """
    call_confusion_matrix(expected, predicted, class_names, os.path.join(config['base_path'],config['viz_dir']), 5)
    score =  model.score(X_test, y_test)
    print ("Overall Score is: %8.2f" % score)
    print ("______________________________________________________________________________________")
    print

    print("Cohen's Kappa Score compares estimator performance with a random baseline estimator.")
    print("Avg. Cohen-Kappa Score %8.2f\n" % ck_score)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', dest='config_file', type=str, help='Configuration file', required=True)
    args = parser.parse_args()
    
    config = load_config(args.config_file)
    logger.info('Using configuration: {}'.format(config))

    df = pd.read_csv(config['filename'])
    
    plot_correlation(df,config)
    plot_feature_impact_chart(df,'status',config)
    '''
    There are a number of models that can be selected for this research. Replace the model passed to the 'classifier' variable.

    - A Restricted Boltzmann machine pipeline (with a logistic regression classifier)
    - A Logistic Regression Classifier with built-in Cross Validation
    - A Random Forest Classifier
    - A Multi-Layer Perceptron Classifier
    - A Linear Discriminant Analysis Classifier

    '''
    
    #  RBM Pipeline
    rbm = BernoulliRBM(n_components = 200, learning_rate=0.001, n_iter=20)   
    logistic = linear_model.LogisticRegression(C=100.0, penalty='l2', solver = 'liblinear', tol=1e-6)
    model_rbm = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

    #  Logistic Regression with Cross-Validation
    logisticcv = linear_model.LogisticRegressionCV(class_weight='balanced', scoring='roc_auc', n_jobs=-1, max_iter=10000, verbose=1)

    #  Random Forest
    '''
    A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset.
    And uses averaging to improve the predictive accuracy and control over-fitting. 
    
    The main parameters to adjust when using this method is n_estimators and max_features.
    The former is the number of trees in the forest. The larger the better, but also the longer it will take to compute. 
        - In addition, note that results will stop getting significantly better beyond a critical number of trees. 
    The latter is the size of the random subsets of features to consider when splitting a node. 
        - The lower the greater the reduction of variance, but also the greater the increase in bias. 
    Empirical good default values are max_features=n_features for regression problems, and max_features=sqrt(n_features).
    Good results are often achieved when setting max_depth=None in combination with min_samples_split=1 (i.e., when fully developing the trees).
    '''
    model_rf = RandomForestClassifier(n_estimators=81, criterion='gini', bootstrap=False, min_samples_leaf = 4, min_samples_split = 8, max_depth = None)

    #  Multi-layer Perceptron (neural net)
    model_nn= MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, hidden_layer_sizes=(100,80,3))

    #  LDA Model
    model_lda = LDA(n_components=3)

    # finalize a model
    classifier = model_rf
    
    runModel(config['filename'], 'status', classifier)
    
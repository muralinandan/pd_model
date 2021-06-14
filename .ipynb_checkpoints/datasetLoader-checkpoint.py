import os
from copy import copy
import pandas as pd
import numpy as np
import math
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from matplotlib import pyplot as plt
import seaborn as sns


class ParkinsonData:
    SUB_NUM = 'subject#'
    SUB_AGE = 'age'
    TIME = 'time'
    STATUS = 'status'
    FEATURES = ['Jitter(%)','Jitter(Abs)','Jitter:RAP','Jitter:PPQ5','Jitter:DDP','Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA','NHR','HNR','DFA','PPE','age','time']
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        
    def get_time_encodings(self, n_position, emb_dim, padding_idx=None):
        '''
        These encodings help to encode the position/time component along with the existing speech features.
        Both the speech features as well as time embeddings are used for training.
        '''
        time_embedding = np.array([[pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)] if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
        time_embedding[:, 0::2] = np.sin(time_embedding[:, 0::2])  # dim 2i
        time_embedding[:, 1::2] = np.cos(time_embedding[:, 1::2])  # dim 2i+1

        if padding_idx is not None:
            # zero vector for padding dimension
            time_embedding[padding_idx] = 0.
        
        emb_columns = []
        for dim in range(emb_dim):
            emb_columns.append('time_'+str(dim))
        
        self.time_emb = time_embedding
        self.time_feats = emb_columns
    
    def get_time_difference(self,df):
        new_df = pd.DataFrame()
        grouped_df = df.groupby(ParkinsonData.SUB_NUM)
        min_df = grouped_df.min()
        max_df = grouped_df.max()
        diff_df = pd.DataFrame({'minTime':min_df['time'].values, 'maxTime':max_df['time'].values})
        max_time_difference = -1
        for index, row in diff_df.iterrows():
            curr_diff = row['maxTime'] - row['minTime']
            max_time_difference = max(max_time_difference, curr_diff)
        return max_time_difference
    
    def sample_pair_progression(self, df):
        np.random.seed(0)
        old_cols, new_cols = [feat+'_old' for feat in ParkinsonData.FEATURES if feat!='time'], [feat+'_new' for feat in ParkinsonData.FEATURES if feat!='time']
        old_cols.extend(new_cols)
        old_cols.extend(self.time_feats)
        ParkinsonData.FEATURES = old_cols
        old_cols.append(ParkinsonData.STATUS)
        
        grouped_df = df.groupby(ParkinsonData.SUB_NUM,as_index=False)
        new_data = []
        for name,group in tqdm(grouped_df,total=len(grouped_df)):
            for i in range(300):
                rows = np.asarray(group.sample(n=2).index)
                rowData1, rowData2 = group.loc[rows[0],:], group.loc[rows[1],:]
                time1, time2 = rowData1.time, rowData2.time
                older = rows[0] if rowData1.time > rowData2.time else rows[1]
                new_status = group.loc[older,ParkinsonData.STATUS]
                rowData1 = rowData1.drop([ParkinsonData.SUB_NUM, ParkinsonData.STATUS, ParkinsonData.TIME],inplace=False)
                rowData2 =  rowData2.drop([ParkinsonData.SUB_NUM, ParkinsonData.STATUS, ParkinsonData.TIME],inplace=False)
                older_curr, newer_curr = rowData1.values.tolist(), rowData2.values.tolist()
                curr_time = self.time_emb[int(abs(time1 - time2)-1),:]
                curr_data = older_curr
                curr_data.extend(newer_curr)
                curr_data.extend(curr_time)
                curr_data.append(new_status)
                new_data.append(curr_data)
        new_df = pd.DataFrame(new_data, columns = old_cols)
        return new_df
        
    def load_data(self, path = 'data.csv', progression = True):
        '''
        Load and clean data to remove missing or NaN values 
        '''
        file_path = os.path.join(self.config['base_path'],path)
        self.logger.info('Loading the dataset from - {}'.format(file_path))
        df = pd.read_csv(file_path)
        
        self.logger.info('Cleaning the dataset by removing "missing values" and "NaN"...')
        missing_vals = df.index[df.isna().any(axis=1)]
        if len(missing_vals) > 0:
            df = df.drop(index=missing_vals)
        df.reset_index(drop=True, inplace=True)
        
        if progression:
            max_diff = int(math.ceil(self.get_time_difference(df)))
            self.get_time_encodings(max_diff, self.config['emb_dim'])
            new_df = self.sample_pair_progression(df)
            return new_df
        return df
    
    def normalize_data(self, df, scaler=MinMaxScaler(), inplace=True):
        '''
        Scale the dataframe by applying the given 'scaler'
        Only the trainable-features of the dataset are scaled/normalized
        There are three scalers that are effective-
            - MaxAbsScaler is better with sparse data.
            - MinMaxScaler is best for our work.
            - StandardScaler shows similar results.
        '''
        self.logger.info('Normalizing the dataset...')
        data = df if inplace else df.copy()
        normalizers= {}
        for feature in ParkinsonData.FEATURES:
            # mean and std of the data
            print (feature)
            scaled_data = scaler.fit_transform(data[feature].values.reshape(-1, 1))
            data[feature] = scaled_data
            normalizers[feature] = copy(scaler)
        return normalizers if inplace else data, normalizers
    
    def make_split_data(self, df, size, flag=True):
        '''
        @param df: dataframe
        @param size: the size of the test set in percentage of the whole
        @param flag: if true, perform SMOT
        @return ->  'Train' and 'Test' as numpy nd-arrays, 'Target' values with status of [0,1,2] as a column vector
        
        Inbalanced Learning
        Inside this function there is the option to use imbalance learning to increase the sample size by creating intelligent, synthetic datapoints.
        This is a technique that is common when a dataset is unbalanced in favor of one class or if it's too small to yield consistent results.
        In this case, let's employ the Synthetic Minority Oversampling Technique (SMOT) algorithm. 
            https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

        TLDR; 
        To oversample using SMOT, take a sample from the dataset, and consider its k nearest neighbors (in feature space).
        To create a synthetic data point, take the vector between one of those k neighbors, and the current data point.
        Multiply this vector by a random number x which lies between 0, and 1. Add this to the current data point to create the new, synthetic data point.
        '''
        self.logger.info('Making the train, test split from the dataset...')
        labels = df[ParkinsonData.STATUS].values # select the 'target' column
        labels = np.array(labels).astype(int) # in case the target is a float

        features = df[ParkinsonData.FEATURES].values
        features = np.array(features)

        X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=size)
        if flag:
            # oversample the training set.
            self.logger.info('Performing the oversampling of the training set using SMOT...')
            smote = SMOTEENN()
            X_train, y_train = smote.fit_resample(X_train, y_train)
            
        return X_train, X_test, y_train, y_test
    
    def perform_feature_elimination(self, model, X, y, cross_vals=4):
        '''
        A function to perform recursive feature elimination on the dataset
        @param model: Classification model to analyze
        @param cross_vals: number of cross validation folds
        @return -> feature_mask: Dict to store if a feature is selected or not
        '''
        self.logger.info('Performing the recursive feature elimination...')
        feature_mask, accuracy_log = None, None
        cv_split = KFold(n_splits=10, shuffle=True)
        def plot_feature_vs_models(size=10):               
                all_scores, all_names = [], []
                for i in range(2, 17):
                    rfe = RFE(estimator=model, n_features_to_select=i,verbose=0)
                    pipe = Pipeline(steps=[('s',rfe),('m',model)])
                    scores = cross_val_score(pipe, X, y, scoring='accuracy', cv=cv_split, n_jobs=-1, error_score='raise',verbose=0)
                    all_scores.append(scores)
                    all_names.append(str(i))
                    print('{} - {} ({})'.format(i, np.mean(scores), np.std(scores)))
                with sns.axes_style("white"):
                    f, ax = plt.subplots(figsize=(size, size))
                    all_scores = np.array(all_scores).T
                    df = pd.DataFrame(all_scores, columns=all_names)
                    ax = sns.boxplot(data=df)
                    ax.set_title('Selected Features vs Classification Accuracy for ' + str(type(model).__name__))
                    ax.set_xlabel('# Features')
                    ax.set_ylabel('Accuracy')
                    for patch in ax.artists:
                        r, g, b, a = patch.get_facecolor()
                        patch.set_facecolor((r, g, b, .6))
                    plt.tight_layout()
                    f.savefig(os.path.join(self.config['base_path'],os.path.join(self.config['viz_dir'],'Selected Features vs Classification Accuracy for ' + str(type(model).__name__) + '.png')),bbox_inches='tight')
                    plt.close()
        
#         plot_feature_vs_models()
        rfecv = RFECV(estimator=model, step=1, cv=cv_split, scoring='accuracy', n_jobs=-1, verbose=0)
        rfecv.fit(X, y)
        for i in range(X.shape[1]):
            print('Column: %d, Selected %s, Rank: %.3f' % (i, rfecv.support_[i], rfecv.ranking_[i]))
        feature_mask = rfecv.support_
        accuracy_log = rfecv.grid_scores_
        return feature_mask, accuracy_log
                    
        
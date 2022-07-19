from sklearn.base import TransformerMixin
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as sparse_hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
import numpy as np
import pandas as pd

class numeric_Transformer(TransformerMixin):
    '''
    Imputes all numeric columns with their medians on the training set.
    Parameters
    ----------
    None
    Returns
    ----------
    Sparse Matrix
            Matrix consisting of the numeric columns after imputation
            
    '''

    
    def __init__(self):
        self.numeric_features = ['bmi']
        self.num_medians = np.zeros(len(self.numeric_features))

    def fit(self, X, y=None):
        '''
        Learn the median values of the numeric columns
        Parameters
        ----------
        None
        Returns
        ----------
        self
        ''' 
        X = X.copy()
        self.num_medians = X.median()
        self.numeric_features = X.columns
        return self
    
    def transform(self, X, y=None):
        '''
        Transforms all anon numeric to a sparse matrix
        Parameters
        ----------
        None
        Returns
        ----------
        Sparse Matrix
            Matrix consisting of all numeric features after imputation with median values
        '''   
        X_copy = X.copy()
        if X_copy.shape[1] == 1:
            for col in X_copy.columns:
                if col in self.numeric_features and X_copy[col] == np.nan:
                    X_copy[col] = self.num_medians[col]
        else:
            for col in X_copy.columns:
                X_copy[col].fillna(self.num_medians[col], inplace=True)

        return X_copy
    
    def get_feature_names(self):
        '''
        returns a list of feature names consisting of each of the numeric_features.
        Parameters
        ----------
        None
        Returns
        ----------
        List
            A list of feature names
        '''   
        return [x.lower() for x in self.numeric_features]    
    
class standardscaler2:
    def __init__(self):
        self.feat_means = []
        self.feat_stds =[]
    def fit(self, df):
        for col in df.columns:
            self.feat_means.append(df[col].mean())
            self.feat_stds.append(df[col].std())
        return self
    def transform(self, df):
        for col, mean, std in zip(df.columns, self.feat_means, self.feat_stds):
            for row in df.index.tolist():
                if std!=0:
                    df.loc[row,col] = ( df.loc[row,col] - mean)/std
                else:
                    df.loc[row,col] =  df.loc[row,col] - mean                    
        return df
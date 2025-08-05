
# based on https://www.kaggle.com/code/tolgadincer/upsampling-multilabel-data-with-mlsmote
from typing import Tuple
import numpy as np
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors


class MultiLabelSMOTE:
    """
    MultiLabelSMOTE class for generating synthetic samples for multi-label datasets.
    This class implements the MLSMOTE algorithm to handle imbalanced multi-label datasets.
    """

    def __init__(self, n_sample=100, neigh=5, random_state=None):
        self.n_sample = n_sample
        self.neigh = neigh
        self.random_state = random_state

    def fit_resample(self, X, y) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit the model and generate synthetic samples.
        args:
            X: pandas.DataFrame, input feature vector DataFrame
            y: pandas.DataFrame, target vector DataFrame
        return:
            updated_X: pandas.DataFrame, augmented feature vector DataFrame
            updated_Y: pandas.DataFrame, augmented target vector DataFrame
        """
        if self.random_state is not None:
            random.seed(self.random_state)
        X_sub, y_sub = self.get_minority_samples(X, y, ql=[0.05, 1.])
        new_X, new_y = self.generate_samples(X_sub, y_sub)
        updated_X = pd.concat([X, new_X], ignore_index=True)
        updated_Y = pd.concat([y, new_y], ignore_index=True)
        return updated_X, updated_Y

    def get_tail_label(self, df: pd.DataFrame, ql=[0.05, 1.]) -> list:
        irlbl = df.sum(axis=0)
        irlbl = irlbl[(irlbl > irlbl.quantile(ql[0])) & (
            (irlbl < irlbl.quantile(ql[1])))]  # Filtering
        irlbl = irlbl.max() / irlbl
        threshold_irlbl = irlbl.median()
        tail_label = irlbl[irlbl > threshold_irlbl].index.tolist()
        return tail_label

    def get_minority_samples(self, X: pd.DataFrame, y: pd.DataFrame, ql=[0.05, 1.]):
        tail_labels = self.get_tail_label(y, ql=ql)
        index = y[y[tail_labels].apply(lambda x: (
            x == 1).any(), axis=1)].index.tolist()

        X_sub = X[X.index.isin(index)].reset_index(drop=True)
        y_sub = y[y.index.isin(index)].reset_index(drop=True)
        return X_sub, y_sub

    def nearest_neighbour(self, X: pd.DataFrame, neigh) -> list:
        nbs = NearestNeighbors(
            n_neighbors=neigh, metric='euclidean', algorithm='kd_tree').fit(X)
        _, indices = nbs.kneighbors(X)
        return indices

    def generate_samples(self, X, y):
        """
        Give the augmented data using MLSMOTE algorithm

        args
        X: pandas.DataFrame, input vector DataFrame
        y: pandas.DataFrame, feature vector dataframe
        n_sample: int, number of newly generated sample

        return
        new_X: pandas.DataFrame, augmented feature vector data
        target: pandas.DataFrame, augmented target vector data
        """
        indices2 = self.nearest_neighbour(X, neigh=5)
        n = len(indices2)
        new_X = np.zeros((self.n_sample, X.shape[1]))
        target = np.zeros((self.n_sample, y.shape[1]))
        for i in range(self.n_sample):
            reference = random.randint(0, n-1)
            neighbor = random.choice(indices2[reference, 1:])
            all_point = indices2[reference]
            nn_df = y[y.index.isin(all_point)]
            ser = nn_df.sum(axis=0, skipna=True)
            target[i] = np.array([1 if val > 0 else 0 for val in ser])
            ratio = random.random()
            gap = X.loc[reference, :] - X.loc[neighbor, :]
            new_X[i] = np.array(X.loc[reference, :] + ratio * gap)
        new_X = pd.DataFrame(new_X, columns=X.columns)
        target = pd.DataFrame(target, columns=y.columns)
        return new_X, target

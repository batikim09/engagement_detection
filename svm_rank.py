import numpy as np
np.random.seed(1337)
from sklearn.metrics import f1_score,recall_score,confusion_matrix
from sklearn import svm, linear_model, cross_validation
import scipy.stats as st
from helper import transform_pairwise

class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model

    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.

    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """
    def fit(self, X, y):
        """
        Fit a pairwise ranking model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)

        Returns
        -------
        self
        """
        X_trans, y_trans = transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_.ravel())

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        The item is given such that items ranked on top have are
        predicted a higher ordering (i.e. 0 means is the last item
        and n_samples would be the item ranked on top).

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.argsort(np.dot(X, self.coef_.ravel()))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)

        pred = super(RankSVM, self).predict(X_trans)

        #tau distance
        tau_d =  1. - np.mean(pred == y_trans)
        #tau correlation
        tau, p_value = st.kendalltau(pred, y_trans)
        #confusion matrix
        cm = confusion_matrix(y_trans, pred)
        prob_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        X_errors = X_trans[pred != y_trans]
        Y_errors = y_trans[pred != y_trans]
        X_Y_errors = np.c_[ X_errors, Y_errors ]

        X_correct = X_trans[pred == y_trans]
        Y_correct = y_trans[pred == y_trans]
        X_Y_correct = np.c_[ X_correct, Y_correct ]

        return tau_d, tau, cm, prob_cm, X_Y_errors, X_Y_correct


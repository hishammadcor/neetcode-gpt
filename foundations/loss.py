import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        '''old solution
        n = len(y_true)
        assert n == len(y_pred), "n should be equal to the length of y_pred"

        false_negative = lambda i: (y_true[i]*np.log(y_pred[i]+1e-7))
        false_positive = lambda i: (1-y_true[i])*np.log(1-(y_pred[i]+1e-7)) 
        L = -(1/n)*sum(false_negative(i) + false_positive(i) for i in range(n))
        return round(L,4)
        '''

        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1-eps)

        ans = round(-np.mean((y_true * np.log(y_pred))+ (1-y_true)*(np.log(1-y_pred))),4)

        return ans

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        '''old solution
        n,c = y_true.shape

        entropy = lambda i,j: (y_true[i][j]*np.log(y_pred[i][j]+1e-7))
        
        L = -(1/n)*sum(sum(entropy(i,j) for j in range(c)) for i in range(n))
        return round(L,4)
        '''
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1-eps)

        ans = round(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)),4)

        return ans
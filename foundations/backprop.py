import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = np.dot(w,x)+b
        y_hat = 1/(1+math.exp(-z))
        L = 0.5 *(y_hat - y_true)**2
        
        error = y_hat - y_true
        sig_derv = y_hat*(1-y_hat)

        dl_db = np.round(error * sig_derv,5)
        dl_dw = np.round(np.dot(dl_db,x),5)

        return (dl_dw, dl_db)
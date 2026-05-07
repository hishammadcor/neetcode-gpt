import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        # Forward Pass
        z1 = np.dot(W1,x) + b1
        a = np.maximum(0,z1)
        z2 = np.dot(W2,a) +b2

        # MSE Loss
        loss = np.round(np.mean((z2-y_true)**2),4)
        
        # Backward Pass

        # Output Gradient (dL/dz2)
        dz2 = 2*(z2-y_true)/len(z2)

        # Layer 2 Weight Gradient (dL/dW2)
        dw2 = np.outer(dz2,a)

        # Layer 2 Bias Gradient (dL/db2)
        db2 = dz2

        # Gradient through ReLU (dL/dz1)
        da = np.dot(dz2,W2)
        dz1 = da * (z1 > 0)

        # Layer 1 Weight Gradient (dL/dW1)
        dw1 = np.outer(dz1,x)

        # Layer 1 Bias Gradient (dL/db1)
        db1 = dz1


        result = {'loss':loss,
                  'dW1':(np.round(dw1,4)+0.0).tolist(),
                  'db1':(np.round(db1,4)+0.0).tolist(),
                  'dW2':(np.round(dw2,4)+0.0).tolist(),
                  'db2':(np.round(db2,4)+0.0).tolist()
        }

        return result
        

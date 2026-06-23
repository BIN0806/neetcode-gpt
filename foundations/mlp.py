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
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        
        res = {}

        # Forward ass
        z1 = x @ W1.T + b1     # pre-activation Layer 1
        a1 = np.maximum(z1, 0) # ReLu Activation
        z2 = a1 @ W2.T + b2    # output (predictions)
        L = np.mean((z2 - y_true)** 2)

        # Backward Pass
        n = len(y_true) 
        dz2 = 2 / n * (z2 - y_true) # dL/dz2
        dW2 = dz2.reshape(-1, 1) @ a1.reshape(1, -1) # outer product
        db2 = dz2 

        da1 = dz2 @ W2
        dz1 = da1 * (z1 > 0) # mask an array
        dW1 = dz1.reshape(-1, 1) @ x.reshape(1, -1)
        db1 = dz1

        res["loss"] = np.round(L, 4)
        res["dW1"] = np.round(dW1, 4).tolist()
        res["db1"] = np.round(db1, 4).tolist()
        res["dW2"] = np.round(dW2, 4).tolist()
        res["db2"] = np.round(db2, 4).tolist()
        return res
import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        output = x
        n = len(weights)
        for i in range(n):
            weight, b = weights[i], biases[i]
            output = output @ weight + b
            if i < n - 1:
                output = np.round(np.maximum(output, 0), 5) # np.maximum is vector max comparison

        return np.round(output, 5)

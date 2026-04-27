import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        max_z = max(z)
        score = []
        for i in z:
            s = round((np.exp(i-max_z))/(sum(np.exp(z-max_z))),4)
            score.append(s)
        return np.array(score)
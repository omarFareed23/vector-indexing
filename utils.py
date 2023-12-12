import numpy as np
import random
from scipy.cluster.vq import kmeans2


class VectorUtils:
    def __init__(self, vector):
        self.vector = vector

    @staticmethod
    def product_quantization(vector: list, num_subvectors: int) -> list:
        return [
            np.mean(sub_vector)
            for sub_vector in np.array_split(vector, num_subvectors)
        ]

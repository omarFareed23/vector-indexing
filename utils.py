import numpy as np
class VectorUtils:
    def __init__(self, vector):
        self.vector = vector

    @staticmethod
    def product_quantization(vector: list, num_subvectors: int, num_bits: int) -> list:
        quantized_vector = []
        for subvector in np.array_split(vector, num_subvectors):
            quantized_value = int(round(np.average(subvector)))
            quantized_vector.append(quantized_value)
        return quantized_vector
    
    
print(VectorUtils.product_quantization([19,12,1,4,5,6,7,8], 4, 2))

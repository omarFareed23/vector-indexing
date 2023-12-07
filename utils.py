import numpy as np
class VectorUtils:
    def __init__(self, vector):
        self.vector = vector

    @staticmethod
    def product_quantization(vector: list, num_subvectors: int, num_bits: int) -> list:
        quantized_vector = []
        for subvector in np.array_split(vector, num_subvectors):
            quantized_value = VectorUtils._quantize(subvector, num_bits)
            quantized_vector.append(quantized_value)
            
    @staticmethod
    def _quantize(vector: list, num_bits: int) -> int:
        max_value = max(vector)
        min_value = min(vector)
        range_value = max_value - min_value
        step = range_value / (2 ** num_bits)
        quantized_value = int((np.average(vector) - min_value) / step)
        return quantized_value
    
    
print(VectorUtils._quantize([4,5,6] ,2))

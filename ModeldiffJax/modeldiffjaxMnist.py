import jax
import numpy as np
import jax.numpy as jnp
from scipy.spatial import distance

# Jax porting modeldiff-Mnist
model_name = "PrakhAI/DigitGAN"


def quantize(tensor, scale, zero_point)
    return np.round(tensor / scale + zero_point).astype(np.int8)

def dequantize(qtensor, scale, zero_point):
    return (qtensor.astype(np.float32) - zero_point) * scale

model_name = "PrakhAI/DigitGAN"
# model1 =



# Quantization
import jax
import jax.numpy as jnp
import numpy as np

def quantize_to_int8(x, scale):
    # Quantize floating point array to int8
    return jnp.clip(jnp.round(x * scale), -128, 127).astype(jnp.int8)

def dequantize_from_int8(x, scale):
    # Convert int8 array back to floating point
    return x / scale

def custom_matmul_int8(a, b):
    # Calculate scale factors for quantization
    scale_a = 127 / jnp.max(jnp.abs(a))
    scale_b = 127 / jnp.max(jnp.abs(b))

    # Quantize
    a_int8 = quantize_to_int8(a, scale_a)
    b_int8 = quantize_to_int8(b, scale_b)

    # Matrix multiplication in int8
    c_int32 = jnp.matmul(a_int8.astype(jnp.int32), b_int8.astype(jnp.int32))  # Ensure int32 computation

    # Dequantize
    return dequantize_from_int8(c_int32, scale_a * scale_b)

# Example usage
key_a = jax.random.PRNGKey(0)
key_b = jax.random.PRNGKey(1)
a = jax.random.normal(key_a, (3, 4))
b = jax.random.normal(key_b, (4, 5))

# Perform quantized matrix multiplication
result = custom_matmul_int8(a, b)
print("Quantized matrix multiplication result:", result)

# runtime error
def get_scale(weight):
    max_abs_val = np.max(np.abs(weight))
    if max_abs_val == 0:
        return 0  # Avoid division by zero; no scaling needed for a zero matrix
    return 127 / max_abs_val



# Load pre-trained weights (assumed to be already loaded into 'weights')
weights = np.load('ViT-B_32.npz', allow_pickle=True)

# Apply quantization to all applicable layers
quantized_weights = {}
# for key, weight in weights.items():
#     if "kernel" in key:  # Assuming 'kernel' indicates a weight matrix for matmul
#         # Apply quantization
#         scale = 127 / np.max(np.abs(weight))
#         quantized_weights[key] = quantize_to_int8(jnp.array(weight), scale)
#     else:
#         # Copy biases and other parameters without quantization
#         quantized_weights[key] = weight

for key, weight in weights.items():
    if "kernel" in key:  # Assuming 'kernel' indicates a weight matrix for matmul
        # Apply quantization
        scale = get_scale(weight)
        if scale != 0:
            quantized_weights[key] = quantize_to_int8(jnp.array(weight), scale)
        else:
            quantized_weights[key] = weight  # Assign original weights if scale is zero
    else:
        # Copy biases and other parameters without quantization
        quantized_weights[key] = weight

# Save the quantized model
np.savez('ViT-B_8.npz', **quantized_weights)
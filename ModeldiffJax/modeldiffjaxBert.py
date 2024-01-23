import jax
import numpy as np
from scipy.spatial import distance
from transformers import FlaxBertForSequenceClassification, BertTokenizer

# Load Model and Tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Original model (model1)
model1 = FlaxBertForSequenceClassification.from_pretrained(model_name)
# Define Quantization and Dequantization Functions
def quantize(tensor, scale, zero_point):
    return np.round(tensor / scale + zero_point).astype(np.int8)

def dequantize(qtensor, scale, zero_point):
    return (qtensor.astype(np.float32) - zero_point) * scale

# Load Model and Tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = BertTokenizer.from_pretrained(model_name)
model1 = FlaxBertForSequenceClassification.from_pretrained(model_name)

# Quantize Model Parameters
scale = 0.02
zero_point = 0
model2 = jax.tree_map(lambda x: quantize(x, scale, zero_point), model1.params)

# Prepare Input
text = "It's great!!"
inputs = tokenizer(text, return_tensors='jax', max_length=512, truncation=True, padding='max_length')

# Inference with Original Model (model1)
outputs1 = model1(**inputs)
predicted_class1 = jax.nn.softmax(outputs1.logits, axis=-1).argmax(-1)

# Inference with Quantized Model (model2)
def infer_with_quantized_model(quantized_params, inputs):
    # Dequantize weights for inference
    dequantized_weights = jax.tree_map(lambda x: dequantize(x, scale, zero_point), quantized_params)

    temp_model = FlaxBertForSequenceClassification(model1.config, dtype=jax.numpy.float32)
    temp_model.params = dequantized_weights

    return temp_model(**inputs)

outputs2 = infer_with_quantized_model(model2, inputs)
predicted_class2 = jax.nn.softmax(outputs2.logits, axis=-1).argmax(-1)

# Compare Results
print(f"Prediction with Original Model (model1): {predicted_class1}")
print(f"Prediction with Quantized Model (model2): {predicted_class2}")



def compute_cosine_distance(a, b):
    return distance.cosine(a, b)

def batch_forward(model, inputs):
    return model(**inputs).logits

def compute_ddv(model, inputs):
    dists = []
    outputs = batch_forward(model, inputs)
    n_pairs = int(outputs.shape[0] / 2)
    for i in range(n_pairs):
        ya = outputs[i]
        yb = outputs[i + n_pairs]
        dist = compute_cosine_distance(ya, yb)
        dists.append(dist)
    return jnp.array(dists)

def compute_similarity_with_ddv(model1, model2, profiling_inputs):
    ddv1 = compute_ddv(model1, profiling_inputs)
    ddv2 = compute_ddv(model2, profiling_inputs)

    normalized_ddv1 = jax.nn.normalize(ddv1)
    normalized_ddv2 = jax.nn.normalize(ddv2)

    ddv_distance = jnp.mean(jnp.abs(normalized_ddv1 - normalized_ddv2))
    model_similarity = 1 - ddv_distance

    return model_similarity

# Example usage
tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
text_pairs = ["Text 1", "Text 2", "Text 3", "Text 4"]  # Replace with actual text pairs
inputs = tokenizer(text_pairs, return_tensors='jax', padding=True, truncation=True)

model_similarity = compute_similarity_with_ddv(model1, model2, inputs)
print(f"Model Similarity: {model_similarity}")




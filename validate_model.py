import os
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training import checkpoints
import tensorflow as tf
import tqdm

# Define the VisionTransformer or Mixer based on the model_name
def create_model(model_config, num_classes=10):
    if model_config['name'].startswith('Mixer'):
        return models.MlpMixer(num_classes=num_classes, **model_config)
    else:
        return models.VisionTransformer(num_classes=num_classes, **model_config)

# Configuration of the model (needs to match with downloaded/pretrained model config)
model_config = {
    'name': 'ViT-B_32',
    'patches': {'size': (32, 32)},
    'hidden_size': 768,
    'transformer': {
        'attention_dropout_rate': 0.0,
        'dropout_rate': 0.0,
        'mlp_dim': 3072,
        'num_heads': 12,
        'num_layers': 12
    }
}

# Path to the pre-trained weights
model_name = 'ViT-B_32'
weights_path = f'{model_name}.npz'

# Ensure the pretrained model weights are downloaded
if not os.path.exists(weights_path):
    if model_name.startswith('ViT'):
        os.system(f'gsutil cp gs://{model_name}.npz .')
    elif model_name.startswith('Mixer'):
        os.system(f'gsutil cp gs://{model_name}.npz .')
assert os.path.exists(weights_path), "Model weights not found!"

# Load CIFAR-10 Data
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0
    y_test = tf.one_hot(y_test.flatten(), 10)
    return x_test, y_test

x_test, y_test = load_cifar10()

# Initialize the model
model = create_model(model_config)

# Load pretrained weights
variables = model.init(jax.random.PRNGKey(0), x_test[:1], train=False)
params = checkpoints.restore_checkpoint(ckpt_dir='.', target=variables, prefix=weights_path)

# Define evaluation function
def evaluate_model(model, params, x_test, y_test, batch_size=64):
    steps = len(x_test) // batch_size
    accuracy = []
    for i in tqdm.trange(steps):
        batch_images = x_test[i * batch_size:(i + 1) * batch_size]
        batch_labels = y_test[i * batch_size:(i + 1) * batch_size]
        logits = model.apply(params, batch_images, train=False)
        predicted_classes = jnp.argmax(logits, axis=1)
        true_classes = jnp.argmax(batch_labels, axis=1)
        accuracy.append(jnp.mean(predicted_classes == true_classes))
    return np.mean(accuracy)

# Calculate accuracy
accuracy = evaluate_model(model, params, x_test, y_test)
print(f"Model accuracy on CIFAR-10: {accuracy:.4f}")

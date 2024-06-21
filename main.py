import os

import jax.numpy as jnp
from datasets import Image
# from jax.experimental import stax
# from jax.experimental.stax import serial
import jax.nn as nn
import jax.random as random
import numpy as np
from opt_einsum.backends import jax
from scipy.spatial import distance
# Load CIFAR-10 data
from tensorflow.keras.datasets import cifar10
from flax import linen as nn

import tensorflow as tf


def load_and_preprocess_cifar10():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32') / 255.0  # Normalize the data
    y_test = tf.one_hot(y_test.flatten(), 10)  # One-hot encode labels
    return x_test, y_test


x_test, y_test = load_and_preprocess_cifar10()
from models_vit import VisionTransformer
from flax import linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
import numpy as np


def create_model(num_classes, patches, hidden_size, transformer_config):
    model = VisionTransformer(
        num_classes=num_classes,
        patches=patches,
        transformer=transformer_config,
        hidden_size=hidden_size,
        classifier='gap'  # assuming we use 'gap' for CIFAR-10
    )
    return model


# Assuming the function to bind weights to model
def bind_params(model, weights):
    # Create dummy input to initialize parameters
    dummy_input = jnp.ones((1, 32, 32, 3), jnp.float32)
    params = model.init(jax.random.PRNGKey(0), dummy_input, train=False)
    new_params = FrozenDict(weights)
    # Bind the parameters
    return model.apply({'params': new_params}, dummy_input, train=False)


# Load weights and create models
weights1 = np.load('ViT-B_32.npz', allow_pickle=True)
weights2 = np.load('ViT-B_8.npz', allow_pickle=True)

model_vit_b32 = create_model(10, {'size': (32, 32)}, 768, {'num_heads': 12, 'mlp_dim': 3072, 'num_layers': 12})
model_vit_b8 = create_model(10, {'size': (32, 32)}, 768, {'num_heads': 12, 'mlp_dim': 3072, 'num_layers': 12})

model1 = bind_params(model_vit_b32, weights1)
model2 = bind_params(model_vit_b8, weights2)


def test_model(model, x_test, y_test):
    # Assuming batch processing for handling whole dataset might be large
    def batch_predict(model, x):
        return jax.vmap(model, in_axes=(0, None))(x, train=False)

    predictions = batch_predict(model, x_test)
    predicted_classes = jnp.argmax(predictions, axis=1)
    true_classes = jnp.argmax(y_test, axis=1)
    accuracy = jnp.mean(predicted_classes == true_classes)
    print(f"Accuracy: {accuracy:.4f}")


# Evaluate both models
test_model(model1, x_test, y_test)
test_model(model2, x_test, y_test)


# DeepArc layer activation hook
def capture_activation(name):
    activations = {}
    def hook(module, inputs, outputs):
        activations[name] = outputs
    return hook

image_path = "/cifar10"

activations = {}
def get_activations(image):
    for name, layer in jax.tree_leaves(model1):
        if name.startswith("layer"):  # Check if layer name starts with "layer"
            layer.register_forward_hook(capture_activation(name))
    # Preprocess CIFAR-10 image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    # Run the model and get activations
    activations = model1(image)
    return activations


# DeepArc layer similarity check
def centering(K):
    n = K.shape[0]
    unit = jnp.ones([n, n])
    I = jnp.eye(n)
    H = I - unit / n
    return jnp.dot(H, jnp.dot(K, H))
def HSIC(Kx, Ky):
    return jnp.trace(jnp.dot(Kx, Ky))

def linear_CKA(X, Y):
    Kx = jnp.dot(X, X.T)
    Ky = jnp.dot(Y, Y.T)
    Kx_centered = centering(Kx)
    Ky_centered = centering(Ky)
    hsic = HSIC(Kx_centered, Ky_centered)
    var_x = HSIC(Kx_centered, Kx_centered)
    var_y = HSIC(Ky_centered, Ky_centered)
    cka_score = hsic / jnp.sqrt(var_x * var_y)
    return cka_score

  def calculate_cka_scores(activations, layer_prefix="layer"):
    cka_scores = []
    for i in range(len(activations) - 1):
        layer_name_i = layer_prefix + str(i)
        layer_name_i_plus_1 = layer_prefix + str(i + 1)

        A = activations[layer_name_i]
        B = activations[layer_name_i_plus_1]

        cka_score = linear_CKA(A, B)
        cka_scores.append(cka_score)

        print(f"{layer_name_i} and {layer_name_i_plus_1} : {cka_score:.3f}")

    return cka_scores


cka_scores = calculate_cka_scores(activations)

# CKA score 낮은 레이어 fault_localized_layers에 넣
fault_localized_layers = []
if cka_scores <= 90:
    fault_localized_layers.append()

# modeldiff - layer ddv calculate

def get_seed_inputs(model1, model2, N_INPUT_PAIRS, rand=False):
    """Generates seed inputs by concatenating outputs from both models."""
    key = random.PRNGKey(0)  # Assuming a fixed seed for reproducibility
    seed_inputs_model1 = model1(random.uniform(key, (N_INPUT_PAIRS, *model1.input_shape), dtype=jnp.float32))
    seed_inputs_model2 = model2(random.uniform(key, (N_INPUT_PAIRS, *model2.input_shape), dtype=jnp.int8))
    seed_inputs = jnp.concatenate([seed_inputs_model1, seed_inputs_model2])
    return seed_inputs



# gen_input code


def _gen_profiling_inputs_search(comparator, seed_inputs, use_torch=False, epsilon=0.2):
    input_shape = seed_inputs.shape[1:]
    n_inputs = seed_inputs.shape[0]
    max_iterations = 10
    # max_steps = 10
    model1 = comparator.model1
    model2 = comparator.model2

    ndims = jnp.prod(input_shape).item()

    initial_outputs1 = model1.batch_forward(seed_inputs)
    initial_outputs2 = model2.batch_forward(seed_inputs)

    def evaluate_inputs(inputs):
        outputs1 = model1.batch_forward(inputs)
        outputs2 = model2.batch_forward(inputs)
        metrics1 = comparator.input_metrics(comparator.model1, inputs)
        metrics2 = comparator.input_metrics(comparator.model2, inputs)

        output_dist1 = jnp.mean(distance.cdist(outputs1, initial_outputs1, 'euclidean').diagonal())
        output_dist2 = jnp.mean(distance.cdist(outputs2, initial_outputs2, 'euclidean').diagonal())
        print(f'  output distance: {output_dist1},{output_dist2}')
        print(f'  metrics: {metrics1},{metrics2}')

        return output_dist1 * output_dist2 * metrics1 * metrics2

    inputs = seed_inputs
    score = evaluate_inputs(inputs)
    print(f'score={score}')

    key = random.PRNGKey(0)

    for i in range(max_iterations):
        comparator._compute_distance(inputs)
        print(f'mutation {i}-th iteration')

        key, subkey = random.split(key)
        mutation_pos = random.randint(subkey, (), 0, ndims)
        mutation = jnp.zeros(ndims)
        mutation = mutation.at[mutation_pos].set(epsilon)
        mutation = mutation.reshape(input_shape)

        key, subkey = random.split(key)
        mutation_idx = random.randint(subkey, (), 0, n_inputs)
        mutation_batch = jnp.zeros_like(inputs)
        mutation_batch = mutation_batch.at[mutation_idx].set(mutation)

        mutate_right_inputs = inputs + mutation_batch
        mutate_right_score = evaluate_inputs(mutate_right_inputs)
        mutate_left_inputs = inputs - mutation_batch
        mutate_left_score = evaluate_inputs(mutate_left_inputs)

        if mutate_right_score <= score and mutate_left_score <= score:
            continue
        if mutate_right_score > mutate_left_score:
            print(f'mutate right: {score}->{mutate_right_score}')
            inputs = mutate_right_inputs
            score = mutate_right_score
        else:
            print(f'mutate left: {score}->{mutate_left_score}')
            inputs = mutate_left_inputs
            score = mutate_left_score
    return inputs







# from fault layer -> compute selected layers





# compare faults

def compare_faults(model1, model2, faults, gen_inputs=None, use_jax=True):
    # Generate seed inputs
    seed_inputs = get_seed_inputs(model1, model2, N_INPUT_PAIRS=10)  # Adjust N_INPUT_PAIRS if needed

    # Generate profiling inputs (replace with your actual input generation logic)
    if not gen_inputs:
        profiling_inputs = seed_inputs  # Placeholder, replace with actual input generation
    else:
        profiling_inputs = gen_inputs(model1, model2, seed_inputs, use_jax=use_jax)

    # Extract activations for faulty layers
    activations_model1 = extract_activations(model1, faults, profiling_inputs)
    activations_model2 = extract_activations(model2, faults, profiling_inputs)

    # Compute DDV for faulty layers
    fault_ddv_similarities = [] #fault_layer from Deeparc
    for fault_layer, ddv_sim in zip(faults, fault_ddv_similarities):
        print(f"Fault layer: {fault_layer}, DDV similarity: {ddv_sim:.3f}")
        if ddv_sim < 0.90:  # Identify fault-localized layers below 90% similarity
            fault_localized_layers.append(fault_layer)
        print(f"Fault-localized layers: {fault_localized_layers}")
        return fault_localized_layers


def extract_activations(model, layer_names, inputs):
    """Extracts activations from specified layers in a model."""
    activations = {}

    def store_activation(name, module, inputs, outputs):
        if name in layer_names:
            activations[name] = outputs

    model.apply(inputs, mutable=False, methods={'__call__': store_activation})
    return activations


# Compute fault layers’ ddv similarity


def forward(model, params, inputs):
    return model.apply(params, inputs)

activations = {}
def compute_similarity_for_selected_layers(model1, params1, model2, params2, inputs, faults):
    def get_outputs(model, params, inputs):
        def save_activations(layer_name, layer_fn, layer_inputs):
            output = layer_fn(layer_inputs)
            activations[layer_name] = output
            return output
        traced_model = jax.tree_map(lambda fn: jax.vmap(lambda x: save_activations(fn.__name__, fn, x)), model)
        _ = traced_model.apply(params, inputs)
        return activations

    # Get outputs from both models
    activations1 = get_outputs(model1, params1, inputs)
    activations2 = get_outputs(model2, params2, inputs)

    # Prepare to compute cosine similarity for selected layers
    feature_dists = {}
    layer_names = list(activations1.keys())

    selected_layers = [layer_names[i] for i in faults if i < len(layer_names)]
    repairing_layers = []

    for layer_name in selected_layers:
        if layer_name in activations2:
            feature1 = activations1[layer_name].ravel()
            feature2 = activations2[layer_name].ravel()
            dist = jnp.dot(feature1, feature2) / (jnp.linalg.norm(feature1) * jnp.linalg.norm(feature2))
            feature_dists[layer_name] = dist.item()
            print(f"Layer: {layer_name}, Similarity: {feature_dists[layer_name]:.4f}")
            if feature_dists[layer_name] < 0.9:
                repairing_layers.append(layer_name)

    return feature_dists, repairing_layers


def compare_ddv_activations(ddv1, ddv2):
    average_distance = jnp.mean(ddv1 - ddv2)
    return average_distance














# Phase (2)
# dequantize layer

def dequantize_faults(model1, model2, fault_localized_layers, save_dir='dequantized_faults'):
    """
  model2의 fault_localized_layers에 해당하는 레이어를 디퀀타이즈합니다.
  만약 레이어가 디퀀타이즈 불가능하면, 해당 레이어를 model1에서 복사합니다.
  모든 디퀀타이즈된 weight를 하나의 파일 (modified_Vit-B.npy)에 저장합니다.

  Args:
    model1: 디퀀타이즈 전의 원본 모델입니다.
    model2: 디퀀타이즈할 모델입니다.
    fault_localized_layers: DDM 유사도가 낮은 (fault_localized) 레이어들의 이름 목록입니다.
    save_dir: 디퀀타이즈된 weight 저장할 디렉토리 경로입니다. (default: 'dequantized_faults')
  """

    # 만약 save_dir 디렉토리가 없으면 생성합니다.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 두 모델의 module tree (내부 레이어 구조)를 가져옵니다.
    module_tree1 = model1.module_dict
    module_tree2 = model2.module_dict

    # 모든 디퀀타이즈된 weight를 담을 빈 리스트입니다.
    dequantized_weights = []

    # fault_localized_layers 리스트에 있는 레이어마다 반복합니다.
    for fault_layer in fault_localized_layers:
        layer_module2 = module_tree2[fault_layer]

        # 해당 레이어가 디퀀타이즈 가능한지 확인합니다.
        if hasattr(layer_module2, 'qparams'):
            # 디퀀타이즈 가능하면, 기존 코드처럼 진행합니다.
            qparams = layer_module2.qparams
            dequantized_weight = qparams.dequantize(layer_module2.params)
            dequantized_weights.append(dequantized_weight)
            print(f"레이어 {fault_layer}의 디퀀타이즈된 weight 를 dequantized_weights 리스트에 추가했습니다.")
        else:
            # 디퀀타이즈 불가능하면, model1에서 레이어 복사를 시도합니다.
            try:
                layer_module1 = module_tree1[fault_layer]
                copied_weight = layer_module1.params
                dequantized_weights.append(copied_weight)
                print(f"레이어 {fault_layer}의 weight 를 model1에서 복사하여 dequantized_weights 리스트에 추가했습니다.")
            except KeyError:  # Layer not found in model1
                print(f"레이어 {fault_layer}이 model1에 없으므로, 디퀀타이즈를 건너뜁니다.")

    # 모든 레이어 디퀀타이즈/복사 완료 후, weight 리스트를 하나의 파일로 저장합니다.
    save_path = os.path.join(save_dir, 'modified_Vit-B.npy')
    np.save(save_path, dequantized_weights)
    print(f"모든 디퀀타이즈된 weight를 {save_path} 파일에 저장했습니다.")

# test code

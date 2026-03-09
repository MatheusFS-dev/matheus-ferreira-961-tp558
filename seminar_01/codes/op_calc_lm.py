import os

# Specify GPU to use (e.g., GPU:0, CPU:-1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Allow TensorFlow to allocate GPU memory as needed
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# If it fails to determine best cudnn convolution algorithm
# os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"

from _imports import * # Centralized file containing all imports
from _helpers import * # Centralized file containing all helper functions

from keras.config import enable_unsafe_deserialization

enable_unsafe_deserialization()

import sys
import types

def _deserialize_tensor(value):
    if isinstance(value, dict) and value.get("class_name") == "__tensor__":
        config = value.get("config", {})
        dtype = config.get("dtype", "float32")
        try:
            dtype = tf.as_dtype(dtype)
        except TypeError:
            dtype = tf.float32
        return tf.convert_to_tensor(config.get("value"), dtype=dtype)
    return value

for ports in [3, 4, 5, 6, 7, 10, 15]:
    module_name = f"temp_monitor_nas_lm_{ports}_ports_v2.0"
    module = types.ModuleType(module_name)
    module.tf = tf
    module.TOTAL_FEATURES = 100
    sys.modules[module_name] = module

DATA_SEED = 111
TOTAL_FEATURES = 100

# %%
models = {
    3: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_3_ports_v2.0_3_ports/optuna_study/models/top_1_trial_80.keras",
        "batch_size": 256,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_3_ports_v2.0_3_ports/optuna_study/scaler/top_1_trial_80.pkl",
    },
    4: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_4_ports_v2.0_4_ports/optuna_study/models/top_1_trial_304.keras",
        "batch_size": 256,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_4_ports_v2.0_4_ports/optuna_study/scaler/top_1_trial_304.pkl",
    },
    5: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_5_ports_v2.0_5_ports/optuna_study/models/top_1_trial_177.keras",
        "batch_size": 512,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_5_ports_v2.0_5_ports/optuna_study/scaler/top_1_trial_177.pkl",
    },
    6: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_6_ports_v2.0_6_ports/optuna_study/models/top_1_trial_958.keras",
        "batch_size": 256,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_6_ports_v2.0_6_ports/optuna_study/scaler/top_1_trial_958.pkl",
    },
    7: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_7_ports_v2.0_7_ports/optuna_study/models/top_1_trial_835.keras",
        "batch_size": 256,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_7_ports_v2.0_7_ports/optuna_study/scaler/top_1_trial_835.pkl",
    },
    10: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_10_ports_v2.0_10_ports/optuna_study/models/top_1_trial_885.keras",
        "batch_size": 256,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_10_ports_v2.0_10_ports/optuna_study/scaler/top_1_trial_885.pkl",
    },
    15: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_15_ports_v2.0_15_ports/optuna_study/models/top_1_trial_579.keras",
        "batch_size": 256,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/lm/nas_lm_15_ports_v2.0_15_ports/optuna_study/scaler/top_1_trial_579.pkl",
    },
}

THRESHOLD = 0.323
SNR_LINEAR = 1.0

# %%
POLICY = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(POLICY)

BYTES_PER_PARAM = tf.dtypes.as_dtype(POLICY.variable_dtype).size

# %%
# Set to an existing dir to resume training
RUN_DIR = f"runs/{get_caller_stem()}"  # (e.g. "runs/nas_1")

# %% [markdown]
# ## 3. Data Loading and Preprocessing

# %%
# Load simulated channel data and build train/validation splits
kappa0_mu1_m50, kappa5_mu1_m50 = load_data()
kappa0_mu1_m50 = kappa0_mu1_m50[kappa0_mu1_m50.shape[0] // 2 :]
kappa5_mu1_m50 = kappa5_mu1_m50[kappa5_mu1_m50.shape[0] // 2 :]
dataset = np.concatenate((kappa0_mu1_m50, kappa5_mu1_m50), axis=0)

# %% [markdown]
# ## Main

# %%
datasets = {
    "rayleigh": kappa0_mu1_m50,
    "rician": kappa5_mu1_m50,
}

# %%
# Precompute ideal OP once per dataset
ideal_ops_global = [getIdealOP(data, THRESHOLD, SNR_LINEAR) for data in datasets.values()]

for name, ideal_op in zip(datasets.keys(), ideal_ops_global):
    print(f"{name}: IdealOP={ideal_op:.8f}")

ideal_results = {f"{n}_idealOP": o for n, o in zip(datasets.keys(), ideal_ops_global)}
ideal_filepath = os.path.join(RUN_DIR, "ideal_ops.txt")
os.makedirs(RUN_DIR, exist_ok=True)
with open(ideal_filepath, "w") as f:
    for name, ideal_op in ideal_results.items():
        f.write(f"{name}: {ideal_op:.8f}\n")

# %%
dataset_entries = [
    (name, data, ideal_op)
    for (name, data), ideal_op in zip(datasets.items(), ideal_ops_global)
]
results_by_dataset = {name: [] for name in datasets}

for n_ports, model_info in models.items():
    print(f"\n=== {n_ports} ports ===")

    model_path = model_info["model_path"]
    batch_size = model_info["batch_size"]
    scaler_path = model_info["scaler_path"]

    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Lambda):
            globals_dict = getattr(layer.function, "__globals__", {})
            globals_dict.setdefault("tf", tf)
            globals_dict.setdefault("TOTAL_FEATURES", TOTAL_FEATURES)
            # Lambda closures may deserialize tensors as plain dicts; convert them back.
            for name, value in list(globals_dict.items()):
                converted_value = _deserialize_tensor(value)
                if converted_value is not value:
                    globals_dict[name] = converted_value

            if layer.function.__defaults__:
                layer.function.__defaults__ = tuple(
                    _deserialize_tensor(val) for val in layer.function.__defaults__
                )

            if layer.function.__closure__:
                for cell in layer.function.__closure__:
                    converted_value = _deserialize_tensor(cell.cell_contents)
                    if converted_value is not cell.cell_contents:
                        cell.cell_contents = converted_value

    print(f"Loading scaler from {scaler_path}")
    with open(scaler_path, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    per_dataset_results = []

    for name, data, ideal_op in dataset_entries:
        X_test, idxs = get_observed_ports(data, n_ports, TOTAL_FEATURES)

        print(f"Evaluating {name} with shape {X_test.shape}...")

        X_test_scaled = scaler.transform(X_test)
        print("Transformed X_test shape:", X_test_scaled.shape)

        loss = model.evaluate(X_test_scaled, data, batch_size=batch_size, verbose=1)
        loss_value = float(loss[0]) if isinstance(loss, (list, tuple, np.ndarray)) else float(loss)

        y_pred = model.predict(X_test_scaled, verbose=1)
        model_op = getOP(idxs, y_pred, data, THRESHOLD, SNR_LINEAR, TOTAL_FEATURES)
        observed_op = getObservedOP(idxs, data, THRESHOLD, SNR_LINEAR)

        per_dataset_results.append((name, loss_value, model_op, observed_op, ideal_op))
        results_by_dataset[name].append(
            {
                "N": n_ports,
                "loss": loss_value,
                "model_op": float(model_op),
                "observed_op": float(observed_op),
                "ideal_op": float(ideal_op),
            }
        )

    print(f"\n=== {n_ports} observed ports ===")
    for name, loss_value, model_op, observed_op, _ in per_dataset_results:
        print(f"{name}: Loss={loss_value:.6f}, OP={model_op:.6f}, ObsOP={observed_op:.6f}")

    results_filepath = os.path.join(RUN_DIR, f"results_{n_ports}_ports.txt")
    with open(results_filepath, "w") as results_file:
        for name, loss_value, model_op, observed_op, _ in per_dataset_results:
            results_file.write(
                f"{name}: Loss={loss_value:.6f}, Model_OP={model_op:.6f}, Obs_Ports_OP={observed_op:.6f}\n"
            )
    print(f"Results saved to {results_filepath}")


# %%
import re

tex_lines = []
for dataset_name, rows in results_by_dataset.items():
    if not rows:
        continue

    safe_name = re.sub(r'[^A-Za-z0-9]+', '_', dataset_name.lower()).strip('_') or 'dataset'
    sorted_rows = sorted(rows, key=lambda item: item['N'])

    tex_lines.append(r'\pgfplotstableread{')
    tex_lines.append('N   yModel   yObs    yIdeal')
    for row in sorted_rows:
        tex_lines.append(
            f"{row['N']}   {row['model_op']:.6f}   {row['observed_op']:.6f}   {row['ideal_op']:.14f}"
        )
    tex_lines.append(rf'}}\datatable_{safe_name}')
    tex_lines.append('')

tex_path = os.path.join(RUN_DIR, 'results_tables.tex')
with open(tex_path, 'w') as tex_file:
    tex_file.write('\n'.join(tex_lines))

print(f'LaTeX tables saved to {tex_path}')

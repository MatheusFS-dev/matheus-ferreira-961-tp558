import ast
import os
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = REPO_ROOT / "src"
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from keras.config import enable_unsafe_deserialization
from utils.imports_keras import *  # noqa: F401,F403

import spektral.layers.convolutional.cheb_conv as _cheb_conv


enable_unsafe_deserialization()

_original_cheb_call = _cheb_conv.ChebConv.call


def _patched_cheb_call(self, inputs, mask=None):
    if isinstance(mask, (list, tuple)) and mask and mask[0] is None:
        mask = None
    return _original_cheb_call(self, inputs, mask=mask)


_cheb_conv.ChebConv.call = _patched_cheb_call


@tf.keras.utils.register_keras_serializable(package="gnn")
class ScatterObservedPorts(layers.Layer):
    def __init__(self, node_indices, num_nodes, **kwargs):
        super().__init__(**kwargs)
        self.num_nodes = int(num_nodes)
        self.node_indices = [int(i) for i in node_indices]
        self._num_observed = len(self.node_indices)

    def build(self, input_shape):
        expected = input_shape[-1]
        if expected is not None and expected != self._num_observed:
            raise ValueError(f"Expected {self._num_observed} observed features, got {expected}.")
        dtype = tf.dtypes.as_dtype(self.dtype or tf.float32)
        self._scatter_matrix = tf.one_hot(self.node_indices, depth=self.num_nodes, dtype=dtype)
        super().build(input_shape)

    def call(self, inputs):
        inputs = tf.cast(inputs, self._scatter_matrix.dtype)
        full = tf.matmul(inputs, self._scatter_matrix)
        return full[..., tf.newaxis]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_nodes, 1)

    def get_config(self):
        config = super().get_config()
        config.update({"node_indices": self.node_indices, "num_nodes": self.num_nodes})
        return config


SCRIPT_PATHS = {
    "dnn": REPO_ROOT / "src/architectures/loss_minimization_rayleigh_rician/op/op_calc_dnn_quantization.py",
    "cnn1d": REPO_ROOT / "src/architectures/loss_minimization_rayleigh_rician/op/op_calc_cnn1d_quantization.py",
    "gnn": REPO_ROOT / "src/architectures/loss_minimization_rayleigh_rician/op/op_calc_gnn_quantization.py",
    "lm": REPO_ROOT / "src/architectures/loss_minimization_rayleigh_rician/op/op_calc_lm_quantization.py",
    "lstm": REPO_ROOT / "src/architectures/loss_minimization_rayleigh_rician/op/op_calc_lstm_quantization.py",
    "tcnn1d": REPO_ROOT / "src/architectures/loss_minimization_rayleigh_rician/op/op_calc_tcnn1d_quantization.py",
}

QUANTIZATION_BITS = (4, 8, 16, 32)
FLOAT_BITS = 32
TOTAL_FEATURES = 100
OUTPUT_DIR = REPO_ROOT / "runs/quantization_summary"
PLOTS_DIR = OUTPUT_DIR / "plots"


def extract_models_from_script(script_path: Path) -> dict[int, dict[str, object]]:
    source = script_path.read_text()
    module = ast.parse(source, filename=str(script_path))

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "MODELS":
                return ast.literal_eval(node.value)

    raise ValueError(f"Could not find MODELS in {script_path}")


def build_registry() -> dict[str, dict[int, dict[str, object]]]:
    return {
        model_name: extract_models_from_script(script_path)
        for model_name, script_path in SCRIPT_PATHS.items()
    }


def select_embedding_layer_name(model: tf.keras.Model) -> str:
    if len(model.layers) < 2:
        raise ValueError("Model must contain at least one hidden layer before output.")
    return model.layers[-2].name


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


def prepare_lm_environment() -> None:
    for ports in [3, 4, 5, 6, 7, 10, 15]:
        module_name = f"temp_monitor_nas_lm_{ports}_ports_v2.0"
        if module_name in sys.modules:
            continue
        module = types.ModuleType(module_name)
        module.tf = tf
        module.TOTAL_FEATURES = TOTAL_FEATURES
        sys.modules[module_name] = module


def restore_lm_lambda_tensors(model: tf.keras.Model) -> None:
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Lambda):
            continue

        globals_dict = getattr(layer.function, "__globals__", {})
        globals_dict.setdefault("tf", tf)
        globals_dict.setdefault("TOTAL_FEATURES", TOTAL_FEATURES)

        for name, value in list(globals_dict.items()):
            converted_value = _deserialize_tensor(value)
            if converted_value is not value:
                globals_dict[name] = converted_value

        if layer.function.__defaults__:
            layer.function.__defaults__ = tuple(
                _deserialize_tensor(value) for value in layer.function.__defaults__
            )

        if layer.function.__closure__:
            for cell in layer.function.__closure__:
                converted_value = _deserialize_tensor(cell.cell_contents)
                if converted_value is not cell.cell_contents:
                    cell.cell_contents = converted_value


def load_model_for_architecture(model_name: str, model_path: str) -> tf.keras.Model:
    if model_name == "gnn":
        return tf.keras.models.load_model(
            model_path, custom_objects={"ScatterObservedPorts": ScatterObservedPorts}
        )

    if model_name == "lm":
        prepare_lm_environment()
        model = tf.keras.models.load_model(model_path, safe_mode=False)
        restore_lm_lambda_tensors(model)
        return model

    return tf.keras.models.load_model(model_path)


def get_embedding_size_rows() -> list[dict[str, float | int | str]]:
    registry = build_registry()
    rows: list[dict[str, float | int | str]] = []

    for model_name, models in registry.items():
        for n_ports in sorted(models.keys()):
            model_info = models[n_ports]
            model_path = str(model_info["model_path"])
            print(f"[embedding] Loading {model_name} model for {n_ports} ports: {model_path}")

            model = load_model_for_architecture(model_name, model_path)
            embedding_layer_name = select_embedding_layer_name(model)
            embedding_layer = model.get_layer(embedding_layer_name)
            embedding_shape = tuple(embedding_layer.output.shape)
            embedding_dim = int(np.prod([int(dim) for dim in embedding_shape[1:] if dim is not None]))

            row: dict[str, float | int | str] = {
                "model": model_name,
                "observed_ports": int(n_ports),
                "embedding_layer": embedding_layer_name,
                "embedding_dim": embedding_dim,
                "size_no_quant_bits": int(embedding_dim * FLOAT_BITS),
                "size_no_quant_bytes": float((embedding_dim * FLOAT_BITS) / 8.0),
            }

            for bits in QUANTIZATION_BITS:
                row[f"size_q{bits}_bits"] = int(embedding_dim * bits)
                row[f"size_q{bits}_bytes"] = float((embedding_dim * bits) / 8.0)
                row[f"compression_q{bits}x"] = float(FLOAT_BITS / bits)

            rows.append(row)
            tf.keras.backend.clear_session()

    return rows


def build_and_save_tables(rows: list[dict[str, float | int | str]], output_dir: Path) -> tuple[Path, Path, pd.DataFrame]:
    if not rows:
        raise ValueError("No embedding-size rows were generated.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["model", "observed_ports"]).reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)

    csv_path = output_dir / "quantization_embedding_size_table.csv"
    txt_path = output_dir / "quantization_embedding_size_table.txt"

    df.to_csv(csv_path, index=False)
    with open(txt_path, "w") as table_file:
        table_file.write(df.to_string(index=False))
        table_file.write("\n")

    return csv_path, txt_path, df


def plot_embedding_size_curves(df: pd.DataFrame, plots_dir: Path) -> None:
    os.makedirs(plots_dir, exist_ok=True)

    size_columns = [
        ("size_no_quant_bits", "No Quant", "s"),
        ("size_q4_bits", "Q=4", "^"),
        ("size_q8_bits", "Q=8", "v"),
        ("size_q16_bits", "Q=16", "D"),
        ("size_q32_bits", "Q=32", "P"),
    ]

    for model_name, model_df in df.groupby("model"):
        model_df = model_df.sort_values("observed_ports")
        x_ports = model_df["observed_ports"].to_list()

        plt.figure(figsize=(9, 6.5))
        for column_name, label, marker in size_columns:
            plt.plot(
                x_ports,
                model_df[column_name].to_list(),
                marker=marker,
                linewidth=2.0,
                label=label,
            )

        plt.xticks(x_ports)
        plt.grid(which="major", linestyle=":", linewidth=0.8, color="#bbbbbb")
        plt.xlabel("Number of observed ports")
        plt.ylabel("Embedding payload size [bits]")
        plt.title(f"{model_name.upper()} embedding size vs ports")
        plt.legend(loc="best", fontsize="small")
        plt.tight_layout()

        plot_path = plots_dir / f"embedding_size_vs_ports_{model_name}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[embedding] Saved plot: {plot_path}")


def run_summary() -> None:
    rows = get_embedding_size_rows()
    csv_path, txt_path, df = build_and_save_tables(rows, OUTPUT_DIR)
    plot_embedding_size_curves(df, PLOTS_DIR)

    print(f"[embedding] Saved CSV table: {csv_path}")
    print(f"[embedding] Saved TXT table: {txt_path}")
    print(f"[embedding] Total rows: {len(rows)}")


run_summary()

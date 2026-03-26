import os
import re
import sys
from pathlib import Path

# Specify GPU to use (e.g., GPU:0, CPU:-1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Allow TensorFlow to allocate GPU memory as needed
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = next(parent for parent in CURRENT_FILE.parents if (parent / "src").is_dir())
PROJECT_SRC_DIR = REPO_ROOT / "src"
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))


from utils.imports_keras import *  # Centralized file containing all imports
from utils.data import *  # Centralized file containing all helper functions

DATA_SEED = 111
TOTAL_FEATURES = 100
THRESHOLD = 0.323
SNR_LINEAR = 1.0
QUANTIZATION_BITS = (4, 8, 16, 32)

# If this value is None, the script uses the last hidden layer as embedding.
# If this value is a layer name, that exact layer output is quantized.
EMBEDDING_LAYER_NAME = None

MODELS = {
    3: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_3_ports_v2.0_3_ports/optuna_study/models/top_1_trial_607.keras",
        "batch_size": 64,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_3_ports_v2.0_3_ports/optuna_study/scaler/top_1_trial_607.pkl",
    },
    4: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_4_ports_v2.0_4_ports/optuna_study/models/top_1_trial_234.keras",
        "batch_size": 64,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_4_ports_v2.0_4_ports/optuna_study/scaler/top_1_trial_234.pkl",
    },
    5: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_5_ports_v2.0_5_ports/optuna_study/models/top_1_trial_846.keras",
        "batch_size": 64,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_5_ports_v2.0_5_ports/optuna_study/scaler/top_1_trial_846.pkl",
    },
    6: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_6_ports_v2.0_6_ports/optuna_study/models/top_1_trial_753.keras",
        "batch_size": 64,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_6_ports_v2.0_6_ports/optuna_study/scaler/top_1_trial_753.pkl",
    },
    7: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_7_ports_v2.0_7_ports/optuna_study/models/top_1_trial_640.keras",
        "batch_size": 64,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_7_ports_v2.0_7_ports/optuna_study/scaler/top_1_trial_640.pkl",
    },
    10: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_10_ports_v2.0_10_ports/optuna_study/models/top_1_trial_454.keras",
        "batch_size": 64,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_10_ports_v2.0_10_ports/optuna_study/scaler/top_1_trial_454.pkl",
    },
    15: {
        "model_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_15_ports_v2.0_15_ports/optuna_study/models/top_1_trial_412.keras",
        "batch_size": 64,
        "scaler_path": "/media/matheus/SSD-2/matheus/results/fluidra/nas/v2/cnn/nas_cnn1d_15_ports_v2.0_15_ports/optuna_study/scaler/top_1_trial_412.pkl",
    },
}

RUN_DIR = f"runs/{get_caller_stem()}"
PLOTS_DIR = os.path.join(RUN_DIR, "plots")


def preprocess_model_inputs(x_values: np.ndarray) -> np.ndarray:
    """Reshape scaled features for CNN1D model input format.

    Args:
        x_values (np.ndarray): Scaled observed-port features with shape
            ``(n_samples, n_ports)``.

    Returns:
        np.ndarray: Reshaped tensor with shape ``(n_samples, n_ports, 1)``.

    Raises:
        ValueError: If the input tensor does not have rank 2.
    """
    if x_values.ndim != 2:
        raise ValueError(f"Expected rank-2 input before CNN1D reshape, got shape {x_values.shape}")
    return x_values.reshape(x_values.shape[0], x_values.shape[1], 1)


def build_datasets() -> dict[str, np.ndarray]:
    """Load and prepare Rayleigh/Rician datasets for OP evaluation.

    This function mirrors the existing OP scripts to keep reported values
    comparable with prior runs: it loads the raw RR datasets and keeps only
    the second half of each set for evaluation.

    Args:
        None: This helper does not receive runtime parameters.

    Returns:
        dict[str, np.ndarray]: Dictionary with two keys:
            - ``"rayleigh"``: evaluation matrix for the Rayleigh condition.
            - ``"rician"``: evaluation matrix for the Rician condition.

    Raises:
        FileNotFoundError: If one of the expected source files is missing.
        KeyError: If required MATLAB keys are not available in source files.
        OSError: If dataset files cannot be read.

    Examples:
        >>> datasets = build_datasets()
        >>> sorted(datasets.keys())
        ['rayleigh', 'rician']
    """
    kappa0_mu1_m50, kappa5_mu1_m50 = load_raw_rr()

    # Keep the second half exactly as in the existing OP calculations to avoid
    # changing evaluation protocol across scripts.
    kappa0_mu1_m50 = kappa0_mu1_m50[kappa0_mu1_m50.shape[0] // 2 :]
    kappa5_mu1_m50 = kappa5_mu1_m50[kappa5_mu1_m50.shape[0] // 2 :]

    return {
        "rayleigh": kappa0_mu1_m50,
        "rician": kappa5_mu1_m50,
    }


def write_ideal_ops(datasets: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute and save ideal OP values for each channel condition.

    Args:
        datasets (dict[str, np.ndarray]): Mapping from channel name to
            full-target SINR matrix.

    Returns:
        dict[str, float]: Ideal OP by channel name.

    Raises:
        OSError: If output directory or text file cannot be written.
    """
    ideal_ops: dict[str, float] = {}
    for dataset_name, values in datasets.items():
        ideal_ops[dataset_name] = float(getIdealOP(values, THRESHOLD, SNR_LINEAR))

    os.makedirs(RUN_DIR, exist_ok=True)
    ideal_path = os.path.join(RUN_DIR, "ideal_ops.txt")
    with open(ideal_path, "w") as output_file:
        for dataset_name in ("rayleigh", "rician"):
            output_file.write(f"{dataset_name}_idealOP: {ideal_ops[dataset_name]:.8f}\n")

    print(f"Ideal OP values saved to {ideal_path}")
    return ideal_ops


def select_embedding_layer_name(model: tf.keras.Model, layer_name: str | None) -> str:
    """Select which layer output is treated as the quantized embedding.

    Args:
        model (tf.keras.Model): Loaded full DNN model.
        layer_name (str | None): Embedding selection mode.
            If this value is a valid layer name, that exact layer output is
            used as the embedding and quantized before server inference.
            If this value is ``None``, the function automatically chooses the
            last hidden layer (the layer right before model output), which is a
            practical default representation for split inference.

    Returns:
        str: Name of the selected embedding layer.

    Raises:
        ValueError: If ``layer_name`` is provided but not found.
        ValueError: If the model does not have a hidden layer before output.
    """
    if layer_name is not None:
        try:
            model.get_layer(layer_name)
        except ValueError as error:
            raise ValueError(f"Embedding layer '{layer_name}' was not found in the model.") from error
        return layer_name

    if len(model.layers) < 2:
        raise ValueError("Model must contain at least one hidden layer before output.")

    # Use the representation right before final output because it naturally
    # simulates a compact feature tensor being transmitted to the server.
    return model.layers[-2].name


def build_client_server_models(
    model: tf.keras.Model,
    embedding_layer_name: str,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Split a full model into client embedding and server inference stages.

    Args:
        model (tf.keras.Model): Full end-to-end DNN model.
        embedding_layer_name (str): Layer that defines the split point.
            Layers up to and including this layer form the client model.
            Layers after this layer form the server model.

    Returns:
        tuple[tf.keras.Model, tf.keras.Model]:
            - ``embedding_model`` maps observed inputs to embeddings.
            - ``server_model`` maps embeddings to full-port predictions.

    Raises:
        ValueError: If the split point is the last layer or cannot be built.
    """
    embedding_layer = model.get_layer(embedding_layer_name)
    embedding_model = tf.keras.Model(inputs=model.input, outputs=embedding_layer.output)

    reached_split = False

    # Keras 3 layers like Dropout may not expose ``output_shape`` reliably.
    # Build the server input from the symbolic tensor shape instead.
    embedding_output_shape = tuple(embedding_layer.output.shape)
    if len(embedding_output_shape) < 2:
        raise ValueError(
            "Embedding output must include a feature dimension after batch axis. "
            f"Got shape: {embedding_output_shape}"
        )
    server_input_shape = embedding_output_shape[1:]
    server_input = tf.keras.Input(shape=server_input_shape, name="server_embedding_input")
    server_output = server_input

    for layer in model.layers:
        if reached_split:
            server_output = layer(server_output)
        if layer.name == embedding_layer_name:
            reached_split = True

    if not reached_split:
        raise ValueError(f"Split layer '{embedding_layer_name}' was not found while building server model.")

    if server_output is server_input:
        raise ValueError("Split layer cannot be the final model layer.")

    server_model = tf.keras.Model(inputs=server_input, outputs=server_output)
    return embedding_model, server_model


def quantize_embeddings_uniform(
    embeddings: np.ndarray,
    quantization_bits: int,
) -> tuple[np.ndarray, float, float, float]:
    """Apply paper-style uniform quantization to an embedding tensor.

    The quantizer follows the analytical model described in the paper:
    ``Delta = (x_max - x_min) / (2^q - 1)`` with nearest-level rounding and
    reconstruction back to floating point.

    Args:
        embeddings (np.ndarray): Client embedding tensor to be quantized.
        quantization_bits (int): Number of quantization bits ``q``.
            If this value is lower, latency-oriented compression is stronger,
            but quantization noise is larger and representation fidelity drops.
            If this value is higher, quantization levels are denser, noise is
            smaller, and fidelity is better at the cost of larger payload.

    Returns:
        tuple[np.ndarray, float, float, float]:
            ``(dequantized_embeddings, delta, sigma_e2, psnr_db)`` where:
            - ``dequantized_embeddings`` is the reconstructed tensor.
            - ``delta`` is the quantization step size.
            - ``sigma_e2`` is the paper-style noise power ``delta^2 / 12``.
            - ``psnr_db`` is computed with peak power ``x_max^2``.

    Raises:
        ValueError: If ``quantization_bits`` is not strictly positive.
    """
    if quantization_bits <= 0:
        raise ValueError("quantization_bits must be a positive integer.")

    features = embeddings.astype(np.float32, copy=False)
    x_min = float(np.min(features))
    x_max = float(np.max(features))

    levels = (2**quantization_bits) - 1
    if levels < 1 or x_max <= x_min:
        # Degenerate range produces zero quantization noise.
        sigma_e2 = 0.0
        psnr_db = float("inf")
        return features.copy(), 0.0, sigma_e2, psnr_db

    delta = (x_max - x_min) / levels

    # Perform uniform scalar quantization and dequantization exactly as the
    # paper model assumes for analytical PSNR calculations.
    normalized = (features - x_min) / delta
    quantized_levels = np.clip(np.round(normalized), 0, levels)
    dequantized = (quantized_levels * delta) + x_min

    sigma_e2 = (delta**2) / 12.0
    peak_power = max(x_max**2, 1e-12)
    psnr_db = 10.0 * np.log10(peak_power / max(sigma_e2, 1e-12))

    return dequantized.astype(np.float32), float(delta), float(sigma_e2), float(psnr_db)


def quantize_embeddings_uniform_with_range(
    embeddings: np.ndarray,
    quantization_bits: int,
    x_min: float,
    x_max: float,
) -> tuple[np.ndarray, float, float, float]:
    """Quantize one embedding batch using a dataset-level min/max range."""
    if quantization_bits <= 0:
        raise ValueError("quantization_bits must be a positive integer.")

    features = embeddings.astype(np.float32, copy=False)
    levels = (2**quantization_bits) - 1
    if levels < 1 or x_max <= x_min:
        sigma_e2 = 0.0
        psnr_db = float("inf")
        return features.copy(), 0.0, sigma_e2, psnr_db

    delta = (x_max - x_min) / levels
    normalized = (features - x_min) / delta
    quantized_levels = np.clip(np.round(normalized), 0, levels)
    dequantized = (quantized_levels * delta) + x_min

    sigma_e2 = (delta**2) / 12.0
    peak_power = max(x_max**2, 1e-12)
    psnr_db = 10.0 * np.log10(peak_power / max(sigma_e2, 1e-12))

    return dequantized.astype(np.float32), float(delta), float(sigma_e2), float(psnr_db)


def evaluate_dataset_with_quantization(
    *,
    embedding_model: tf.keras.Model,
    server_model: tf.keras.Model,
    scaler: Any,
    n_ports: int,
    dataset_values: np.ndarray,
    batch_size: int,
) -> tuple[float, float, dict[int, float], dict[int, dict[str, float]]]:
    """Evaluate OP without and with multiple quantization bit-widths.

    Args:
        embedding_model (tf.keras.Model): Client-side model that produces
            embeddings from observed ports.
        server_model (tf.keras.Model): Server-side model that maps embeddings
            to full-port predictions.
        scaler (Any): Input scaler compatible with ``transform`` method.
        n_ports (int): Number of observed ports used for this evaluation.
        dataset_values (np.ndarray): Ground-truth full-port values for one
            channel condition.
        batch_size (int): Batch size used during inference.

    Returns:
        tuple[float, float, dict[int, float], dict[int, dict[str, float]]]:
            - OP with no quantization.
            - Observed-only OP baseline.
            - Mapping from quantization bits to OP.
            - Mapping from quantization bits to quantization diagnostics
              (``delta``, ``sigma_e2``, ``psnr_db``).

    Raises:
        ValueError: If model outputs or dataset dimensions are incompatible.
        Exception: Propagates scaler and model inference runtime errors.
    """
    x_test, observed_indices = get_observed_ports(dataset_values, n_ports, TOTAL_FEATURES)
    x_test_scaled = preprocess_model_inputs(scaler.transform(x_test))

    dataset_embedding_min = float("inf")
    dataset_embedding_max = float("-inf")

    for start_idx in range(0, x_test_scaled.shape[0], batch_size):
        end_idx = start_idx + batch_size
        embedding_batch = embedding_model(x_test_scaled[start_idx:end_idx], training=False).numpy()
        dataset_embedding_min = min(dataset_embedding_min, float(np.min(embedding_batch)))
        dataset_embedding_max = max(dataset_embedding_max, float(np.max(embedding_batch)))

    no_quant_batches: list[np.ndarray] = []
    quantized_prediction_batches: dict[int, list[np.ndarray]] = {
        bits: [] for bits in QUANTIZATION_BITS
    }
    quantization_stats: dict[int, dict[str, float]] = {}

    for bits in QUANTIZATION_BITS:
        _, delta, sigma_e2, psnr_db = quantize_embeddings_uniform_with_range(
            np.zeros((1,), dtype=np.float32),
            bits,
            dataset_embedding_min,
            dataset_embedding_max,
        )
        quantization_stats[bits] = {
            "delta": delta,
            "sigma_e2": sigma_e2,
            "psnr_db": psnr_db,
        }

    for start_idx in range(0, x_test_scaled.shape[0], batch_size):
        end_idx = start_idx + batch_size
        embedding_batch = embedding_model(x_test_scaled[start_idx:end_idx], training=False).numpy().astype(
            np.float32,
            copy=False,
        )
        no_quant_batches.append(server_model(embedding_batch, training=False).numpy())

        for bits in QUANTIZATION_BITS:
            dequantized_batch, _, _, _ = quantize_embeddings_uniform_with_range(
                embedding_batch,
                bits,
                dataset_embedding_min,
                dataset_embedding_max,
            )
            quantized_prediction_batches[bits].append(server_model(dequantized_batch, training=False).numpy())

    y_pred_no_quant = np.concatenate(no_quant_batches, axis=0)
    no_quant_op = float(
        getOP(observed_indices, y_pred_no_quant, dataset_values, THRESHOLD, SNR_LINEAR, TOTAL_FEATURES)
    )

    observed_op = float(getObservedOP(observed_indices, dataset_values, THRESHOLD, SNR_LINEAR))

    quantized_ops: dict[int, float] = {}

    for bits in QUANTIZATION_BITS:
        y_pred_quant = np.concatenate(quantized_prediction_batches[bits], axis=0)
        quantized_ops[bits] = float(
            getOP(observed_indices, y_pred_quant, dataset_values, THRESHOLD, SNR_LINEAR, TOTAL_FEATURES)
        )

    return no_quant_op, observed_op, quantized_ops, quantization_stats


def write_port_results(
    n_ports: int,
    per_dataset_rows: list[dict[str, Any]],
) -> None:
    """Write one text report for one observation budget.

    Args:
        n_ports (int): Number of observed ports represented by this report.
        per_dataset_rows (list[dict[str, Any]]): Per-dataset evaluation rows.
            Each row must contain keys:
            ``name``, ``op_no_quant``, ``op_observed``, and ``op_q``.

    Returns:
        None: This helper writes one file and returns nothing.

    Raises:
        OSError: If the report file cannot be written.
    """
    output_path = os.path.join(RUN_DIR, f"results_{n_ports}_ports.txt")
    with open(output_path, "w") as output_file:
        for row in per_dataset_rows:
            q_values_text = " ".join([f"Q{bits}_OP={row['op_q'][bits]:.6f}" for bits in QUANTIZATION_BITS])
            output_file.write(
                f"{row['name']}: "
                f"Model_NoQuant_OP={row['op_no_quant']:.6f}, "
                f"Obs_Ports_OP={row['op_observed']:.6f}, "
                f"{q_values_text}\n"
            )
    print(f"Results saved to {output_path}")


def write_results_tables(results_by_dataset: dict[str, list[dict[str, float]]]) -> None:
    """Write LaTeX-ready data tables for OP vs ports curves.

    Args:
        results_by_dataset (dict[str, list[dict[str, float]]]): Aggregated
            rows grouped by dataset name. Every row must contain:
            ``N``, ``op_no_quant``, ``op_q4``, ``op_q8``, ``op_q16``,
            ``op_q32``, ``op_observed``, and ``op_ideal``.

    Returns:
        None: The function writes a ``results_tables.tex`` file.

    Raises:
        OSError: If the output table file cannot be created or written.
    """
    tex_lines = []
    for dataset_name, rows in results_by_dataset.items():
        if not rows:
            continue

        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", dataset_name.lower()).strip("_") or "dataset"
        sorted_rows = sorted(rows, key=lambda row: row["N"])

        tex_lines.append(r"\pgfplotstableread{")
        tex_lines.append("N   yNoQuant   yQ4   yQ8   yQ16   yQ32   yObs   yIdeal")
        for row in sorted_rows:
            tex_lines.append(
                f"{row['N']}   {row['op_no_quant']:.6f}   {row['op_q4']:.6f}   "
                f"{row['op_q8']:.6f}   {row['op_q16']:.6f}   {row['op_q32']:.6f}   "
                f"{row['op_observed']:.6f}   {row['op_ideal']:.14f}"
            )
        tex_lines.append(rf"}}\datatable_{safe_name}")
        tex_lines.append("")

    table_path = os.path.join(RUN_DIR, "results_tables.tex")
    with open(table_path, "w") as tex_file:
        tex_file.write("\n".join(tex_lines))

    print(f"LaTeX tables saved to {table_path}")


def plot_results(results_by_dataset: dict[str, list[dict[str, float]]]) -> None:
    """Plot OP curves for each channel condition.

    Args:
        results_by_dataset (dict[str, list[dict[str, float]]]): Aggregated OP
            rows grouped by dataset name.

    Returns:
        None: This function writes PNG files into ``PLOTS_DIR``.

    Raises:
        OSError: If the plots directory cannot be created.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    for dataset_name, rows in results_by_dataset.items():
        if not rows:
            continue

        sorted_rows = sorted(rows, key=lambda row: row["N"])
        x_ports = [row["N"] for row in sorted_rows]

        plt.figure(figsize=(9, 6.5))

        plt.plot(x_ports, [row["op_observed"] for row in sorted_rows], marker="o", linewidth=2.2, label="Observed")
        plt.plot(
            x_ports,
            [row["op_no_quant"] for row in sorted_rows],
            marker="s",
            linewidth=2.2,
            label="CNN1D (No Quant)",
        )
        plt.plot(x_ports, [row["op_q4"] for row in sorted_rows], marker="^", linewidth=1.9, label="CNN1D (Q=4)")
        plt.plot(x_ports, [row["op_q8"] for row in sorted_rows], marker="v", linewidth=1.9, label="CNN1D (Q=8)")
        plt.plot(x_ports, [row["op_q16"] for row in sorted_rows], marker="D", linewidth=1.9, label="CNN1D (Q=16)")
        plt.plot(x_ports, [row["op_q32"] for row in sorted_rows], marker="P", linewidth=1.9, label="CNN1D (Q=32)")

        ideal_value = sorted_rows[0]["op_ideal"]
        plt.axhline(ideal_value, linestyle="--", linewidth=2.0, color="#444444", label="Ideal")

        plt.yscale("log")
        plt.xticks(x_ports)
        plt.grid(which="major", linestyle=":", linewidth=0.8, color="#bbbbbb")
        plt.xlabel("Number of observed ports")
        plt.ylabel("Outage probability (OP)")
        plt.title(f"{dataset_name.capitalize()} OP vs ports with embedding quantization")
        plt.legend(loc="best", fontsize="small")
        plt.tight_layout()

        plot_path = os.path.join(PLOTS_DIR, f"op_vs_ports_{dataset_name}_cnn1d_quantization.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Saved plot to {plot_path}")


def run_cnn1d_quantized_op_evaluation() -> None:
    """Run CNN1D OP evaluation with paper-style embedding quantization sweep.

    The pipeline executes these steps:
    1. Load evaluation datasets and ideal OP baselines.
    2. For each observed-port budget, load the corresponding CNN1D model.
    3. Split model into client embedding and server inference submodels.
    4. Evaluate OP for no quantization and for each bit-width in
       ``QUANTIZATION_BITS``.
    5. Save per-budget reports, LaTeX tables, and publication-ready plots.

    Args:
        None: This orchestration helper does not receive runtime parameters.

    Returns:
        None: The function performs evaluation side effects and file outputs.

    Raises:
        FileNotFoundError: If model or scaler files are missing.
        ValueError: If model split or data dimensions are invalid.
        OSError: If output files cannot be created.
        Exception: Propagates inference failures.
    """
    datasets = build_datasets()
    ideal_ops = write_ideal_ops(datasets)

    results_by_dataset: dict[str, list[dict[str, float]]] = {
        dataset_name: [] for dataset_name in datasets.keys()
    }

    for n_ports in sorted(MODELS.keys()):
        model_info = MODELS[n_ports]
        model_path = model_info["model_path"]
        scaler_path = model_info["scaler_path"]
        batch_size = int(model_info["batch_size"])

        print(f"\n=== {n_ports} observed ports ===")
        print(f"Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path)

        embedding_layer_name = select_embedding_layer_name(model, EMBEDDING_LAYER_NAME)
        print(f"Using embedding layer for split quantization: {embedding_layer_name}")

        embedding_model, server_model = build_client_server_models(model, embedding_layer_name)

        print(f"Loading scaler: {scaler_path}")
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)

        per_dataset_rows: list[dict[str, Any]] = []

        for dataset_name, dataset_values in datasets.items():
            op_no_quant, op_observed, op_q, q_stats = evaluate_dataset_with_quantization(
                embedding_model=embedding_model,
                server_model=server_model,
                scaler=scaler,
                n_ports=n_ports,
                dataset_values=dataset_values,
                batch_size=batch_size,
            )

            print(
                f"{dataset_name}: "
                f"NoQ_OP={op_no_quant:.6f}, "
                f"Obs_OP={op_observed:.6f}, "
                f"Q4_OP={op_q[4]:.6f}, "
                f"Q8_OP={op_q[8]:.6f}, "
                f"Q16_OP={op_q[16]:.6f}, "
                f"Q32_OP={op_q[32]:.6f}"
            )
            print(
                f"{dataset_name}: "
                f"PSNR[dB] -> Q4={q_stats[4]['psnr_db']:.2f}, "
                f"Q8={q_stats[8]['psnr_db']:.2f}, "
                f"Q16={q_stats[16]['psnr_db']:.2f}, "
                f"Q32={q_stats[32]['psnr_db']:.2f}"
            )

            per_dataset_rows.append(
                {
                    "name": dataset_name,
                    "op_no_quant": op_no_quant,
                    "op_observed": op_observed,
                    "op_q": op_q,
                }
            )

            results_by_dataset[dataset_name].append(
                {
                    "N": int(n_ports),
                    "op_no_quant": float(op_no_quant),
                    "op_q4": float(op_q[4]),
                    "op_q8": float(op_q[8]),
                    "op_q16": float(op_q[16]),
                    "op_q32": float(op_q[32]),
                    "op_observed": float(op_observed),
                    "op_ideal": float(ideal_ops[dataset_name]),
                }
            )

        write_port_results(n_ports, per_dataset_rows)

    write_results_tables(results_by_dataset)
    plot_results(results_by_dataset)


run_cnn1d_quantized_op_evaluation()

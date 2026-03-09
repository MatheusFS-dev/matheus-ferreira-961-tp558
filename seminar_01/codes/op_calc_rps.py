import os

# Specify GPU to use (e.g., GPU:0, CPU:-1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Allow TensorFlow to allocate GPU memory as needed
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

from _imports import *  # Centralized file containing all imports
from _helpers import *  # Centralized file containing all helper functions

DATA_SEED = 111
TOTAL_FEATURES = 100
OBSERVATION_BUDGETS = [3, 4, 5, 6, 7, 10, 15]
AVG_REPETITIONS_LIST = [20, 200, 2000]
THRESHOLD = 0.323
SNR_LINEAR = 1.0

# Set to an existing dir to resume training
RUN_DIR = f"runs/{get_caller_stem()}"


def build_datasets() -> dict[str, np.ndarray]:
    """Load and prepare channel datasets used by the RPS OP evaluation.

    The function mirrors the data policy used by the OP scripts: it loads
    Rayleigh and Rician sets and keeps only the second half of each dataset
    for evaluation so the resulting slices remain consistent with prior runs.

    Returns:
        dict[str, np.ndarray]:
            Dictionary containing two entries:
            - ``"rayleigh"``: Rayleigh subset with shape ``(n_samples, 100)``.
            - ``"rician"``: Rician subset with shape ``(n_samples, 100)``.

    Raises:
        FileNotFoundError: If any expected source dataset file does not exist.
        KeyError: If the required key is missing inside a loaded MATLAB file.
        OSError: If SciPy fails to read source files from disk.
    """
    kappa0_mu1_m50, kappa5_mu1_m50 = load_data()

    # Keep only the second half to preserve comparability with existing
    # OP calculation scripts and previously reported values.
    kappa0_mu1_m50 = kappa0_mu1_m50[kappa0_mu1_m50.shape[0] // 2 :]
    kappa5_mu1_m50 = kappa5_mu1_m50[kappa5_mu1_m50.shape[0] // 2 :]

    return {
        "rayleigh": kappa0_mu1_m50,
        "rician": kappa5_mu1_m50,
    }


def write_ideal_ops(datasets: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute and persist ideal OP values for each dataset.

    Args:
        datasets (dict[str, np.ndarray]): Channel datasets keyed by channel name.

    Returns:
        dict[str, float]:
            Mapping from channel name to ideal OP value.

    Raises:
        OSError: If the destination directory or output file cannot be created.
    """
    ideal_results = {}
    for name, data in datasets.items():
        ideal_results[name] = float(getIdealOP(data, THRESHOLD, SNR_LINEAR))

    os.makedirs(RUN_DIR, exist_ok=True)
    ideal_filepath = os.path.join(RUN_DIR, "ideal_ops.txt")
    with open(ideal_filepath, "w") as output_file:
        for name in ("rayleigh", "rician"):
            output_file.write(f"{name}_idealOP: {ideal_results[name]:.8f}\n")

    print(f"Ideal OP values saved to {ideal_filepath}")
    return ideal_results


def run_rps_for_budget(
    datasets: dict[str, np.ndarray],
    n_ports: int,
    rng: np.random.Generator,
) -> list[tuple[str, float, dict[int, float]]]:
    """Compute observed-only outage probability for one random-port budget.

    For each channel dataset, the function computes one single-draw OP curve
    and multiple averaged curves, where each average uses a run count from
    ``AVG_REPETITIONS_LIST`` independent random draws.

    Args:
        datasets (dict[str, np.ndarray]): Channel datasets keyed by name.
        n_ports (int): Observation budget (number of randomly selected ports).
        rng (np.random.Generator): Random generator controlling sampled indices.
            If the generator uses a fixed seed, the sampled combinations are
            reproducible. If the seed changes, sampled indices and OP values
            can change accordingly.

    Returns:
        list[tuple[str, float, dict[int, float]]]:
            Per-channel tuples ``(channel_name, single_op, avg_ops)`` where:
            - ``single_op`` comes from exactly one sampled subset of ports.
            - ``avg_ops`` maps each repetition count to its arithmetic mean OP.

    Raises:
        ValueError: If ``n_ports`` is outside ``[1, TOTAL_FEATURES]``.
        ValueError: If any dataset does not have exactly ``TOTAL_FEATURES``
            columns.

    Examples:
        >>> rng = np.random.default_rng(111)
        >>> data = {"rayleigh": np.ones((4, 100)), "rician": np.ones((4, 100))}
        >>> out = run_rps_for_budget(data, 5, rng)
        >>> len(out)
        2
    """
    results = []

    if not AVG_REPETITIONS_LIST or any(rep <= 0 for rep in AVG_REPETITIONS_LIST):
        raise ValueError("AVG_REPETITIONS_LIST must contain positive integers.")

    # Validate budget once so downstream sampling code receives a safe range.
    if n_ports < 1 or n_ports > TOTAL_FEATURES:
        raise ValueError(
            f"n_ports must be in [1, {TOTAL_FEATURES}], got {n_ports}."
        )

    for name, data in datasets.items():
        # Each dataset must match the expected fixed number of available ports
        # so all budgets are directly comparable across channels.
        if data.ndim != 2 or data.shape[1] != TOTAL_FEATURES:
            raise ValueError(
                f"Dataset '{name}' must have shape (n_samples, {TOTAL_FEATURES}). "
                f"Received shape {data.shape}."
            )

        # First curve: use only one random subset so it represents a single
        # realization of the random-port process.
        _, sampled_indices_single = get_random_observed_ports(
            sinr_data=data,
            num_observed_ports=n_ports,
            total_ports=TOTAL_FEATURES,
            rng=rng,
        )
        rps_single_op = float(
            getObservedOP(sampled_indices_single, data, THRESHOLD, SNR_LINEAR)
        )

        # Additional curves: average over multiple run counts. We reuse the
        # largest run budget once, then slice prefixes for lower budgets so all
        # curves are generated from one consistent random sequence.
        max_repetitions = max(AVG_REPETITIONS_LIST)
        repeated_ops = []
        for _ in range(max_repetitions):
            _, sampled_indices = get_random_observed_ports(
                sinr_data=data,
                num_observed_ports=n_ports,
                total_ports=TOTAL_FEATURES,
                rng=rng,
            )
            repeated_ops.append(
                float(getObservedOP(sampled_indices, data, THRESHOLD, SNR_LINEAR))
            )
        avg_ops = {
            repetitions: float(np.mean(repeated_ops[:repetitions]))
            for repetitions in AVG_REPETITIONS_LIST
        }

        avg_parts = [
            f"RPS_avg_{repetitions}={avg_ops[repetitions]:.6f}"
            for repetitions in AVG_REPETITIONS_LIST
        ]

        print(
            f"{name} | N={n_ports} | sampled_single={sampled_indices_single.tolist()} | "
            f"RPS_single={rps_single_op:.6f} | "
            + " | ".join(avg_parts)
        )
        results.append((name, rps_single_op, avg_ops))

    return results


def write_results_file(
    n_ports: int, rps_results: list[tuple[str, float, dict[int, float]]]
) -> None:
    """Write per-budget RPS OP values to a text report.

    Args:
        n_ports (int): Observation budget represented by this file.
        rps_results (list[tuple[str, float, dict[int, float]]]): Per-channel
            RPS values in the form ``(channel_name, single_op, avg_ops)``.

    Returns:
        None: This function writes a file and does not return a value.

    Raises:
        OSError: If the output file cannot be created or written.
    """
    output_path = os.path.join(RUN_DIR, f"results_{n_ports}_ports.txt")
    with open(output_path, "w") as output_file:
        for channel_name, single_op, avg_ops in rps_results:
            avg_text = ", ".join(
                f"RPS_avg_{repetitions}={avg_ops[repetitions]:.6f}"
                for repetitions in AVG_REPETITIONS_LIST
            )
            output_file.write(
                f"{channel_name}: "
                f"RPS_single={single_op:.6f}, "
                f"{avg_text}\n"
            )
    print(f"Results saved to {output_path}")


def write_tex_tables(results_by_dataset: dict[str, list[dict[str, float]]]) -> None:
    """Generate LaTeX-ready tables containing both RPS curves and ideal OP.

    Args:
        results_by_dataset (dict[str, list[dict[str, float]]]): Aggregated rows
            per dataset. Each row must contain ``N``, ``rps_single_op``,
            ``rps_avg_ops``, and ``ideal_op`` keys.

    Returns:
        None: The function writes a ``results_tables.tex`` file.

    Raises:
        OSError: If the output file cannot be created or written.
    """
    import re

    tex_lines = []
    for dataset_name, rows in results_by_dataset.items():
        if not rows:
            continue

        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", dataset_name.lower()).strip("_") or "dataset"
        sorted_rows = sorted(rows, key=lambda item: item["N"])
        avg_headers = "   ".join(f"yRPSAvg{rep}" for rep in AVG_REPETITIONS_LIST)

        tex_lines.append(r"\pgfplotstableread{")
        tex_lines.append(f"N   yRPSSingle   {avg_headers}   yIdeal")
        for row in sorted_rows:
            avg_values = "   ".join(
                f"{row['rps_avg_ops'][rep]:.6f}" for rep in AVG_REPETITIONS_LIST
            )
            tex_lines.append(
                f"{row['N']}   "
                f"{row['rps_single_op']:.6f}   "
                f"{avg_values}   "
                f"{row['ideal_op']:.14f}"
            )
        tex_lines.append(rf"}}\datatable_{safe_name}")
        tex_lines.append("")

    tex_path = os.path.join(RUN_DIR, "results_tables.tex")
    with open(tex_path, "w") as tex_file:
        tex_file.write("\n".join(tex_lines))

    print(f"LaTeX tables saved to {tex_path}")


def run_rps_evaluation() -> None:
    """Run the end-to-end random-port sampling OP evaluation pipeline.

    The pipeline computes ideal OP values, evaluates RPS outage curves for
    predefined observation budgets, and writes text and LaTeX outputs into
    this script's run directory.

    Returns:
        None: This function performs file I/O and printing side effects only.

    Raises:
        OSError: If output files cannot be written.
        ValueError: If the sampled budget or dataset dimensions are invalid.
    """
    datasets = build_datasets()
    ideal_ops = write_ideal_ops(datasets)

    rng = np.random.default_rng(DATA_SEED)
    results_by_dataset = {name: [] for name in datasets}

    for n_ports in OBSERVATION_BUDGETS:
        print(f"\n=== RPS with {n_ports} observed ports ===")
        per_budget_results = run_rps_for_budget(datasets, n_ports, rng)
        write_results_file(n_ports, per_budget_results)

        for channel_name, rps_single_op, rps_avg_ops in per_budget_results:
            results_by_dataset[channel_name].append(
                {
                    "N": n_ports,
                    "rps_single_op": float(rps_single_op),
                    "rps_avg_ops": {
                        repetitions: float(rps_avg_ops[repetitions])
                        for repetitions in AVG_REPETITIONS_LIST
                    },
                    "ideal_op": float(ideal_ops[channel_name]),
                }
            )

    write_tex_tables(results_by_dataset)


run_rps_evaluation()

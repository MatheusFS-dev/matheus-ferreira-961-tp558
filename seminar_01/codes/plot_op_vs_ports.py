#!/usr/bin/env python3
"""Create log-scale OP vs. ports plots with matplotlib for rayleigh and rician."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

os.environ.setdefault("MPLBACKEND", "agg")
from matplotlib import pyplot as plt  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results" / "op"
OUTPUT_DIR = RESULTS_DIR / "plots"
CHANNELS = ("rayleigh", "rician")
RPS_RESULTS_DIR = RESULTS_DIR / "op_calc_rps"
RPS_RUNS_DIR = ROOT_DIR / "runs" / "op_calc_rps"
RELATED_WORK_DIRS = {
    "op_calc_related_work_1",
    "op_calc_related_work_2",
    "op_calc_related_work_3",
}


def iter_models() -> Iterable[Path]:
    if not RESULTS_DIR.exists():
        return ()
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if model_dir.is_dir():
            yield model_dir


def split_model_dirs(model_dirs: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    base_dirs: list[Path] = []
    related_dirs: list[Path] = []
    for model_dir in model_dirs:
        if model_dir.name == "plots":
            continue
        if model_dir.name == "op_calc_rps":
            continue
        if model_dir.name in RELATED_WORK_DIRS:
            related_dirs.append(model_dir)
        else:
            base_dirs.append(model_dir)
    return base_dirs, related_dirs


def load_ideal_values(model_dirs: Iterable[Path]) -> Dict[str, float]:
    """Return the ideal OP values present in any ideal_ops.txt file."""
    ideal_re = re.compile(r"(rayleigh|rician)_idealOP:\s*([0-9.eE+-]+)", re.IGNORECASE)
    ideal: Dict[str, float] = {}
    for model_dir in model_dirs:
        ideal_file = model_dir / "ideal_ops.txt"
        if not ideal_file.exists():
            continue
        for line in ideal_file.read_text().splitlines():
            match = ideal_re.match(line.strip())
            if not match:
                continue
            ideal[match.group(1).lower()] = float(match.group(2))
        if len(ideal) == len(CHANNELS):
            break
    if len(ideal) != len(CHANNELS):
        raise RuntimeError("Ideal OP values not found for all channels.")
    return ideal


def parse_results(model_dirs: Iterable[Path]):
    name_re = re.compile(r"results_(\d+)_ports\.txt$")
    line_model_obs_re = re.compile(
        r"^(rayleigh|rician):.*Model_OP=([0-9.eE+-]+).*Obs_Ports_OP=([0-9.eE+-]+)", re.IGNORECASE
    )
    line_rps_prefix_re = re.compile(r"^(rayleigh|rician):", re.IGNORECASE)
    line_rps_single_re = re.compile(r"RPS_single=([0-9.eE+-]+)")
    line_rps_avg_re = re.compile(r"RPS_avg_(\d+)=([0-9.eE+-]+)")
    line_rps_only_re = re.compile(r"^(rayleigh|rician):.*RPS_OP=([0-9.eE+-]+)", re.IGNORECASE)

    observed: Dict[str, Dict[int, float]] = {channel: {} for channel in CHANNELS}
    rps_single: Dict[str, Dict[int, float]] = {channel: {} for channel in CHANNELS}
    rps_avg_by_repetition: Dict[int, Dict[str, Dict[int, float]]] = {}
    models: Dict[str, Dict[str, Dict[int, float]]] = {}

    for model_dir in model_dirs:
        model_name = model_dir.name
        model_data: Dict[str, Dict[int, float]] = {channel: {} for channel in CHANNELS}
        for file in sorted(model_dir.iterdir()):
            match = name_re.match(file.name)
            if not match:
                continue
            ports = int(match.group(1))
            for line in file.read_text().splitlines():
                line = line.strip()
                match_model_obs = line_model_obs_re.match(line)
                if match_model_obs:
                    channel = match_model_obs.group(1).lower()
                    model_val = float(match_model_obs.group(2))
                    obs_val = float(match_model_obs.group(3))
                    model_data[channel][ports] = model_val
                    prev_obs = observed[channel].get(ports)
                    if prev_obs is not None and not math.isclose(prev_obs, obs_val, rel_tol=1e-8, abs_tol=1e-12):
                        raise RuntimeError(
                            f"Inconsistent Obs_Ports_OP for {channel} at {ports} ports: {prev_obs} vs {obs_val}"
                        )
                    observed[channel][ports] = obs_val
                    continue

                # New RPS format can contain one single-run value and multiple
                # averages (for example 20/200/2000 runs) in the same line.
                match_rps_prefix = line_rps_prefix_re.match(line)
                if match_rps_prefix and "RPS_" in line:
                    channel = match_rps_prefix.group(1).lower()
                    match_rps_single = line_rps_single_re.search(line)
                    if match_rps_single is not None:
                        rps_single_val = float(match_rps_single.group(1))

                        prev_rps_single = rps_single[channel].get(ports)
                        if prev_rps_single is not None and not math.isclose(
                            prev_rps_single, rps_single_val, rel_tol=1e-8, abs_tol=1e-12
                        ):
                            raise RuntimeError(
                                f"Inconsistent RPS_single for {channel} at {ports} ports: "
                                f"{prev_rps_single} vs {rps_single_val}"
                            )
                        rps_single[channel][ports] = rps_single_val

                    avg_matches = line_rps_avg_re.findall(line)
                    for repetitions_str, avg_value_str in avg_matches:
                        repetitions = int(repetitions_str)
                        avg_value = float(avg_value_str)

                        if repetitions not in rps_avg_by_repetition:
                            rps_avg_by_repetition[repetitions] = {
                                c: {} for c in CHANNELS
                            }
                        prev_rps_avg = rps_avg_by_repetition[repetitions][channel].get(ports)
                        if prev_rps_avg is not None and not math.isclose(
                            prev_rps_avg, avg_value, rel_tol=1e-8, abs_tol=1e-12
                        ):
                            raise RuntimeError(
                                f"Inconsistent RPS_avg_{repetitions} for {channel} at {ports} ports: "
                                f"{prev_rps_avg} vs {avg_value}"
                            )
                        rps_avg_by_repetition[repetitions][channel][ports] = avg_value

                    # If the line follows the new format, skip legacy parsing.
                    if match_rps_single is not None or avg_matches:
                        continue

                # Legacy RPS files have a single baseline value, so we map it
                # to both curves to keep old experiments plottable.
                match_rps = line_rps_only_re.match(line)
                if match_rps:
                    channel = match_rps.group(1).lower()
                    rps_val = float(match_rps.group(2))
                    prev_rps_single = rps_single[channel].get(ports)
                    if prev_rps_single is not None and not math.isclose(
                        prev_rps_single, rps_val, rel_tol=1e-8, abs_tol=1e-12
                    ):
                        raise RuntimeError(
                            f"Inconsistent RPS_OP for {channel} at {ports} ports: {prev_rps_single} vs {rps_val}"
                        )
                    rps_single[channel][ports] = rps_val

                    # Legacy files have no explicit averaging count. Store them
                    # as 1-run averages to keep a visible baseline curve.
                    if 1 not in rps_avg_by_repetition:
                        rps_avg_by_repetition[1] = {c: {} for c in CHANNELS}
                    prev_rps_avg = rps_avg_by_repetition[1][channel].get(ports)
                    if prev_rps_avg is not None and not math.isclose(
                        prev_rps_avg, rps_val, rel_tol=1e-8, abs_tol=1e-12
                    ):
                        raise RuntimeError(
                            f"Inconsistent RPS_OP for {channel} at {ports} ports: {prev_rps_avg} vs {rps_val}"
                        )
                    rps_avg_by_repetition[1][channel][ports] = rps_val
        models[model_name] = model_data
    return observed, rps_single, rps_avg_by_repetition, models


def select_best_rps_results():
    """Select and load the most informative available RPS results source.

    The plotting workflow can receive RPS outputs from two common locations:
    ``results/op/op_calc_rps`` (legacy repository output) and
    ``runs/op_calc_rps`` (direct script output). This function parses each
    existing candidate and returns the one with the richest set of RPS curves
    so plotting includes all available run-count averages.

    Returns:
        tuple[Dict[str, Dict[int, float]], Dict[int, Dict[str, Dict[int, float]]], Optional[Path]]:
            A tuple containing:
            - ``rps_single``: per-channel single-run curve values.
            - ``rps_avg_by_repetition``: averaged curves keyed by repetition count.
            - ``selected_dir``: the chosen source directory, or ``None`` if no
              valid RPS directory was found.

    Raises:
        RuntimeError: If a candidate directory exists but has internally
            inconsistent RPS values across files.
    """
    candidates = [RPS_RESULTS_DIR, RPS_RUNS_DIR]
    best_score = -1
    best_dir: Optional[Path] = None
    best_single: Dict[str, Dict[int, float]] = {channel: {} for channel in CHANNELS}
    best_avg: Dict[int, Dict[str, Dict[int, float]]] = {}

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue

        _, rps_single, rps_avg_by_repetition, _ = parse_results([candidate])

        # Prefer sources with more averaging curves first, then more populated
        # points across all channels so we select the most complete dataset.
        repetition_score = len(rps_avg_by_repetition)
        point_score = sum(len(rps_single[channel]) for channel in CHANNELS)
        point_score += sum(
            len(series[channel])
            for series in rps_avg_by_repetition.values()
            for channel in CHANNELS
        )
        score = repetition_score * 1_000_000 + point_score

        if score > best_score:
            best_score = score
            best_dir = candidate
            best_single = rps_single
            best_avg = rps_avg_by_repetition

    return best_single, best_avg, best_dir


def plot_channel(
    channel: str,
    observed: Dict[str, Dict[int, float]],
    rps_single: Dict[str, Dict[int, float]],
    rps_avg_by_repetition: Dict[int, Dict[str, Dict[int, float]]],
    models: Dict[str, Dict[str, Dict[int, float]]],
    ideal_value: float,
    include_rps_curves: bool,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    obs_series = observed.get(channel, {})
    if not obs_series:
        raise RuntimeError(f"No observed OP data for {channel}.")
    ports = sorted(obs_series)
    obs_values = [obs_series[p] for p in ports]
    ax.plot(
        ports,
        obs_values,
        label="Observed",
        color="#1f1f1f",
        linestyle="-",
        linewidth=2.5,
        marker="o",
        markersize=6,
    )

    # The RPS baseline is optional and only plotted when RPS results are
    # available in the dedicated results directory.
    if include_rps_curves:
        # The single-run RPS curve is intentionally omitted per plotting
        # requirements; only averaged multi-run RPS curves are displayed.
        avg_repetitions = [rep for rep in sorted(rps_avg_by_repetition) if rep > 1]
        avg_color_count = max(1, len(avg_repetitions))
        for idx, repetitions in enumerate(avg_repetitions):
            rps_avg_series = rps_avg_by_repetition[repetitions].get(channel, {})
            if not rps_avg_series:
                continue
            rps_ports = sorted(rps_avg_series)
            rps_values = [rps_avg_series[p] for p in rps_ports]
            color_position = 0.45 + (0.45 * idx / max(1, avg_color_count - 1))
            ax.plot(
                rps_ports,
                rps_values,
                label=f"RPS (avg {repetitions} runs)",
                color=plt.cm.plasma(color_position),
                linestyle="--",
                linewidth=2.1,
                marker="d",
                markersize=5,
            )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (model_name, channel_data) in enumerate(sorted(models.items())):
        data = channel_data.get(channel, {})
        if not data:
            continue
        model_ports = sorted(data)
        model_values = [data[p] for p in model_ports]
        ax.plot(
            model_ports,
            model_values,
            label=f"{model_name}",
            color=colors[idx % len(colors)],
            linewidth=1.8,
            marker="s",
            markersize=5,
            alpha=0.85,
        )

    ax.axhline(
        ideal_value,
        color="#444444",
        linestyle="--",
        linewidth=2,
        label="Ideal",
    )

    ax.set_yscale("log")
    ax.grid(which="major", linestyle=":", linewidth=0.8, color="#bbbbbb")
    ax.set_title(f"{channel.capitalize()} outage probability", pad=16)
    ax.set_xlabel("Number of ports")
    ax.set_ylabel("Outage probability (OP)")
    ports_ticks = {p for data in models.values() for p in data.get(channel, {})}
    ports_ticks.update(obs_series.keys())
    for repetitions in rps_avg_by_repetition:
        if repetitions <= 1:
            continue
        ports_ticks.update(rps_avg_by_repetition[repetitions].get(channel, {}).keys())
    ax.set_xticks(sorted(ports_ticks))
    ax.legend(fontsize="small", loc="best", ncol=1)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"op_vs_ports_{channel}.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def plot_models_only(
    channel: str,
    models: Dict[str, Dict[str, Dict[int, float]]],
    *,
    title: Optional[str] = None,
    filename: Optional[str] = None,
    port_range: Optional[tuple[int, int]] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ports_ticks = set()
    plotted_any = False
    for idx, (model_name, channel_data) in enumerate(sorted(models.items())):
        data = channel_data.get(channel, {})
        if not data:
            continue
        model_ports = sorted(data)
        if port_range is not None:
            min_port, max_port = port_range
            model_ports = [p for p in model_ports if min_port <= p <= max_port]
            if not model_ports:
                continue
        plotted_any = True
        ports_ticks.update(model_ports)
        model_values = [data[p] for p in model_ports]
        ax.plot(
            model_ports,
            model_values,
            label=f"{model_name}",
            color=colors[idx % len(colors)],
            linewidth=1.8,
            marker="s",
            markersize=5,
            alpha=0.9,
        )

    if not plotted_any:
        raise RuntimeError(f"No model OP data for {channel}.")

    ax.set_yscale("log")
    ax.grid(which="major", linestyle=":", linewidth=0.8, color="#bbbbbb")
    if title is None:
        title = f"{channel.capitalize()} model outage probability"
    ax.set_title(title, pad=16)
    ax.set_xlabel("Number of ports")
    ax.set_ylabel("Model outage probability (OP)")
    if port_range is not None:
        min_port, max_port = port_range
        ax.set_xlim(min_port - 0.2, max_port + 0.2)
        ax.set_xticks(list(range(min_port, max_port + 1)))
    else:
        ax.set_xticks(sorted(ports_ticks))
    ax.legend(fontsize="small", loc="best", ncol=1)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"op_vs_ports_models_{channel}.png"
    out_file = OUTPUT_DIR / filename
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    return out_file


def main():
    all_model_dirs = list(iter_models())
    if not all_model_dirs:
        raise SystemExit("No model directories found under results/op.")
    base_model_dirs, related_model_dirs = split_model_dirs(all_model_dirs)
    if not base_model_dirs:
        base_model_dirs = [d for d in all_model_dirs if d.name not in {"plots", "op_calc_rps"}]
    ideal_values = load_ideal_values(base_model_dirs)
    observed, _, _, models = parse_results(base_model_dirs)

    rps_single, rps_avg_by_repetition, selected_rps_dir = select_best_rps_results()
    include_rps_curves = any(
        any(bool(series[channel]) for channel in CHANNELS)
        for rep, series in rps_avg_by_repetition.items()
        if rep > 1
    )
    if selected_rps_dir is not None:
        print(f"Using RPS source: {selected_rps_dir}")

    saved_files = []
    for channel in CHANNELS:
        saved_files.append(
            plot_channel(
                channel,
                observed,
                rps_single,
                rps_avg_by_repetition,
                models,
                ideal_values[channel],
                include_rps_curves=include_rps_curves,
            )
        )
    for channel in CHANNELS:
        saved_files.append(plot_models_only(channel, models))
    if related_model_dirs:
        combined_dirs = base_model_dirs + related_model_dirs
        _, _, _, models_with_related = parse_results(combined_dirs)
        for channel in CHANNELS:
            saved_files.append(
                plot_models_only(
                    channel,
                    models_with_related,
                    title=f"{channel.capitalize()} model + related work outage probability",
                    filename=f"op_vs_ports_models_related_work_{channel}.png",
                )
            )
        saved_files.append(
            plot_models_only(
                "rayleigh",
                models_with_related,
                title="Rayleigh model + related work outage probability (5-7 ports)",
                filename="op_vs_ports_models_related_work_rayleigh_5_7_ports.png",
                port_range=(5, 7),
            )
        )
    for saved in saved_files:
        print(f"Saved {saved}")


if __name__ == "__main__":
    main()

"""Run and score the ClearSky-AI Sentinel model against prepared demo samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import statistics

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import uniform_filter

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from infer_clearsky_sentinel import choose_device, run_inference
from clearsky_sentinel import ClearSkySentinel, load_clearsky_weights


def box_ssim(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray, window: int = 11) -> float:
    """Return masked, windowed SSIM averaged over the 13 spectral bands."""
    if window < 3 or window % 2 == 0:
        raise ValueError("SSIM window must be an odd integer >= 3.")
    scores = []
    valid_float = valid.astype(np.float64)
    for band in range(prediction.shape[0]):
        pred = prediction[band].astype(np.float64) / 10000.0
        ref = reference[band].astype(np.float64) / 10000.0
        pred_valid = pred * valid_float
        ref_valid = ref * valid_float

        mu_pred = uniform_filter(pred_valid, size=window)
        mu_ref = uniform_filter(ref_valid, size=window)
        mu_pred = mu_pred / np.maximum(uniform_filter(valid_float, size=window), 1e-12)
        mu_ref = mu_ref / np.maximum(uniform_filter(valid_float, size=window), 1e-12)
        pred_sq = uniform_filter(pred_valid * pred_valid, size=window)
        ref_sq = uniform_filter(ref_valid * ref_valid, size=window)
        pred_ref = uniform_filter(pred_valid * ref_valid, size=window)
        weight = uniform_filter(valid_float, size=window)
        var_pred = np.maximum(pred_sq / weight - mu_pred * mu_pred, 0.0)
        var_ref = np.maximum(ref_sq / weight - mu_ref * mu_ref, 0.0)
        cov = pred_ref / weight - mu_pred * mu_ref

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        map_ssim = (
            (2 * mu_pred * mu_ref + c1) * (2 * cov + c2)
            / ((mu_pred ** 2 + mu_ref ** 2 + c1) * (var_pred + var_ref + c2))
        )
        scores.append(float(map_ssim[valid].mean()))
    return float(np.mean(scores))


def spectral_angle(prediction: np.ndarray, reference: np.ndarray, valid: np.ndarray) -> float:
    pred = prediction[:, valid].astype(np.float64).T
    ref = reference[:, valid].astype(np.float64).T
    dot = np.sum(pred * ref, axis=1)
    norms = np.linalg.norm(pred, axis=1) * np.linalg.norm(ref, axis=1)
    valid_spectrum = norms > 1e-12
    if not valid_spectrum.any():
        return float("nan")
    cosine = np.clip(dot[valid_spectrum] / norms[valid_spectrum], -1.0, 1.0)
    return float(np.degrees(np.mean(np.arccos(cosine))))


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(prediction_path: Path, reference_path: Path, ssim_window: int = 11) -> dict:
    with rasterio.open(prediction_path) as prediction, rasterio.open(reference_path) as reference:
        if (
            prediction.count != 13
            or reference.count != 13
            or prediction.width != reference.width
            or prediction.height != reference.height
            or prediction.crs != reference.crs
            or prediction.transform != reference.transform
        ):
            raise ValueError("Prediction and clear reference are not on the same 13-band grid.")
        estimate = prediction.read().astype(np.float32)
        target = reference.read().astype(np.float32)
        valid = (prediction.dataset_mask() > 0) & (reference.dataset_mask() > 0)
    if not valid.any():
        raise ValueError("No valid reference pixels are available for scoring.")
    difference = (estimate - target)[:, valid]
    mae = float(np.abs(difference).mean()) / 10000.0
    rmse = float(np.sqrt(np.square(difference).mean())) / 10000.0
    psnr = float("inf") if rmse == 0 else float(20.0 * np.log10(1.0 / rmse))
    ssim = box_ssim(estimate, target, valid, window=ssim_window)
    sam = spectral_angle(estimate, target, valid)
    return {
        "MAE_0_1": mae,
        "RMSE_0_1": rmse,
        "PSNR_dB": psnr,
        "SSIM": ssim,
        "SAM_degrees": sam,
        "valid_pixels": int(valid.sum()),
    }


def write_benchmark_plots(results: list[dict], output_dir: Path) -> list[Path]:
    """Create a compact visual benchmark report for held-out test patches."""
    test_results = [row for row in results if row["split"].lower() == "test"]
    if not test_results:
        return []

    plt.style.use("seaborn-v0_8-darkgrid")
    psnr_gain = np.array([row["PSNR_gain_dB"] for row in test_results], dtype=np.float64)
    cloudy_psnr = np.array([row["input_PSNR_dB"] for row in test_results], dtype=np.float64)
    output_psnr = np.array([row["PSNR_dB"] for row in test_results], dtype=np.float64)
    cloudy_mae = np.array([row["input_MAE_0_1"] for row in test_results], dtype=np.float64)
    output_mae = np.array([row["MAE_0_1"] for row in test_results], dtype=np.float64)
    cloudy_ssim = np.array([row["input_SSIM"] for row in test_results], dtype=np.float64)
    output_ssim = np.array([row["SSIM"] for row in test_results], dtype=np.float64)

    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].hist(psnr_gain, bins=30, color="#2563eb", edgecolor="white", alpha=0.9)
    axes[0].axvline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    axes[0].axvline(float(psnr_gain.mean()), color="#16a34a", linestyle="-", linewidth=2)
    axes[0].set_title("Held-out PSNR Improvement")
    axes[0].set_xlabel("PSNR gain (dB)")
    axes[0].set_ylabel("Patch count")

    axes[1].scatter(cloudy_psnr, output_psnr, s=18, alpha=0.55, color="#7c3aed")
    limit = max(float(cloudy_psnr.max()), float(output_psnr.max())) + 1
    axes[1].plot([0, limit], [0, limit], "--", color="#64748b", linewidth=1)
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Cloudy vs Reconstruction PSNR")
    axes[1].set_xlabel("Cloudy PSNR (dB)")
    axes[1].set_ylabel("Reconstructed PSNR (dB)")
    psnr_path = output_dir / "psnr_benchmark.png"
    fig.savefig(psnr_path, dpi=180)
    plt.close(fig)
    paths.append(psnr_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    labels = ["Cloudy", "Reconstruction"]
    axes[0].boxplot([cloudy_mae, output_mae], tick_labels=labels, patch_artist=True)
    axes[0].set_title("MAE (lower is better)")
    axes[0].set_ylabel("Normalized MAE")

    axes[1].boxplot([cloudy_ssim, output_ssim], tick_labels=labels, patch_artist=True)
    axes[1].set_title("SSIM (higher is better)")
    axes[1].set_ylabel("SSIM")
    quality_path = output_dir / "quality_benchmark.png"
    fig.savefig(quality_path, dpi=180)
    plt.close(fig)
    paths.append(quality_path)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sorted_gain = np.sort(psnr_gain)
    fraction_at_least = 1.0 - np.arange(len(sorted_gain)) / len(sorted_gain)
    ax.plot(sorted_gain, fraction_at_least * 100, color="#2563eb", linewidth=2)
    ax.axvline(0, color="#dc2626", linestyle="--", linewidth=1.5)
    ax.set_xlim(left=float(sorted_gain.min()) - 1)
    ax.set_ylim(0, 100)
    ax.set_title("Cumulative Held-out PSNR Gain")
    ax.set_xlabel("At least this PSNR gain (dB)")
    ax.set_ylabel("Patches (%)")
    cumulative_path = output_dir / "cumulative_psnr_gain.png"
    fig.savefig(cumulative_path, dpi=180)
    plt.close(fig)
    paths.append(cumulative_path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_ROOT / "data" / "sentinel_demo" / "manifest.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "eval" / "sentinel_demo"
    )
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "weights" / "clearskkyai.pth")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ssim-window", type=int, default=11)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be zero or a positive integer.")
    if args.start_index < 0:
        raise ValueError("--start-index must be zero or a positive integer.")
    if args.count < 0:
        raise ValueError("--count must be zero or a positive integer.")
    if args.ssim_window < 3 or args.ssim_window % 2 == 0:
        raise ValueError("--ssim-window must be an odd integer >= 3.")
    if not args.manifest.is_file():
        raise FileNotFoundError(
            f"Gallery manifest not found: {args.manifest}. Prepare samples first with "
            "scripts/prepare_sen12mscr_demo.py."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with args.manifest.open(newline="", encoding="utf-8") as stream:
        cases = list(csv.DictReader(stream))
    if args.split != "all":
        cases = [case for case in cases if case["split"].lower() == args.split]
    if args.limit:
        cases = cases[: args.limit]
    if args.start_index:
        cases = cases[args.start_index :]
    if args.count:
        cases = cases[: args.count]
    if not cases:
        raise ValueError("Gallery manifest has no cases.")

    device = choose_device(args.device)
    model = ClearSkySentinel().to(device)
    load_clearsky_weights(model, str(args.checkpoint), map_location=device)
    model.eval()

    for index, case in enumerate(cases, start=1):
        sample_root = args.manifest.parent
        cloudy = sample_root / case["cloudy_s2"]
        sar = sample_root / case["sar_s1"]
        target = sample_root / case["clear_target"]
        output = args.output_dir / f"{case['sample_id']}_estimate.tif"
        print(f"[{index}/{len(cases)}] Inferring {case['sample_id']} ({case['split']} split)", flush=True)
        run_inference(
            cloudy,
            sar,
            args.checkpoint,
            output,
            tile=256,
            overlap=32,
            device_name=args.device,
            model=model,
        )
        row = {
            key: case[key]
            for key in ("sample_id", "split", "season", "scene", "patch", "note")
        }
        input_metrics = score(cloudy, target, ssim_window=args.ssim_window)
        output_metrics = score(output, target, ssim_window=args.ssim_window)
        row.update(output_metrics)
        row.update({f"input_{key}": value for key, value in input_metrics.items()})
        row["PSNR_gain_dB"] = output_metrics["PSNR_dB"] - input_metrics["PSNR_dB"]
        row["MAE_gain_0_1"] = input_metrics["MAE_0_1"] - output_metrics["MAE_0_1"]
        row["prediction"] = str(output)
        results.append(row)
        print(
            f"  input PSNR={input_metrics['PSNR_dB']:.2f} dB -> "
            f"output PSNR={output_metrics['PSNR_dB']:.2f} dB "
            f"({row['PSNR_gain_dB']:+.2f} dB); "
            f"output MAE={output_metrics['MAE_0_1']:.5f}; reference-based patch metrics",
            flush=True,
        )

    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote per-case results to {csv_path}")

    def summarize(rows: list[dict], label: str) -> dict:
        finite_psnr_gains = [
            row["PSNR_gain_dB"] for row in rows if math.isfinite(row["PSNR_gain_dB"])
        ]
        return {
            "label": label,
            "patches": len(rows),
            "cloudy": {
                "MAE_0_1": statistics.mean(row["input_MAE_0_1"] for row in rows),
                "RMSE_0_1": statistics.mean(row["input_RMSE_0_1"] for row in rows),
                "PSNR_dB": statistics.mean(row["input_PSNR_dB"] for row in rows),
                "SSIM": statistics.mean(row["input_SSIM"] for row in rows),
                "SAM_degrees": statistics.mean(row["input_SAM_degrees"] for row in rows),
            },
            "output": {
                "MAE_0_1": statistics.mean(row["MAE_0_1"] for row in rows),
                "RMSE_0_1": statistics.mean(row["RMSE_0_1"] for row in rows),
                "PSNR_dB": statistics.mean(row["PSNR_dB"] for row in rows),
                "SSIM": statistics.mean(row["SSIM"] for row in rows),
                "SAM_degrees": statistics.mean(row["SAM_degrees"] for row in rows),
            },
            "mean_PSNR_gain_dB": (
                statistics.mean(finite_psnr_gains) if finite_psnr_gains else float("nan")
            ),
            "patches_with_positive_PSNR_gain": sum(
                row["PSNR_gain_dB"] > 0 for row in rows
            ),
        }

    splits = {
        "all": summarize(results, "all"),
    }
    test_results = [row for row in results if row["split"].lower() == "test"]
    train_results = [row for row in results if row["split"].lower() == "train"]
    if test_results:
        splits["test"] = summarize(test_results, "held-out test")
        test_summary = splits["test"]
        cloudy_psnr = test_summary["cloudy"]["PSNR_dB"]
        output_psnr = test_summary["output"]["PSNR_dB"]
        psnr_gain = test_summary["mean_PSNR_gain_dB"]
        improved = test_summary["patches_with_positive_PSNR_gain"]
        print(
            f"Held-out per-patch mean PSNR: {cloudy_psnr:.2f} dB cloudy -> "
            f"{output_psnr:.2f} dB output; mean gain {psnr_gain:+.2f} dB "
            f"({improved}/{len(test_results)} patches improved)."
        )
    if train_results:
        splits["train"] = summarize(train_results, "training split")

    summary_path = args.output_dir / "summary.json"
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": (
            checkpoint_sha256(args.checkpoint) if args.checkpoint.is_file() else None
        ),
        "manifest": str(args.manifest.resolve()),
        "training_sources": [
            "SEN12MS-CR",
            "AllClear",
        ],
        "training_source_reported_size_gb": 90,
        "diffcr_role": "Source repository/reference implementation",
        "prepared_training_examples": 96,
        "validation_split_recorded": False,
        "held_out_test_patches": len(test_results),
        "split_filter": args.split,
        "limit": args.limit,
        "ssim_window": args.ssim_window,
        "metrics": "MAE/RMSE/PSNR normalized over 13 bands; SSIM mean per-band; SAM in degrees",
        "splits": splits,
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(f"Wrote aggregate results to {summary_path}")
    plot_paths = write_benchmark_plots(results, args.output_dir)
    for plot_path in plot_paths:
        print(f"Wrote benchmark graph: {plot_path}")

    if test_results:
        test_summary = splits["test"]
        report_path = args.output_dir / "benchmark_report.md"
        report = [
            "# ClearSky-AI Sentinel Held-out Benchmark",
            "",
            f"- Patches: {test_summary['patches']}",
            f"- Checkpoint SHA-256: `{summary['checkpoint_sha256']}`",
            f"- Cloudy mean PSNR: {test_summary['cloudy']['PSNR_dB']:.2f} dB",
            f"- Reconstruction mean PSNR: {test_summary['output']['PSNR_dB']:.2f} dB",
            f"- Mean PSNR gain: {test_summary['mean_PSNR_gain_dB']:+.2f} dB",
            f"- Patches with positive PSNR gain: {test_summary['patches_with_positive_PSNR_gain']}/{test_summary['patches']}",
            f"- Cloudy MAE: {test_summary['cloudy']['MAE_0_1']:.5f}",
            f"- Reconstruction MAE: {test_summary['output']['MAE_0_1']:.5f}",
            f"- Cloudy SSIM: {test_summary['cloudy']['SSIM']:.5f}",
            f"- Reconstruction SSIM: {test_summary['output']['SSIM']:.5f}",
            f"- Cloudy SAM: {test_summary['cloudy']['SAM_degrees']:.3f} degrees",
            f"- Reconstruction SAM: {test_summary['output']['SAM_degrees']:.3f} degrees",
            "",
            "## PSNR benchmark",
            "",
            "![PSNR benchmark](psnr_benchmark.png)",
            "",
            "## MAE and SSIM benchmark",
            "",
            "![Quality benchmark](quality_benchmark.png)",
            "",
            "## Cumulative PSNR gain",
            "",
            "![Cumulative PSNR gain](cumulative_psnr_gain.png)",
            "",
            "## Interpretation",
            "",
            "These are paired-reference scores from SEN12MS-CR test patches. "
            "Clear references can be from a different acquisition date. "
            "The benchmark is not an independent external validation.",
        ]
        report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"Wrote benchmark report: {report_path}")
    print("Train-split scores are demonstrations, not held-out performance.")
    print("Test-split examples are from SEN12MS-CR, not an independent external dataset.")


if __name__ == "__main__":
    main()

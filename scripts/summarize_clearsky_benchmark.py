"""Merge chunked ClearSky-AI benchmark metrics into a full report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate_sen12mscr_demo import write_benchmark_plots


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunk-dir",
        action="append",
        type=Path,
        required=True,
        help="Chunk evaluation directory; repeat once per chunk.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "sentinel_demo",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "weights" / "clearskkyai.pth",
    )
    args = parser.parse_args()

    rows = []
    for chunk_dir in args.chunk_dir:
        metrics_path = chunk_dir / "metrics.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing chunk metrics: {metrics_path}")
        with metrics_path.open("newline", encoding="utf-8") as stream:
            rows.extend(list(csv.DictReader(stream)))
    if not rows:
        raise ValueError("No benchmark rows found.")
    for row in rows:
        for key in (
            "MAE_0_1", "RMSE_0_1", "PSNR_dB", "SSIM", "SAM_degrees",
            "input_MAE_0_1", "input_RMSE_0_1", "input_PSNR_dB",
            "input_SSIM", "input_SAM_degrees", "PSNR_gain_dB",
            "MAE_gain_0_1", "valid_pixels", "input_valid_pixels",
        ):
            row[key] = float(row[key])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.output_dir / "metrics.csv"
    with merged_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    gains = [row["PSNR_gain_dB"] for row in rows]
    test_summary = {
        "label": "held-out test",
        "patches": len(rows),
        "cloudy": {
            "MAE_0_1": mean(rows, "input_MAE_0_1"),
            "RMSE_0_1": mean(rows, "input_RMSE_0_1"),
            "PSNR_dB": mean(rows, "input_PSNR_dB"),
            "SSIM": mean(rows, "input_SSIM"),
            "SAM_degrees": mean(rows, "input_SAM_degrees"),
        },
        "output": {
            "MAE_0_1": mean(rows, "MAE_0_1"),
            "RMSE_0_1": mean(rows, "RMSE_0_1"),
            "PSNR_dB": mean(rows, "PSNR_dB"),
            "SSIM": mean(rows, "SSIM"),
            "SAM_degrees": mean(rows, "SAM_degrees"),
        },
        "mean_PSNR_gain_dB": statistics.mean(gains),
        "patches_with_positive_PSNR_gain": sum(gain > 0 for gain in gains),
    }
    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint.is_file() else None,
        "manifest": str((PROJECT_ROOT / "data" / "sentinel_demo" / "manifest.csv").resolve()),
        "split_filter": "test",
        "patches": len(rows),
        "ssim_window": 11,
        "metrics": "MAE/RMSE/PSNR normalized over 13 bands; SSIM mean per-band; SAM in degrees",
        "splits": {"all": test_summary, "test": test_summary},
    }
    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    plot_paths = write_benchmark_plots(rows, args.output_dir)
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
        "These are paired-reference scores from SEN12MS-CR held-out test patches.",
        "Clear references can be from a different acquisition date.",
        "The benchmark is not an independent external validation.",
    ]
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {merged_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    for plot_path in plot_paths:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()

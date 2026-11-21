"""ClearSky-AI demo for SAR-guided Sentinel-2 cloud reconstruction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import rasterio
import streamlit as st
from rasterio.enums import Resampling


PROJECT_ROOT = Path(__file__).resolve().parent
INFERENCE_SCRIPT = PROJECT_ROOT / "scripts" / "infer_clearsky_sentinel.py"
GALLERY_MANIFEST = PROJECT_ROOT / "data" / "sentinel_demo" / "manifest.csv"
EVAL_SUMMARY = PROJECT_ROOT / "data" / "eval" / "sentinel_demo" / "summary.json"
MAX_PREVIEW_EDGE = 900
REFLECTANCE_MAX = 10000.0
ERROR_PREVIEW_SCALE = 48.0
GALLERY_LFS_PENDING = False


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb") as stream:
            return stream.read(128).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def checkpoint_ready(path: Path) -> bool:
    return path.is_file() and not is_lfs_pointer(path)


def read_gallery() -> list[dict]:
    global GALLERY_LFS_PENDING
    GALLERY_LFS_PENDING = False
    if not GALLERY_MANIFEST.is_file():
        return []
    with GALLERY_MANIFEST.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("cloudy_s2", "sar_s1", "clear_target"):
            row[key] = (GALLERY_MANIFEST.parent / row[key]).resolve()
    usable_rows = []
    for row in rows:
        paths = (row["cloudy_s2"], row["sar_s1"], row["clear_target"])
        if any(is_lfs_pointer(path) for path in paths):
            GALLERY_LFS_PENDING = True
            continue
        if any(not path.is_file() for path in paths):
            continue
        usable_rows.append(row)
    rows = usable_rows
    rows.sort(
        key=lambda row: (
            0 if row.get("cloud_coverage_pct") else 1,
            -float(row["cloud_coverage_pct"]) if row.get("cloud_coverage_pct") else 0.0,
            0 if row["split"].lower() == "test" else 1,
            row["season"],
            row["scene"],
            row["patch"],
        )
    )
    return rows


def read_evaluation_summary() -> dict | None:
    if not EVAL_SUMMARY.is_file():
        return None
    try:
        with EVAL_SUMMARY.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError):
        return None


def stage_upload(uploaded_file, destination: Path) -> Path:
    destination.write_bytes(uploaded_file.getvalue())
    return destination


def _preview_shape(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, MAX_PREVIEW_EDGE / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def read_true_color(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        if source.count < 4:
            raise ValueError(f"Expected at least four optical bands in {path.name}.")
        width, height = _preview_shape(source.width, source.height)
        # True-color order: B04 red, B03 green, B02 blue.
        return source.read(
            indexes=(4, 3, 2),
            out_shape=(3, height, width),
            resampling=Resampling.average,
            masked=True,
        ).filled(0).astype(np.float32)


def read_sar_preview(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        if source.count != 2:
            raise ValueError("Sentinel-1 input must contain VV then VH.")
        width, height = _preview_shape(source.width, source.height)
        vv = source.read(1, out_shape=(height, width), resampling=Resampling.average).astype(np.float32)
    gray = np.clip((vv + 25.0) / 25.0, 0.0, 1.0)
    return np.rint(gray[..., None] * 255).astype(np.uint8)


def make_shared_previews(
    *images: np.ndarray, stretch_indices: tuple[int, ...] | None = None
) -> list[np.ndarray]:
    finite_images = [np.nan_to_num(x, nan=0.0, posinf=REFLECTANCE_MAX, neginf=0.0) for x in images]
    stretch_images = (
        [finite_images[index] for index in stretch_indices]
        if stretch_indices is not None
        else finite_images
    )
    combined = np.concatenate([image.reshape(3, -1) for image in stretch_images], axis=1)
    low, high = np.percentile(combined, (2, 98), axis=1)
    high = np.maximum(high, low + 1.0)
    output = []
    for image in finite_images:
        stretched = np.clip(
            (image - low[:, None, None]) / (high - low)[:, None, None], 0.0, 1.0
        )
        output.append(np.rint(np.moveaxis(stretched, 0, -1) * 255).astype(np.uint8))
    return output


def difference_preview(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Show absolute RGB preview error on a fixed 0–48 display-level scale."""
    error = np.abs(left - right)
    # Inputs are display previews in HWC layout; reducing axis 0 collapsed the
    # heatmap to a 3-pixel-wide strip instead of averaging each pixel's RGB.
    magnitude = np.mean(error, axis=-1)
    value = np.clip(magnitude / ERROR_PREVIEW_SCALE, 0.0, 1.0)
    # Keep colors comparable across samples instead of stretching every sample
    # independently until its 99th percentile becomes yellow.
    stops = np.array((0.0, 0.33, 0.66, 1.0), dtype=np.float32)
    palette = np.array(
        ((30, 25, 90), (45, 115, 185), (40, 185, 155), (250, 230, 70)),
        dtype=np.float32,
    )
    return np.stack(
        [np.interp(value, stops, palette[:, channel]) for channel in range(3)],
        axis=-1,
    ).round().astype(np.uint8)


def _score_pair(reference: np.ndarray, estimate: np.ndarray, valid: np.ndarray) -> dict:
    difference = (estimate - reference)[:, valid]
    mae = float(np.abs(difference).mean()) / REFLECTANCE_MAX
    rmse = float(np.sqrt(np.square(difference).mean())) / REFLECTANCE_MAX
    psnr = float("inf") if rmse == 0 else float(20.0 * np.log10(1.0 / rmse))
    return {"MAE": mae, "RMSE": rmse, "PSNR": psnr}


def compare_reference(cloudy_path: Path, output_path: Path, target_path: Path) -> dict | None:
    with (
        rasterio.open(cloudy_path) as cloudy,
        rasterio.open(output_path) as output,
        rasterio.open(target_path) as target,
    ):
        datasets = (cloudy, output, target)
        grid = (cloudy.count, cloudy.width, cloudy.height, cloudy.crs, cloudy.transform)
        if any(
            (ds.count, ds.width, ds.height, ds.crs, ds.transform) != grid for ds in datasets[1:]
        ):
            return None
        cloudy_values = cloudy.read().astype(np.float32)
        output_values = output.read().astype(np.float32)
        target_values = target.read().astype(np.float32)
        valid = np.logical_and.reduce([ds.dataset_mask() > 0 for ds in datasets])
    if not valid.any():
        return None
    return {
        "cloudy": _score_pair(target_values, cloudy_values, valid),
        "output": _score_pair(target_values, output_values, valid),
    }


def run_model(s2_path, sar_path, output_path, checkpoint, tile, overlap, device):
    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--s2-cloudy", str(s2_path),
        "--s1-vv-vh", str(sar_path),
        "--checkpoint", str(checkpoint),
        "--output", str(output_path),
        "--tile", str(tile),
        "--overlap", str(overlap),
        "--device", device,
    ]
    return subprocess.run(
        command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )


def readable_case(row: dict) -> str:
    split = "TRAIN DEMO" if row["split"].lower() == "train" else "HELD-OUT TEST"
    tags = []
    if row.get("cloud_coverage_pct"):
        coverage = float(row["cloud_coverage_pct"])
        if coverage >= 80:
            tags.append(f"HEAVY CLOUD {coverage:.0f}%")
        elif row.get("land_cover"):
            tags.append(f"URBAN / BUILT-UP · {coverage:.0f}% CLOUD")
    if row.get("land_cover") and not tags:
        tags.append("URBAN / BUILT-UP")
    prefix = " · ".join(tags + [split])
    return f"{prefix} · {row['season'].title()} · {row['scene']} · patch {row['patch']}"


def render_sidebar(cases: list[dict]):
    st.sidebar.title("Reconstruction")
    gallery_mode = "Prepared paired sample gallery"
    mode = st.sidebar.radio(
        "Input mode",
        (gallery_mode, "Upload raster pair", "Use local file paths"),
        index=0 if cases else 1,
        key="input_mode",
    )
    checkpoint = PROJECT_ROOT / "weights" / "clearskkyai.pth"
    model_label = "ClearSky-AI Sentinel"
    if not checkpoint_ready(checkpoint):
        st.sidebar.error("Checkpoint is not available. Run `git lfs pull`.")
    if GALLERY_LFS_PENDING:
        st.sidebar.warning("Sample imagery is still stored as Git LFS pointers. Run `git lfs pull`.")

    cloudy_path = sar_path = target_path = None
    uploads = []
    input_name = ""
    sample_note = ""
    selected = None

    if mode == gallery_mode:
        if cases:
            filtered_cases = [row for row in cases if row["split"].lower() != "train"]
            st.sidebar.success(f"{len(filtered_cases)} held-out paired samples")
            seasons = sorted({row["season"].title() for row in filtered_cases})
            season_filter = st.sidebar.selectbox("Season", ("All seasons", *seasons))
            if season_filter != "All seasons":
                filtered_cases = [
                    row for row in filtered_cases if row["season"].title() == season_filter
                ]
            sample_by_id = {row["sample_id"]: row for row in filtered_cases}
            if st.session_state.get("gallery_sample") not in sample_by_id:
                st.session_state["gallery_sample"] = next(iter(sample_by_id))
            selected_id = st.sidebar.selectbox(
                "Select paired sample",
                list(sample_by_id),
                format_func=lambda sample_id: readable_case(sample_by_id[sample_id]),
                key="gallery_sample",
            )
            selected = sample_by_id[selected_id]
            cloudy_path = selected["cloudy_s2"]
            sar_path = selected["sar_s1"]
            target_path = selected["clear_target"]
            sample_note = selected["note"]
            input_name = selected_id
        else:
            st.sidebar.warning("No paired gallery samples are available.")
    elif mode == "Upload raster pair":
        s2_upload = st.sidebar.file_uploader(
            "Cloudy Sentinel-2 · 13 bands", type=("tif", "tiff"), key="s2_upload"
        )
        sar_upload = st.sidebar.file_uploader(
            "Sentinel-1 · VV then VH in dB", type=("tif", "tiff"), key="sar_upload"
        )
        uploads = [s2_upload, sar_upload]
        input_name = ", ".join(upload.name for upload in uploads if upload is not None)
    else:
        cloudy_text = st.sidebar.text_input(
            "Cloudy Sentinel-2 path", placeholder=r"C:\data\s2_cloudy_13band.tif"
        )
        sar_text = st.sidebar.text_input(
            "Sentinel-1 VV,VH path", placeholder=r"C:\data\s1_vv_vh.tif"
        )
        if cloudy_text.strip() and sar_text.strip():
            cloudy_path = Path(cloudy_text.strip()).expanduser().resolve()
            sar_path = Path(sar_text.strip()).expanduser().resolve()
            input_name = f"{cloudy_path.name}, {sar_path.name}"

    with st.sidebar.expander("Inference settings"):
        tile = st.select_slider("Tile size", options=(128, 256, 384, 512), value=256)
        overlap = st.slider("Tile overlap (pixels)", 0, tile - 1, min(32, tile - 1))
        device = st.selectbox("Device", ("auto", "cpu", "cuda"), index=0)
    run_clicked = st.sidebar.button("☁️ Run Real Cloud Removal", type="primary", width="stretch")
    return {
        "mode": mode,
        "cloudy_path": cloudy_path,
        "sar_path": sar_path,
        "target_path": target_path,
        "uploads": uploads,
        "input_name": input_name,
        "sample_note": sample_note,
        "selected": selected,
        "tile": tile,
        "overlap": overlap,
        "device": device,
        "checkpoint": checkpoint,
        "model_label": model_label,
        "run_clicked": run_clicked,
    }


def run_selected_input(selection: dict) -> None:
    if not checkpoint_ready(selection["checkpoint"]):
        st.error("Model weights are missing or still Git LFS pointers. Install Git LFS and run `git lfs pull`.")
        return
    if selection["mode"] == "Upload raster pair":
        if any(upload is None for upload in selection["uploads"]):
            st.sidebar.error("Upload both the cloudy S2 and aligned VV/VH SAR GeoTIFFs.")
            return
    elif selection["cloudy_path"] is None or selection["sar_path"] is None:
        st.sidebar.error("Select a gallery sample or supply both required raster paths.")
        return

    run_directory = tempfile.TemporaryDirectory(prefix="clearsky_sentinel_")
    try:
        work = Path(run_directory.name)
        cloudy_path = selection["cloudy_path"]
        sar_path = selection["sar_path"]
        if selection["mode"] == "Upload raster pair":
            cloudy_path = stage_upload(selection["uploads"][0], work / "cloudy_s2.tif")
            sar_path = stage_upload(selection["uploads"][1], work / "sar_s1.tif")
        output_path = work / "clearsky_reconstruction.tif"

        with st.spinner("Checking the grids and running the 13-band reconstruction model…"):
            completed = run_model(
                cloudy_path,
                sar_path,
                output_path,
                selection["checkpoint"],
                selection["tile"],
                selection["overlap"],
                selection["device"],
            )
            cloudy_raw = read_true_color(cloudy_path)
            output_raw = read_true_color(output_path)
            target_raw = read_true_color(selection["target_path"]) if selection["target_path"] else None
            inputs = [cloudy_raw, output_raw]
            if target_raw is not None:
                inputs.append(target_raw)
            previews = make_shared_previews(
                *inputs,
                stretch_indices=(1, 2) if target_raw is not None else None,
            )
            sar_preview = read_sar_preview(sar_path)
            if target_raw is not None:
                difference = difference_preview(previews[1], previews[2])
                metrics = compare_reference(cloudy_path, output_path, selection["target_path"])
                difference_caption = (
                    "RGB preview error · dark = lower · yellow = 48+ display levels"
                )
            else:
                difference = difference_preview(previews[0], previews[1])
                metrics = None
                difference_caption = (
                    "RGB preview change · dark = lower · yellow = 48+ display levels"
                )
            with rasterio.open(output_path) as result_ds:
                info = {
                    "width": result_ds.width,
                    "height": result_ds.height,
                    "crs": result_ds.crs.to_string() if result_ds.crs else "No CRS (pixel patch)",
                    "dtype": result_ds.dtypes[0],
                }

        previous = st.session_state.get("result_tempdir")
        if previous is not None:
            previous.cleanup()
        st.session_state["result_tempdir"] = run_directory
        st.session_state["last_result"] = {
            "output": str(output_path),
            "input_name": selection["input_name"],
            "model_label": selection["model_label"],
            "sample_note": selection["sample_note"],
            "sample": selection["selected"],
            "cloudy": previews[0],
            "estimate": previews[1],
            "target": previews[2] if target_raw is not None else None,
            "sar": sar_preview,
            "difference": difference,
            "difference_caption": difference_caption,
            "metrics": metrics,
            "stdout": completed.stdout,
            **info,
        }
        run_directory = None
    except subprocess.CalledProcessError as exc:
        lines = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()
        st.error("Reconstruction failed. Check the band order, units, checkpoint, and co-registration.")
        st.code("\n".join(lines[-18:]))
    except Exception as exc:
        st.error(f"Could not complete the reconstruction: {exc}")
    finally:
        if run_directory is not None:
            run_directory.cleanup()


def display_results(result: dict | None, current_input: str) -> None:
    st.markdown(
        "<style>"
        ".block-container{padding-top:1.5rem;padding-bottom:2rem;}"
        "[data-testid='stSidebar']{background:#20212b;}"
        "[data-testid='stMetric']{background:#171922;padding:14px;border-radius:10px;}"
        "[data-testid='stImage'] img{border-radius:8px;max-width:100%!important;"
        "height:auto!important;object-fit:contain;}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.title("🛰️ ClearSky-AI: SAR–Optical Cloud Reconstruction")
    st.caption(
        "Sentinel-1-guided Sentinel-2 reconstruction"
    )

    if result is None or result["input_name"] != current_input:
        st.info("Choose a paired sample or provide both inputs in the control panel, then run reconstruction.")
        return

    st.caption(
        f"Scene: {result['width']:,} × {result['height']:,} px · {result['crs']}"
    )
    columns = st.columns(4, gap="small")
    headings = ("1. SAR (Radar)", "2. Cloudy Sentinel-2", "3. ClearSky-AI Estimate", "4. Clear Reference")
    images = (result["sar"], result["cloudy"], result["estimate"], result["target"])
    descriptions = (
        "Sentinel-1 VV backscatter",
        "Genuine cloudy Sentinel-2 observation",
        "SAR-guided reconstruction estimate",
        "Paired clear reference" if result["target"] is not None else "No reference supplied",
    )
    for column, heading, image, caption in zip(columns, headings, images, descriptions):
        column.markdown(f"### {heading}")
        if image is None:
            column.info("A clear target is available only for paired gallery samples.")
        else:
            column.image(image, caption=caption, width="stretch")

    st.divider()
    scene_col, difference_col, metrics_col = st.columns([1.0, 1.25, 2.1], gap="large")
    scene_col.markdown("### 🛰️ Scene")
    sample = result["sample"]
    if sample:
        scene_col.write(f"**Split:** {sample['split'].upper()}")
        scene_col.write(f"**Season:** {sample['season'].title()}")
        scene_col.write(f"**Scene:** {sample['scene']}")
        scene_col.write(f"**Patch:** {sample['patch']}")
        if sample.get("cloud_coverage_pct"):
            scene_col.write(f"**Annotated cloud coverage:** {float(sample['cloud_coverage_pct']):.2f}%")
        if sample.get("land_cover"):
            scene_col.write(f"**Land cover:** {sample['land_cover']}")
        scene_col.caption(sample["note"])
    else:
        scene_col.write("**Source:** User-provided raster pair")
        scene_col.write(f"**Input files:** {result['input_name']}")

    difference_col.markdown("### 🔎 Reconstruction Difference")
    difference_col.image(
        result["difference"],
        caption=result["difference_caption"],
        width="stretch",
    )

    metrics_col.markdown("### 📊 Reference Comparison")
    metrics = result["metrics"]
    if metrics is None:
        metrics_col.info("No paired clear target: quality metrics are not available for this input.")
    else:
        top = metrics_col.columns(3)
        top[0].metric("Cloudy Input PSNR", f"{metrics['cloudy']['PSNR']:.2f} dB")
        top[1].metric("Reconstruction PSNR", f"{metrics['output']['PSNR']:.2f} dB")
        gain = metrics["output"]["PSNR"] - metrics["cloudy"]["PSNR"]
        top[2].metric("PSNR Gain", f"{gain:+.2f} dB")
        bottom = metrics_col.columns(2)
        bottom[0].metric("Cloudy Input MAE", f"{metrics['cloudy']['MAE']:.4f}")
        bottom[1].metric("Reconstruction MAE", f"{metrics['output']['MAE']:.4f}")
        st.caption(
            "Metrics use all 13 bands, reflectance normalized to 0–1. The clear reference is "
            "co-registered but may be from another date; treat these as paired-reference scores, "
            "not perfect ground truth. Train-split scores are not held-out performance."
        )

    st.download_button(
        "Download reconstructed 13-band GeoTIFF",
        data=lambda: Path(result["output"]).read_bytes(),
        file_name="clearsky_sentinel2_reconstruction.tif",
        mime="image/tiff",
        type="primary",
    )
def reconstruction_page() -> None:
    cases = read_gallery()
    selection = render_sidebar(cases)
    if selection["run_clicked"]:
        run_selected_input(selection)

    current_input = selection["input_name"]
    result = st.session_state.get("last_result")
    display_results(result, current_input)
    st.caption("Cloud removal is reconstruction, not recovery of guaranteed ground truth.")


def benchmark_page() -> None:
    st.title("📊 ClearSky-AI Benchmark")
    st.caption("Full held-out Sentinel-2 reconstruction evaluation")

    evaluation = read_evaluation_summary()
    if evaluation is None:
        st.error("Benchmark results are not available.")
        return

    test_summary = evaluation["splits"].get("test")
    if test_summary is None:
        st.error("No held-out test summary was found in summary.json.")
        return

    cloudy = test_summary["cloudy"]
    output = test_summary["output"]
    gain = test_summary["mean_PSNR_gain_dB"]

    columns = st.columns(5)
    columns[0].metric("Held-out patches", f"{test_summary['patches']:,}")
    columns[1].metric("Cloudy PSNR", f"{cloudy['PSNR_dB']:.2f} dB")
    columns[2].metric("Reconstruction PSNR", f"{output['PSNR_dB']:.2f} dB")
    columns[3].metric("PSNR gain", f"{gain:+.2f} dB")
    columns[4].metric(
        "Improved patches",
        f"{test_summary['patches_with_positive_PSNR_gain']}/{test_summary['patches']}",
    )

    quality = st.columns(5)
    quality[0].metric("Cloudy MAE", f"{cloudy['MAE_0_1']:.5f}")
    quality[1].metric("Reconstruction MAE", f"{output['MAE_0_1']:.5f}")
    quality[2].metric("Cloudy SSIM", f"{cloudy['SSIM']:.5f}")
    quality[3].metric("Reconstruction SSIM", f"{output['SSIM']:.5f}")
    quality[4].metric("Reconstruction SAM", f"{output['SAM_degrees']:.3f}°")

    st.caption(
        "Paired-reference benchmark on 964 SEN12MS-CR held-out patches. "
        "Clear references may come from a different acquisition date."
    )
    with st.expander("How to read these metrics"):
        st.markdown(
            "**PSNR** measures how closely the image matches the clear reference. "
            "Higher is better; each patch gain is reconstruction PSNR minus cloudy PSNR.\n\n"
            "**MAE** is the average absolute difference across all 13 bands. "
            "Lower is better.\n\n"
            "**RMSE** emphasizes larger pixel errors more strongly than MAE. "
            "Lower is better.\n\n"
            "**SSIM** measures structure, brightness, and contrast similarity. "
            "It ranges from 0 to 1; higher is better.\n\n"
            "**SAM** compares spectral direction in degrees across the 13 bands. "
            "Lower is better because it means the reconstructed spectrum is closer "
            "to the clear reference."
        )

    graph_dir = EVAL_SUMMARY.parent
    psnr_tab, quality_tab, gain_tab = st.tabs(
        ("PSNR distribution", "MAE & SSIM", "Cumulative gain")
    )
    graph_specs = (
        (
            psnr_tab,
            "psnr_benchmark.png",
            "Cloudy versus reconstructed PSNR",
            "The left histogram shows how many patches gained each amount of PSNR. "
            "The right scatter plot compares cloudy and reconstructed PSNR; points above "
            "the diagonal indicate an improvement.",
        ),
        (
            quality_tab,
            "quality_benchmark.png",
            "MAE and SSIM distributions",
            "The MAE box plot compares average pixel error: lower values are better. "
            "The SSIM box plot compares structural similarity: higher values are better.",
        ),
        (
            gain_tab,
            "cumulative_psnr_gain.png",
            "Cumulative held-out PSNR gain",
            "This curve shows the percentage of held-out patches at or above each PSNR gain "
            "value. A point above zero PSNR gain represents an improved patch.",
        ),
    )
    for tab, filename, caption, explanation in graph_specs:
        graph_path = graph_dir / filename
        with tab:
            if graph_path.is_file():
                st.image(str(graph_path), caption=caption, width="stretch")
                st.caption(explanation)
            else:
                st.warning(f"Graph not found: {filename}")

    st.divider()
    metrics_path = graph_dir / "metrics.csv"
    report_path = graph_dir / "benchmark_report.md"
    download_columns = st.columns(3)
    if metrics_path.is_file():
        download_columns[0].download_button(
            "Download per-patch metrics CSV",
            metrics_path.read_bytes(),
            file_name="clearsky_benchmark_metrics.csv",
            mime="text/csv",
        )
    if EVAL_SUMMARY.is_file():
        download_columns[1].download_button(
            "Download benchmark summary JSON",
            EVAL_SUMMARY.read_bytes(),
            file_name="clearsky_benchmark_summary.json",
            mime="application/json",
        )
    if report_path.is_file():
        download_columns[2].download_button(
            "Download benchmark report",
            report_path.read_bytes(),
            file_name="clearsky_benchmark_report.md",
            mime="text/markdown",
        )



def main() -> None:
    st.set_page_config(
        page_title="ClearSky-AI",
        page_icon="🛰️",
        layout="wide",
    )
    pages = [
        st.Page(reconstruction_page, title="Reconstruction", icon="🛰️"),
        st.Page(benchmark_page, title="Benchmark", icon="📊"),
    ]
    navigation = st.navigation(pages, position="sidebar")
    navigation.run()


main()

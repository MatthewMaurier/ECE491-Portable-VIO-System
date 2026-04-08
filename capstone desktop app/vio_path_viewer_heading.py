import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import streamlit as st


st.set_page_config(page_title="VIO Path Viewer", layout="wide")

st.title("VIO Path Viewer")
st.write(
    "Upload a VIO pose CSV and recreate the recorded path from the logged `x`, `y`, and `z` columns."
)

DEFAULT_FILE = Path("/mnt/data/vio_pose_20260326_173418.csv")
REQUIRED_COLUMNS = ["x", "y", "z"]
ANGLE_COLUMNS = ["roll_deg", "pitch_deg", "yaw_deg"]


def load_csv(file_obj) -> pd.DataFrame:
    df = pd.read_csv(file_obj)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    return df



def moving_average_1d(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.copy()
    window = min(window, len(values))
    if window % 2 == 0:
        window += 1
        if window > len(values):
            window -= 1
    if window <= 1:
        return values.copy()

    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")



def smooth_coords(coords: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return coords.copy()
    smoothed = np.empty_like(coords, dtype=float)
    for i in range(coords.shape[1]):
        smoothed[:, i] = moving_average_1d(coords[:, i], window)
    return smoothed



def segment_lengths_from_coords(coords: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return np.array([], dtype=float)
    diffs = np.diff(coords, axis=0)
    return np.linalg.norm(diffs, axis=1)



def get_forward_vectors(df: pd.DataFrame) -> np.ndarray | None:
    if not all(col in df.columns for col in ANGLE_COLUMNS):
        return None

    roll = np.deg2rad(df["roll_deg"].to_numpy(dtype=float))
    pitch = np.deg2rad(df["pitch_deg"].to_numpy(dtype=float))
    yaw = np.deg2rad(df["yaw_deg"].to_numpy(dtype=float))

    # Forward direction of the camera/body x-axis using yaw-pitch-roll convention.
    # Roll does not affect the forward-axis direction itself.
    fx = np.cos(yaw) * np.cos(pitch)
    fy = np.sin(yaw) * np.cos(pitch)
    fz = -np.sin(pitch)

    forward = np.column_stack([fx, fy, fz])
    norms = np.linalg.norm(forward, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return forward / norms


@st.cache_data
def prepare_data(df: pd.DataFrame, smooth_window: int, min_step_m: float):
    raw_coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    filtered_coords = smooth_coords(raw_coords, smooth_window)
    forward_vectors = get_forward_vectors(df)

    raw_segments = segment_lengths_from_coords(raw_coords)
    filtered_segments = segment_lengths_from_coords(filtered_coords)

    if len(filtered_segments) > 0:
        kept_segments = np.where(filtered_segments >= min_step_m, filtered_segments, 0.0)
    else:
        kept_segments = filtered_segments

    total_path_raw = float(raw_segments.sum())
    total_path_filtered = float(kept_segments.sum())
    displacement_raw = float(np.linalg.norm(raw_coords[-1] - raw_coords[0])) if len(raw_coords) > 1 else 0.0
    displacement_filtered = float(np.linalg.norm(filtered_coords[-1] - filtered_coords[0])) if len(filtered_coords) > 1 else 0.0

    mins = filtered_coords.min(axis=0)
    maxs = filtered_coords.max(axis=0)

    return {
        "raw_coords": raw_coords,
        "filtered_coords": filtered_coords,
        "forward_vectors": forward_vectors,
        "raw_segments": raw_segments,
        "filtered_segments": filtered_segments,
        "kept_segments": kept_segments,
        "total_path_raw": total_path_raw,
        "total_path_filtered": total_path_filtered,
        "displacement_raw": displacement_raw,
        "displacement_filtered": displacement_filtered,
        "mins": mins,
        "maxs": maxs,
    }



def make_2d_plot(
    coords: np.ndarray,
    view: str,
    show_points: bool,
    equal_axis: bool,
    raw_coords: np.ndarray | None = None,
    selected_index: int | None = None,
    forward_vectors: np.ndarray | None = None,
    arrow_scale: float = 0.15,
):
    fig, ax = plt.subplots(figsize=(7, 7))

    axis_map = {
        "Top-down (X-Y)": (0, 1, "X", "Y"),
        "Front (X-Z)": (0, 2, "X", "Z"),
        "Side (Y-Z)": (1, 2, "Y", "Z"),
    }
    i, j, xlabel, ylabel = axis_map[view]

    if raw_coords is not None:
        ax.plot(raw_coords[:, i], raw_coords[:, j], alpha=0.35, label="Raw")
        ax.plot(coords[:, i], coords[:, j], linewidth=2, label="Filtered")
        ax.legend()
    else:
        ax.plot(coords[:, i], coords[:, j])

    if show_points:
        ax.scatter(coords[0, i], coords[0, j], marker="o", s=60, label="Start")
        ax.scatter(coords[-1, i], coords[-1, j], marker="x", s=60, label="End")
        if raw_coords is None:
            ax.legend()

    if selected_index is not None and 0 <= selected_index < len(coords):
        px, py = coords[selected_index, i], coords[selected_index, j]
        ax.scatter(px, py, s=90, marker="s", label="Selected sample")

        if forward_vectors is not None and selected_index < len(forward_vectors):
            dx = forward_vectors[selected_index, i]
            dy = forward_vectors[selected_index, j]
            ax.quiver(
                px,
                py,
                dx,
                dy,
                angles="xy",
                scale_units="xy",
                scale=1 / max(arrow_scale, 1e-6),
                width=0.006,
            )
        ax.annotate(f"sample {selected_index}", (px, py), textcoords="offset points", xytext=(8, 8))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Path View: {view}")
    ax.grid(True)
    if equal_axis:
        ax.set_aspect("equal", adjustable="box")
    return fig



def make_3d_plot(
    coords: np.ndarray,
    show_points: bool,
    raw_coords: np.ndarray | None = None,
    selected_index: int | None = None,
    forward_vectors: np.ndarray | None = None,
    arrow_scale: float = 0.15,
):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    if raw_coords is not None:
        ax.plot(raw_coords[:, 0], raw_coords[:, 1], raw_coords[:, 2], alpha=0.3, label="Raw")
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], linewidth=2, label="Filtered")
        ax.legend()
    else:
        ax.plot(coords[:, 0], coords[:, 1], coords[:, 2])

    if show_points:
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], marker="o", s=60)
        ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], marker="x", s=60)

    if selected_index is not None and 0 <= selected_index < len(coords):
        px, py, pz = coords[selected_index]
        ax.scatter(px, py, pz, marker="s", s=70)
        if forward_vectors is not None and selected_index < len(forward_vectors):
            dx, dy, dz = forward_vectors[selected_index] * arrow_scale
            ax.quiver(px, py, pz, dx, dy, dz, length=1.0, normalize=False)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Path")
    return fig


uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    source_name = uploaded_file.name
    df = load_csv(uploaded_file)
elif DEFAULT_FILE.exists():
    source_name = DEFAULT_FILE.name
    df = load_csv(DEFAULT_FILE)
    st.info(f"Using bundled example file: {DEFAULT_FILE.name}")
else:
    st.stop()

with st.sidebar:
    st.header("Controls")
    view = st.selectbox("2D view", ["Top-down (X-Y)", "Front (X-Z)", "Side (Y-Z)"])
    show_points = st.checkbox("Show start/end markers", value=True)
    equal_axis = st.checkbox("Keep equal axis scaling", value=True)
    overlay_raw = st.checkbox("Overlay raw path", value=True)

    st.subheader("Jitter reduction")
    smooth_window = st.slider("Smoothing window (samples)", min_value=1, max_value=31, value=7, step=2)
    min_step_m = st.slider("Ignore tiny steps below (m)", min_value=0.0, max_value=0.05, value=0.005, step=0.001)

    st.subheader("Playback")
    max_index = len(df) - 1
    n_points = st.slider("Show first N samples", min_value=2, max_value=max_index + 1, value=max_index + 1)

prepared = prepare_data(df, smooth_window, min_step_m)
raw_coords = prepared["raw_coords"]
filtered_coords = prepared["filtered_coords"]
forward_vectors = prepared["forward_vectors"]
plot_raw = raw_coords[:n_points] if overlay_raw else None
plot_filtered = filtered_coords[:n_points]

with st.sidebar:
    st.subheader("Selected pose")
    selected_index = st.slider("Sample index", min_value=0, max_value=n_points - 1, value=min(n_points - 1, max(0, n_points // 2)))
    arrow_scale = st.slider("Arrow length (m)", min_value=0.02, max_value=1.0, value=0.15, step=0.01)
    show_heading = st.checkbox("Show facing direction arrow", value=True, disabled=forward_vectors is None)

st.subheader("Detected columns")
st.write(list(df.columns))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Samples", f"{len(df):,}")
c2.metric("Raw path length", f"{prepared['total_path_raw']:.3f} m")
c3.metric("Filtered path length", f"{prepared['total_path_filtered']:.3f} m")
c4.metric("Filtered displacement", f"{prepared['displacement_filtered']:.3f} m")

c5, c6, c7 = st.columns(3)
c5.metric("X range", f"{prepared['mins'][0]:.3f} to {prepared['maxs'][0]:.3f}")
c6.metric("Y range", f"{prepared['mins'][1]:.3f} to {prepared['maxs'][1]:.3f}")
c7.metric("Z range", f"{prepared['mins'][2]:.3f} to {prepared['maxs'][2]:.3f}")

if show_heading and forward_vectors is not None:
    sample_row = df.iloc[selected_index]
    pos = filtered_coords[selected_index]
    a1, a2, a3 = st.columns(3)
    a1.metric("Selected sample", str(selected_index))
    a2.metric("Selected time", str(sample_row.get("iso_time", sample_row.get("unix_time", "N/A"))))
    a3.metric("Selected yaw", f"{float(sample_row['yaw_deg']):.2f}°")
    st.write(
        {
            "position_m": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
            "orientation_deg": {
                "roll": float(sample_row["roll_deg"]),
                "pitch": float(sample_row["pitch_deg"]),
                "yaw": float(sample_row["yaw_deg"]),
            },
        }
    )
elif forward_vectors is None:
    st.warning("Heading arrows are unavailable because this CSV does not include roll_deg, pitch_deg, and yaw_deg columns.")

left, right = st.columns(2)
with left:
    st.pyplot(
        make_2d_plot(
            plot_filtered,
            view,
            show_points,
            equal_axis,
            raw_coords=plot_raw,
            selected_index=selected_index,
            forward_vectors=forward_vectors[:n_points] if (show_heading and forward_vectors is not None) else None,
            arrow_scale=arrow_scale,
        ),
        clear_figure=True,
    )
with right:
    st.pyplot(
        make_3d_plot(
            plot_filtered,
            show_points,
            raw_coords=plot_raw,
            selected_index=selected_index,
            forward_vectors=forward_vectors[:n_points] if (show_heading and forward_vectors is not None) else None,
            arrow_scale=arrow_scale,
        ),
        clear_figure=True,
    )

st.write(
    "The filtered path is computed with a moving-average smoother, and the filtered path length ignores tiny frame-to-frame motions below the selected threshold. The selected sample can also show the camera facing direction as an arrow."
)

st.subheader("Data preview")
st.dataframe(df.head(25), use_container_width=True)

csv_buffer = io.StringIO()
out_df = df.copy()
out_df[["x_filtered", "y_filtered", "z_filtered"]] = filtered_coords
out_df["segment_length_raw_m"] = np.concatenate([[0.0], prepared["raw_segments"]]) if len(df) > 1 else [0.0]
out_df["segment_length_filtered_m"] = np.concatenate([[0.0], prepared["filtered_segments"]]) if len(df) > 1 else [0.0]
out_df["segment_length_kept_m"] = np.concatenate([[0.0], prepared["kept_segments"]]) if len(df) > 1 else [0.0]
out_df.to_csv(csv_buffer, index=False)

st.download_button(
    label="Download CSV with filtered coordinates",
    data=csv_buffer.getvalue(),
    file_name=f"processed_{source_name}",
    mime="text/csv",
)

st.caption(
    "Tip: increase the smoothing window or minimum step threshold if VIO jitter is making the path look too zig-zaggy. Use the sample index slider to inspect where the camera was pointing at any moment."
)

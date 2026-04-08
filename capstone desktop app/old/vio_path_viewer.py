import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import streamlit as st


st.set_page_config(page_title="VIO Path Viewer", layout="wide")

st.title("VIO Path Viewer")
st.write(
    "Upload a VIO pose CSV and this app will recreate the recorded path using the logged `x`, `y`, and `z` columns."
)

DEFAULT_FILE = Path("/mnt/data/vio_pose_20260326_173418.csv")
REQUIRED_COLUMNS = ["x", "y", "z"]
OPTIONAL_COLUMNS = ["unix_time", "iso_time", "roll_deg", "pitch_deg", "yaw_deg"]


def load_csv(file_obj) -> pd.DataFrame:
    df = pd.read_csv(file_obj)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")
    return df


@st.cache_data
def compute_metrics(df: pd.DataFrame):
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)
    diffs = np.diff(coords, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_path = float(segment_lengths.sum())
    displacement = float(np.linalg.norm(coords[-1] - coords[0])) if len(coords) > 1 else 0.0
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return coords, segment_lengths, total_path, displacement, mins, maxs


def make_2d_plot(coords: np.ndarray, view: str, show_points: bool, equal_axis: bool):
    fig, ax = plt.subplots(figsize=(7, 7))

    axis_map = {
        "Top-down (X-Y)": (0, 1, "X", "Y"),
        "Front (X-Z)": (0, 2, "X", "Z"),
        "Side (Y-Z)": (1, 2, "Y", "Z"),
    }
    i, j, xlabel, ylabel = axis_map[view]

    ax.plot(coords[:, i], coords[:, j])
    if show_points:
        ax.scatter(coords[0, i], coords[0, j], marker="o", s=60, label="Start")
        ax.scatter(coords[-1, i], coords[-1, j], marker="x", s=60, label="End")
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Path View: {view}")
    ax.grid(True)
    if equal_axis:
        ax.set_aspect("equal", adjustable="box")
    return fig



def make_3d_plot(coords: np.ndarray, show_points: bool):
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2])

    if show_points:
        ax.scatter(coords[0, 0], coords[0, 1], coords[0, 2], marker="o", s=60, label="Start")
        ax.scatter(coords[-1, 0], coords[-1, 1], coords[-1, 2], marker="x", s=60, label="End")
        ax.legend()

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

st.subheader("Detected columns")
st.write(list(df.columns))

coords, segment_lengths, total_path, displacement, mins, maxs = compute_metrics(df)

with st.sidebar:
    st.header("Controls")
    view = st.selectbox("2D view", ["Top-down (X-Y)", "Front (X-Z)", "Side (Y-Z)"])
    show_points = st.checkbox("Show start/end markers", value=True)
    equal_axis = st.checkbox("Keep equal axis scaling", value=True)

    st.subheader("Playback")
    max_index = len(df) - 1
    n_points = st.slider("Show first N samples", min_value=2, max_value=max_index + 1, value=max_index + 1)

plot_coords = coords[:n_points]

c1, c2, c3 = st.columns(3)
c1.metric("Samples", f"{len(df):,}")
c2.metric("Total path length", f"{total_path:.3f} m")
c3.metric("Start-to-end displacement", f"{displacement:.3f} m")

c4, c5, c6 = st.columns(3)
c4.metric("X range", f"{mins[0]:.3f} to {maxs[0]:.3f}")
c5.metric("Y range", f"{mins[1]:.3f} to {maxs[1]:.3f}")
c6.metric("Z range", f"{mins[2]:.3f} to {maxs[2]:.3f}")

left, right = st.columns(2)
with left:
    st.pyplot(make_2d_plot(plot_coords, view, show_points, equal_axis), clear_figure=True)
with right:
    st.pyplot(make_3d_plot(plot_coords, show_points), clear_figure=True)

st.subheader("Data preview")
st.dataframe(df.head(25), use_container_width=True)

csv_buffer = io.StringIO()
out_df = df.copy()
out_df["segment_length_m"] = np.concatenate([[0.0], segment_lengths])
out_df.to_csv(csv_buffer, index=False)

st.download_button(
    label="Download CSV with segment lengths",
    data=csv_buffer.getvalue(),
    file_name=f"processed_{source_name}",
    mime="text/csv",
)

st.caption(
    "Tip: run this with `streamlit run vio_path_viewer.py` and upload any CSV that contains x, y, and z columns."
)

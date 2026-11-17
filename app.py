"""Streamlit Dashboard for Coffee Shop Occupancy Monitoring (Phase 4).

Features:
- Prerecorded video stream playback with ROI polygons and bounding box overlays
- Real-time seat and table availability status (AVAILABLE, OCCUPIED, RESERVED_BY_BELONGINGS, UNKNOWN)
- Operational metrics (Total Capacity, Occupancy Rate %, Occupied/Available Seats & Tables)
- Context-aware rule-based Seating Pressure assessment with transparent explanations
- Duration state machine monitoring (Normal, Approaching Limit, Overtime)
- Event log timeline and person visibility vs seating duration analysis
"""
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.metrics import OccupancyMetrics
from src.pipeline import CoffeeShopPipeline, PipelineConfig
from src.state_machine import AvailabilityState, DurationWarningState

st.set_page_config(
    page_title="Coffee Shop Occupancy & Seating Pressure",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished coffee shop styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #3E2723;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #5D4037;
        margin-bottom: 1.2rem;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-prerecorded {
        background-color: #EDE7F6;
        color: #4527A0;
        border: 1px solid #B39DDB;
    }
    .badge-low {
        background-color: #E8F5E9;
        color: #1B5E20;
    }
    .badge-medium {
        background-color: #FFF3E0;
        color: #E65100;
    }
    .badge-high {
        background-color: #FFEBEE;
        color: #B71C1C;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<div class="main-header">☕ Coffee Shop Seat Availability & Seating Pressure Monitoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        '<span class="badge-pill badge-prerecorded">📹 Prerecorded Video Feed Analysis</span> &nbsp; '
        'Computer Vision System powered by YOLOv8, ByteTrack, Spatial ROI Polygons & Temporal State Machine'
        '</div>',
        unsafe_allow_html=True
    )

    # Sidebar Configurations
    st.sidebar.header("⚙️ Konfigurasi Sistem")

    source_mode = st.sidebar.radio(
        "Sumber Video",
        ["📁 Upload Video Baru", "🎥 Video Sampel / Dataset"],
        index=0,
        help="Pilih apakah ingin mengunggah video sendiri atau menggunakan video sampel."
    )

    upload_dir = Path("data/videos/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    video_source = None
    if source_mode == "📁 Upload Video Baru":
        uploaded_file = st.sidebar.file_uploader(
            "Unggah File Video (.mp4, .avi, .mov, .mkv)",
            type=["mp4", "avi", "mov", "mkv"],
            help="Pilih file video dari komputer Anda untuk dianalisis."
        )
        if uploaded_file is not None:
            save_path = upload_dir / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_source = str(save_path)
            size_mb = uploaded_file.size / (1024 * 1024)
            st.sidebar.success(f"✅ Video siap: **{uploaded_file.name}** ({size_mb:.1f} MB)")
        else:
            st.sidebar.info("👆 Silakan unggah video untuk memulai analisis.")
    else:
        video_dir = Path("data/videos")
        sample_videos = {}
        if (video_dir / "kopisop.mp4").exists():
            sample_videos["kopisop.mp4 (Default Demo - ROI Tuned)"] = str(video_dir / "kopisop.mp4")
        for vf in sorted(video_dir.glob("*.mp4")):
            if vf.name != "kopisop.mp4" and vf.parent.name != "uploads":
                sample_videos[vf.name] = str(vf)

        selected_video_label = st.sidebar.selectbox("Pilih Video Sampel", list(sample_videos.keys()))
        video_source = sample_videos.get(selected_video_label)

    # Info resolusi & Preview Video jika tersedia
    if video_source and Path(video_source).exists():
        try:
            probe_cap = cv2.VideoCapture(video_source)
            v_w = int(probe_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            v_h = int(probe_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            v_fps = probe_cap.get(cv2.CAP_PROP_FPS) or 24.0
            v_cnt = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            probe_cap.release()
            st.sidebar.caption(f"📐 {v_w}x{v_h} | {v_fps:.1f} FPS | ~{v_cnt/v_fps:.1f}s ({v_cnt} frames)")
            with st.sidebar.expander("👁️ Preview Video Asli"):
                st.video(video_source)
        except Exception:
            pass

    default_weights = "models/best_yolo.pt" if Path("models/best_yolo.pt").exists() else "yolov8s.pt"
    model_weights = st.sidebar.text_input("YOLO Model Weights", value=default_weights)
    
    # Preset Tata Letak ROI (Region of Interest)
    is_communal = bool(video_source and ("People_working" in video_source or "communal" in video_source.lower()))
    default_preset_idx = 1 if is_communal else 0

    roi_presets = {
        "📐 Denah Meja Individual (kopisop.mp4 - 7 Meja Kecil)": "config/roi_config.json",
        "📐 Denah Meja Panjang Komunal (People_working... - 1 Meja Panjang)": "config/roi_communal_table.json",
        "⚙️ Kustom (Input Path Manual)": "custom"
    }
    selected_preset = st.sidebar.selectbox(
        "Preset Tata Letak Meja (ROI)",
        list(roi_presets.keys()),
        index=default_preset_idx,
        help="Pilih konfigurasi poligon meja (ROI) yang sesuai dengan tata letak video."
    )
    if roi_presets[selected_preset] == "custom":
        roi_config_path = st.sidebar.text_input("Path File ROI (.json)", value="config/roi_config.json")
    else:
        roi_config_path = roi_presets[selected_preset]
        st.sidebar.caption(f"📁 Menggunakan layout: `{roi_config_path}`")

    tracker_type = st.sidebar.selectbox("MOT Tracker Algorithm", ["bytetrack", "botsort"], index=0)
    conf_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.15, 0.80, 0.25, 0.05)
    
    st.sidebar.subheader("⏱️ Aturan Durasi & State")
    duration_limit_sec = st.sidebar.number_input(
        "Seating Duration Limit (seconds)",
        min_value=5.0,
        max_value=7200.0,
        value=60.0,
        step=5.0,
        help="Batas waktu monitoring durasi pelanggan sebelum status OVERTIME aktif."
    )
    grace_period_sec = st.sidebar.number_input(
        "Customer Grace Period (seconds)",
        min_value=1.0,
        max_value=300.0,
        value=3.0,
        step=1.0,
        help="Buffer toleransi ketidakhadiran sementara pelanggan sebelum status meja di-reset."
    )
    frame_step = st.sidebar.slider("Proses Setiap N Frame", 1, 10, 2, help="Lewati frame untuk pemutaran/analisis lebih cepat.")
    process_all_frames = st.sidebar.checkbox("Proses Seluruh Frame Video", value=False, help="Jika dicentang, analisis akan memproses seluruh durasi video.")
    if not process_all_frames:
        max_frames_to_run = st.sidebar.slider("Batas Maksimal Frame", 20, 1000, 120, 20)
    else:
        max_frames_to_run = 9999999

    # Pipeline setup
    cfg = PipelineConfig(
        video_source=video_source or "data/videos/kopisop.mp4",
        model_weights=model_weights,
        roi_config_path=roi_config_path,
        tracker_type=tracker_type,
        conf_threshold=conf_threshold,
        duration_limit_seconds=float(duration_limit_sec),
        grace_period_seconds=float(grace_period_sec)
    )

    col_btn1, col_btn2 = st.sidebar.columns(2)
    start_btn = col_btn1.button("▶️ Run Analysis", use_container_width=True, type="primary")
    stop_placeholder = col_btn2.empty()

    # Top Level KPI Metrics Containers
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    metric_occ_rate = kpi1.empty()
    metric_pressure = kpi2.empty()
    metric_seats = kpi3.empty()
    metric_tables = kpi4.empty()
    metric_belongings = kpi5.empty()

    # Main area: Video Player on left, Table details & events on right
    col_video, col_info = st.columns([3, 2])

    with col_video:
        st.subheader("Live Processed Video Feed")
        video_placeholder = st.empty()
        status_banner = st.empty()

    with col_info:
        st.subheader("Seating Layout Status")
        table_dataframe_placeholder = st.empty()

        st.subheader("📊 Seating Pressure Assessment")
        pressure_explanation_placeholder = st.empty()

        st.subheader("📜 Event Log & Session History")
        event_log_placeholder = st.empty()

    if start_btn:
        if not video_source or not Path(video_source).exists():
            st.sidebar.error("❌ Silakan unggah atau pilih video terlebih dahulu sebelum menjalankan analisis.")
            return
        try:
            pipeline = CoffeeShopPipeline(cfg)
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")
            return

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            st.error(f"Failed to open video source: {video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_idx = 0
        events: List[str] = []
        prev_table_states: Dict[str, AvailabilityState] = {}

        stop_btn = stop_placeholder.button("⏹️ Stop", use_container_width=True)

        while cap.isOpened() and frame_idx < max_frames_to_run:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / fps
            processed = pipeline.process_frame(frame, frame_idx, timestamp, draw_annotations=True)
            metrics: OccupancyMetrics = processed.metrics

            # Detect state transitions for event log
            for tid, ts in processed.table_states.items():
                prev_state = prev_table_states.get(tid)
                curr_state = ts.availability_state
                if prev_state is not None and prev_state != curr_state:
                    time_str = f"{int(timestamp // 60):02d}:{timestamp % 60:04.1f}"
                    events.insert(0, f"⏱️ [{time_str}] Table {tid} transitioned: {prev_state.value} ➡️ {curr_state.value}")
                    if len(events) > 15:
                        events.pop()
                if ts.duration_warning_state == DurationWarningState.OVERTIME and (prev_state is None or prev_state != curr_state):
                    time_str = f"{int(timestamp // 60):02d}:{timestamp % 60:04.1f}"
                    events.insert(0, f"⚠️ [{time_str}] Table {tid} is OVERTIME (Seated > {duration_limit_sec:.0f}s)")
                prev_table_states[tid] = curr_state

            # Update KPI metrics
            metric_occ_rate.metric(
                label="Seat Occupancy Rate",
                value=f"{metrics.seat_occupancy_rate:.1f}%",
                delta=f"{metrics.effective_seat_occupancy_rate:.1f}% eff."
            )
            metric_pressure.metric(
                label="Seating Pressure",
                value=metrics.seating_pressure,
                delta=f"Score: {metrics.pressure_score:.2f}"
            )
            metric_seats.metric(
                label="Available Seats",
                value=f"{metrics.available_seats} / {metrics.total_seats}",
                delta=f"{metrics.occupied_seats} occupied"
            )
            metric_tables.metric(
                label="Available Tables",
                value=f"{metrics.available_tables} / {metrics.total_tables}",
                delta=f"{metrics.occupied_tables} occupied"
            )
            metric_belongings.metric(
                label="Reserved by Belongings",
                value=f"{metrics.belongings_tables} tables",
                delta=f"{metrics.belongings_seats} seats"
            )

            # Update Video Frame
            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(processed.annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            status_banner.info(
                f"Frame: {frame_idx} | Video Time: {timestamp:.2f}s | "
                f"Tracked Persons: {len(processed.tracked_persons)} | Total Detections: {len(processed.detections)}"
            )

            # Update Seating Layout Table
            table_rows = []
            for tid, ts in processed.table_states.items():
                table_obj = pipeline.layout.tables.get(tid)
                seat_ids_str = ", ".join(s.id for s in table_obj.seats) if table_obj else "-"
                persons_str = ", ".join(f"#{pid}" for pid in ts.active_person_ids) if ts.active_person_ids else "None"
                
                table_rows.append({
                    "Table ID": tid,
                    "Seats": seat_ids_str,
                    "Status": ts.availability_state.value,
                    "Active Person": persons_str,
                    "Occupancy Duration": f"{ts.occupancy_duration:.1f}s",
                    "Belongings Duration": f"{ts.belongings_duration:.1f}s",
                    "Warning": ts.duration_warning_state.value
                })

            df_tables = pd.DataFrame(table_rows)
            table_dataframe_placeholder.dataframe(df_tables, use_container_width=True, hide_index=True)

            # Update Seating Pressure Explanation
            pressure_color = (
                "#D32F2F" if metrics.seating_pressure == "HIGH" else
                "#F57C00" if metrics.seating_pressure == "MEDIUM" else "#388E3C"
            )
            pressure_explanation_placeholder.markdown(
                f"""
                <div style="border-left: 5px solid {pressure_color}; padding: 10px 14px; background-color: #FAFAFA; border-radius: 4px;">
                    <strong style="color: {pressure_color}; font-size: 1.1rem;">Pressure Level: {metrics.seating_pressure}</strong><br/>
                    <span style="color: #424242; font-size: 0.92rem;"><strong>Rule Triggered:</strong> {metrics.pressure_reason}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Update Event Log
            if events:
                event_log_placeholder.markdown("\n".join(f"- {e}" for e in events[:8]))
            else:
                event_log_placeholder.caption("No state transition events recorded yet.")

            frame_idx += 1

        cap.release()
        status_banner.success("✅ Analysis completed successfully!")
    else:
        video_placeholder.info("Click '▶️ Run Analysis' in the sidebar to process the video stream.")
        pressure_explanation_placeholder.caption("Run analysis to assess real-time seating pressure.")
        event_log_placeholder.caption("Event logs will appear here during processing.")


if __name__ == "__main__":
    main()


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
    st.sidebar.header("⚙️ System Configuration")

    video_options = {
        "Default: data/videos/kopisop.mp4": "data/videos/kopisop.mp4",
        "Secondary: data/videos/coffee_shop.mp4": "data/videos/coffee_shop.mp4"
    }
    selected_video_label = st.sidebar.selectbox("Select Video Feed", list(video_options.keys()))
    video_source = video_options[selected_video_label]

    model_weights = st.sidebar.text_input("YOLO Model Weights", value="yolov8s.pt")
    roi_config_path = st.sidebar.text_input("ROI Configuration Path", value="config/roi_config.json")
    tracker_type = st.sidebar.selectbox("MOT Tracker Algorithm", ["bytetrack", "botsort"], index=0)
    conf_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.15, 0.80, 0.25, 0.05)
    
    st.sidebar.subheader("⏱️ Duration & State Rules")
    duration_limit_sec = st.sidebar.number_input(
        "Seating Duration Limit (seconds)",
        min_value=5.0,
        max_value=7200.0,
        value=60.0,
        step=5.0,
        help="Configurable threshold for duration monitoring (e.g. 60s for demo video, 3600s/7200s for operational cafe)."
    )
    grace_period_sec = st.sidebar.number_input(
        "Customer Grace Period (seconds)",
        min_value=1.0,
        max_value=300.0,
        value=3.0,
        step=1.0,
        help="Tolerance buffer allowing temporary customer absence (e.g. restroom/ordering) before table resets."
    )
    frame_step = st.sidebar.slider("Process Every Nth Frame", 1, 10, 2, help="Skip frames for faster playback.")
    max_frames_to_run = st.sidebar.slider("Max Frames to Process", 20, 240, 120, 20)

    # Pipeline setup
    cfg = PipelineConfig(
        video_source=video_source,
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


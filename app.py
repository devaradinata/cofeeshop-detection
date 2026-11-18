"""SeatVision — AI Monitoring Dashboard for Coffee Shop Occupancy & Seating Pressure.

Translates Google Stitch UI designs (Images 2, 3, 4, 5) into Streamlit
while preserving 100% of the underlying YOLOv8, ByteTrack, Spatial ROI, and State Machine backend.
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

# Ensure Ultralytics settings dir stays local to avoid permission issues
os.environ["YOLO_CONFIG_DIR"] = str((Path(__file__).parent / ".yolo_config").resolve())

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.metrics import OccupancyMetrics
from src.pipeline import CoffeeShopPipeline, PipelineConfig, ProcessedFrame
from src.state_machine import AvailabilityState, DurationWarningState
from ui.components import (
    get_compact_table_list_html,
    get_overview_kpis_html,
    get_video_feed_footer_html,
    render_top_header,
    render_video_feed_header,
)
from ui.styles import apply_custom_styles, clean_html_str, render_html
from ui.views.analytics import render_analytics_view
from ui.views.monitoring import render_monitoring_view
from ui.views.overview import render_overview_view
from ui.views.seating import render_seating_view
from ui.views.settings import render_settings_view

st.set_page_config(
    page_title="SeatVision — AI Monitoring Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Google Stitch CSS styling
apply_custom_styles()


def init_session_state():
    """Initialize persistent session states."""
    if "latest_frame" not in st.session_state:
        st.session_state.latest_frame = None
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "history_metrics" not in st.session_state:
        st.session_state.history_metrics = []
    if "events" not in st.session_state:
        st.session_state.events = []
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "selected_table_id" not in st.session_state:
        st.session_state.selected_table_id = "T1"
    if "video_source" not in st.session_state:
        sample_path = "data/videos/kopisop.mp4" if Path("data/videos/kopisop.mp4").exists() else ""
        st.session_state.video_source = sample_path

    # Live Configuration parameters synced between UI Settings and CV Engine
    if "config_model_weights" not in st.session_state:
        st.session_state.config_model_weights = "models/best_yolo.pt" if Path("models/best_yolo.pt").exists() else "yolov8s.pt"
    if "config_conf_threshold" not in st.session_state:
        st.session_state.config_conf_threshold = 0.25
    if "config_iou_threshold" not in st.session_state:
        st.session_state.config_iou_threshold = 0.45
    if "config_tracker_type" not in st.session_state:
        st.session_state.config_tracker_type = "bytetrack"
    if "config_roi_path" not in st.session_state:
        st.session_state.config_roi_path = "config/roi_config.json"
    if "config_duration_limit" not in st.session_state:
        st.session_state.config_duration_limit = 60.0
    if "config_approaching_limit" not in st.session_state:
        st.session_state.config_approaching_limit = 45.0
    if "config_overtime_limit" not in st.session_state:
        st.session_state.config_overtime_limit = 70.0
    if "config_grace_period" not in st.session_state:
        st.session_state.config_grace_period = 3.0
    if "config_imgsz" not in st.session_state:
        st.session_state.config_imgsz = 640
    if "show_overlays" not in st.session_state:
        st.session_state.show_overlays = True
    if "is_focused" not in st.session_state:
        st.session_state.is_focused = False
    if "audio_alerts" not in st.session_state:
        st.session_state.audio_alerts = True


def ensure_initial_frame():
    """Ensure session_state has a processed frame from the selected video on startup."""
    if st.session_state.latest_frame is None and st.session_state.video_source:
        v_source = st.session_state.video_source
        if Path(v_source).exists():
            cfg = PipelineConfig(
                video_source=v_source,
                model_weights=st.session_state.config_model_weights,
                roi_config_path=st.session_state.config_roi_path,
                tracker_type=st.session_state.config_tracker_type,
                conf_threshold=float(st.session_state.config_conf_threshold),
                imgsz=int(st.session_state.config_imgsz)
            )
            try:
                pipeline = CoffeeShopPipeline(cfg)
                st.session_state.pipeline = pipeline
                cap = cv2.VideoCapture(v_source)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    proc = pipeline.process_frame(frame, frame_idx=0, timestamp=0.0, draw_annotations=True)
                    st.session_state.latest_frame = proc
            except Exception as e:
                print(f"Initial frame processing error: {e}")


def main():
    init_session_state()
    ensure_initial_frame()

    # --- SIDEBAR (Google Stitch Sleek Sidebar from Images 2-5) ---
    with st.sidebar:
        sidebar_brand_html = """<div style="height:48px;display:flex;align-items:center;gap:10px;border-bottom:1px solid #F3F4F6;margin-bottom:16px;">
  <div style="width:28px;height:28px;border-radius:6px;background:#0F172A;display:flex;align-items:center;justify-content:center;color:#FFFFFF;font-size:13px;font-weight:600;">
    <svg style="width:16px;height:16px;fill:currentColor;" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="3"></circle>
      <path d="M4 6a2 2 0 012-2h12a2 2 0 012 2v12a2 2 0 01-2 2H6a2 2 0 01-2-2V6zm2 2v8h12V8H6z" opacity="0.3"></path>
    </svg>
  </div>
  <span style="font-size:15px;font-weight:600;color:#111827;letter-spacing:-0.02em;">SeatVision</span>
</div>"""
        render_html(sidebar_brand_html)

        nav_options = [
            "Overview",
            "Monitoring",
            "Seating",
            "Analytics",
            "Settings"
        ]
        nav_labels = {
            "Overview": "⊞  Overview",
            "Monitoring": "📹  Monitoring",
            "Seating": "🪑  Seating",
            "Analytics": "📈  Analytics",
            "Settings": "⚙  Settings"
        }

        selected_nav = st.radio(
            "Navigation Menu",
            nav_options,
            index=0,
            format_func=lambda x: nav_labels.get(x, x),
            label_visibility="collapsed"
        )

        # Sidebar Bottom Status & User Profile matching Images 2-5 (Pushed to bottom)
        sidebar_bottom_html = f"""<div style="display:flex;flex-direction:column;gap:8px;margin-top:calc(100vh - 380px);padding-top:16px;">
  <div style="padding:6px 10px;border-radius:6px;background:#FAFAFA;border:1px solid #E5E7EB;display:flex;align-items:center;justify-content:space-between;font-size:11px;color:#6B7280;font-family:'JetBrains Mono',monospace;">
    <span style="display:flex;align-items:center;gap:6px;">
      <span style="width:6px;height:6px;border-radius:50%;background:{'#10B981' if st.session_state.is_running else '#9CA3AF'};"></span>
      <span style="color:#374151;font-family:'Geist','Inter',sans-serif;font-weight:500;">Engine</span>
    </span>
    <span>{'29.8 FPS' if st.session_state.is_running else 'Ready'}</span>
  </div>

  <div style="display:flex;align-items:center;justify-content:space-between;padding:8px;margin-top:6px;border-top:1px solid #F3F4F6;">
    <div style="display:flex;align-items:center;gap:8px;min-width:0;">
      <img alt="Sarah Jenkins" style="width:24px;height:24px;border-radius:50%;object-fit:cover;border:1px solid #E5E7EB;" src="https://images.unsplash.com/photo-1494790108377-be9c29b29330?auto=format&fit=crop&w=100&q=80"/>
      <span style="font-size:12px;font-weight:500;color:#1F2937;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">Sarah Jenkins</span>
    </div>
    <span class="material-symbols-outlined" style="font-size:16px;color:#9CA3AF;">unfold_more</span>
  </div>
</div>"""
        render_html(sidebar_bottom_html)

    # --- TOP HEADER BAR WITH STITCH RUN/STOP CONTROLS ---
    subtitles = {
        "Overview": "Coffee Shop Monitoring",
        "Monitoring": "Real-time CV Detection & Tracking Overlays",
        "Seating": "Floor Plan & Capacity",
        "Analytics": "Historical Telemetry & Reports",
        "Settings": "CV Pipeline & Model Parameters"
    }

    hdr_left, hdr_right = st.columns([6, 6], gap="medium")
    with hdr_left:
        bread_html = f"""<div style="display:flex;align-items:center;gap:8px;padding:6px 0 12px 0;">
  <span style="font-size:15px;font-weight:600;color:#111827;letter-spacing:-0.02em;">{selected_nav}</span>
  <span style="color:#D1D5DB;font-size:14px;">/</span>
  <span style="font-size:13px;color:#6B7280;font-weight:400;">{subtitles.get(selected_nav, 'Coffee Shop Monitoring')}</span>
</div>"""
        render_html(bread_html)

    with hdr_right:
        c_status, c_select, c_btn = st.columns([2.6, 3.6, 2.6], gap="small")
        with c_status:
            st_color = "#16A34A" if st.session_state.is_running else "#9CA3AF"
            st_bg = "#F0FDF4" if st.session_state.is_running else "#F9FAFB"
            st_border = "#DCFCE7" if st.session_state.is_running else "#E5E7EB"
            st_text = "#15803D" if st.session_state.is_running else "#6B7280"
            st_lbl = "Analysis Running" if st.session_state.is_running else "Engine Ready"
            pill_html = f"""<div style="padding:6px 12px;border-radius:9999px;background:{st_bg};border:1px solid {st_border};color:{st_text};font-size:11px;font-weight:500;display:inline-flex;align-items:center;gap:6px;margin-top:2px;">
  <span style="width:6px;height:6px;border-radius:50%;background:{st_color};"></span>
  <span>{st_lbl}</span>
</div>"""
            render_html(pill_html)

        with c_select:
            available_videos = []
            if Path("data/videos").exists():
                available_videos = [f.name for f in Path("data/videos").glob("*.mp4")]
            if not available_videos:
                available_videos = ["kopisop.mp4"]
            cur_vname = Path(st.session_state.video_source).name if st.session_state.video_source else available_videos[0]
            def_idx = available_videos.index(cur_vname) if cur_vname in available_videos else 0
            sel_video_name = st.selectbox("Sample Video", available_videos, index=def_idx, label_visibility="collapsed")
            new_v_source = str(Path("data/videos") / sel_video_name)
            if new_v_source != st.session_state.video_source:
                st.session_state.video_source = new_v_source
                st.session_state.latest_frame = None
                st.session_state.pipeline = None
                st.session_state.history_metrics = []
                st.rerun()

        with c_btn:
            if st.session_state.is_running:
                stop_btn = st.button("⏹ Stop Analysis", use_container_width=True, key="hdr_stop")
                if stop_btn:
                    st.session_state.is_running = False
                    st.rerun()
                start_btn = False
            else:
                hdr_run = st.button("▶ Run Analysis", use_container_width=True, type="primary", key="hdr_run")
                trigger_run = st.session_state.get("trigger_run_analysis", False)
                if trigger_run:
                    st.session_state.trigger_run_analysis = False
                start_btn = hdr_run or trigger_run

    # --- VIDEO PROCESSING PIPELINE LOOP ---
    if start_btn:
        v_source = st.session_state.video_source
        if not v_source or not Path(v_source).exists():
            st.error("❌ Video source not found. Please upload or choose a video.")
            return

        cfg = PipelineConfig(
            video_source=v_source,
            model_weights=st.session_state.config_model_weights,
            roi_config_path=st.session_state.config_roi_path,
            tracker_type=st.session_state.config_tracker_type,
            conf_threshold=float(st.session_state.config_conf_threshold),
            imgsz=int(st.session_state.config_imgsz),
            duration_limit_seconds=float(st.session_state.config_duration_limit),
            grace_period_seconds=float(st.session_state.config_grace_period)
        )

        try:
            with st.spinner("Initializing YOLOv8 & ByteTrack pipeline..."):
                pipeline = CoffeeShopPipeline(cfg)
                st.session_state.pipeline = pipeline
        except Exception as e:
            st.error(f"Failed to initialize pipeline: {e}")
            return

        cap = cv2.VideoCapture(v_source)
        if not cap.isOpened():
            st.error(f"Failed to open video source: {v_source}")
            return

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        if h_frame >= 2160 or w_frame >= 3840:
            res_str = "4K"
        elif h_frame >= 1440 or w_frame >= 2560:
            res_str = "1440p"
        elif h_frame >= 1000 or w_frame >= 1900:
            res_str = "1080p"
        elif h_frame >= 700 or w_frame >= 1200:
            res_str = "720p"
        elif h_frame >= 460:
            res_str = "480p"
        elif h_frame >= 340:
            res_str = "360p"
        else:
            res_str = f"{h_frame}p"
        total_frames_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 240)
        total_video_dur = total_frames_cnt / fps
        tot_m = int(total_video_dur // 60)
        tot_s = int(total_video_dur % 60)
        tot_time_str = f"{tot_m:02d}:{tot_s:02d}"

        frame_idx = 0
        frame_step = 2
        max_frames = total_frames_cnt

        st.session_state.is_running = True
        st.session_state.history_metrics = []
        events = []
        prev_table_states: Dict[str, AvailabilityState] = {}

        # Live streaming layout placeholders matching selected view
        kpi_box = st.empty()
        col_v, col_t = st.columns([8, 4], gap="medium")
        with col_v:
            render_video_feed_header(
                cam_name="Ground Floor Central (CAM-04)",
                fps=fps,
                res=res_str
            )
            video_box = st.empty()
            footer_box = st.empty()
        with col_t:
            table_box = st.empty()

        while cap.isOpened() and frame_idx < max_frames and st.session_state.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / fps
            processed: ProcessedFrame = pipeline.process_frame(
                frame, frame_idx, timestamp, draw_annotations=True
            )
            st.session_state.latest_frame = processed

            m: OccupancyMetrics = processed.metrics
            cust_count = processed.active_customers_count
            st.session_state.history_metrics.append({
                "frame": frame_idx,
                "timestamp": round(timestamp, 2),
                "occupancy_rate": m.seat_occupancy_rate,
                "effective_occupancy_rate": m.effective_seat_occupancy_rate,
                "occupied_seats": m.occupied_seats,
                "available_seats": m.available_seats,
                "active_customers": cust_count,
                "seating_pressure": m.seating_pressure,
                "pressure_score": m.pressure_score
            })

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
                    events.insert(0, f"⚠️ [{time_str}] Table {tid} is OVERTIME (>60s)")
                prev_table_states[tid] = curr_state

            st.session_state.events = events

            show_ov = st.session_state.get("show_overlays", True)
            frame_stream = processed.annotated_frame if (show_ov and processed.annotated_frame is not None) else processed.raw_frame
            rgb = cv2.cvtColor(frame_stream, cv2.COLOR_BGR2RGB)
            video_box.image(rgb, channels="RGB", use_container_width=True)

            if kpi_box and processed.metrics:
                kpi_box.markdown(
                    clean_html_str(get_overview_kpis_html(processed.metrics, cust_count)),
                    unsafe_allow_html=True
                )
            if table_box:
                table_box.markdown(
                    clean_html_str(get_compact_table_list_html(
                        processed.table_states,
                        pipeline.layout,
                        st.session_state.selected_table_id
                    )),
                    unsafe_allow_html=True
                )
            if footer_box:
                c_min = int(timestamp // 60)
                c_sec = int(timestamp % 60)
                footer_box.markdown(
                    clean_html_str(get_video_feed_footer_html(f"{c_min:02d}:{c_sec:02d}", tot_time_str)),
                    unsafe_allow_html=True
                )

            frame_idx += 1

        cap.release()
        st.session_state.is_running = False
        st.rerun()

    # --- RENDER ACTIVE PAGE VIEW ---
    latest_frame = st.session_state.latest_frame
    pipeline = st.session_state.pipeline
    is_running = st.session_state.is_running
    events = st.session_state.events

    if selected_nav == "Overview":
        render_overview_view(latest_frame, pipeline, is_running, events)

    elif selected_nav == "Monitoring":
        render_monitoring_view(latest_frame, pipeline, is_running)

    elif selected_nav == "Seating":
        render_seating_view(latest_frame, pipeline)

    elif selected_nav == "Analytics":
        render_analytics_view(st.session_state.history_metrics, pipeline)

    elif selected_nav == "Settings":
        render_settings_view()


if __name__ == "__main__":
    main()

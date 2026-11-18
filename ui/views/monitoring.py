"""Live Monitoring View - Detailed CV Telemetry & Overlays."""
from typing import Dict, List, Optional
import cv2
from pathlib import Path
import pandas as pd
import streamlit as st

from src.pipeline import CoffeeShopPipeline, ProcessedFrame
from ui.components import (
    get_video_stream_info,
    render_legend_footer,
    render_overview_kpis,
    render_video_feed_header,
)
from ui.styles import render_html


def render_monitoring_view(
    latest_frame: Optional[ProcessedFrame],
    pipeline: Optional[CoffeeShopPipeline],
    is_running: bool
):
    """Render the full Live Monitoring view matching Stitch specifications with 100% real dynamic data."""
    metrics = latest_frame.metrics if latest_frame else None
    active_customers = latest_frame.active_customers_count if latest_frame else 0
    render_overview_kpis(metrics=metrics, active_customers=active_customers)

    col_video, col_telemetry = st.columns([7, 5], gap="medium")

    with col_video:
        frame_idx = latest_frame.frame_idx if latest_frame else 0
        v_time = latest_frame.timestamp if latest_frame else 0.0

        v_source = st.session_state.get("video_source", "data/videos/kopisop.mp4")
        res_tag, fps_val = get_video_stream_info(v_source)

        render_video_feed_header(
            cam_name="Ground Floor Central (CAM-04)",
            fps=fps_val,
            res=res_tag
        )

        if latest_frame and latest_frame.annotated_frame is not None:
            rgb_frame = cv2.cvtColor(latest_frame.annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, channels="RGB", use_container_width=True)
        else:
            sample_img = "data/videos/coffee_shop.png" if Path("data/videos/coffee_shop.png").exists() else "coffee_shop.png"
            if Path(sample_img).exists():
                st.image(sample_img, use_container_width=True)
            else:
                st.image("https://images.unsplash.com/photo-1554118811-1e0d58224f24?auto=format&fit=crop&w=1200&q=80", use_container_width=True)

        render_legend_footer()

    with col_telemetry:
        # Card 1: Live CV Detections
        with st.container(border=True):
            det_cnt = len(latest_frame.detections) if latest_frame else 0
            render_html(f"""<div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #F3F4F6;padding-bottom:10px;margin-bottom:12px;">
  <span style="font-size:13px;font-weight:600;color:#111827;">Live Computer Vision Detections</span>
  <span style="font-size:11px;color:#2563EB;font-family:'JetBrains Mono',monospace;font-weight:600;">{det_cnt} Detections (YOLOv8)</span>
</div>""")

            if latest_frame and latest_frame.detections:
                det_rows = []
                for d in latest_frame.detections:
                    x1, y1, x2, y2 = d.xyxy
                    det_rows.append({
                        "Class": d.cls_name,
                        "Category": d.category.capitalize(),
                        "Confidence": f"{d.conf:.2f}",
                        "Bounding Box (xyxy)": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                    })
                df_det = pd.DataFrame(det_rows)
                st.dataframe(df_det, use_container_width=True, hide_index=True, height=210)
            else:
                st.caption("No detections on current frame. Run analysis to stream live detections.")

        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

        # Card 2: Tracked Persons (ByteTrack)
        with st.container(border=True):
            track_cnt = len(latest_frame.tracked_persons) if latest_frame else 0
            render_html(f"""<div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #F3F4F6;padding-bottom:10px;margin-bottom:12px;">
  <span style="font-size:13px;font-weight:600;color:#111827;">Tracked Persons & Roles</span>
  <span style="font-size:11px;color:#16A34A;font-family:'JetBrains Mono',monospace;font-weight:600;">{track_cnt} Active Tracks</span>
</div>""")

            if latest_frame and latest_frame.tracked_persons:
                track_rows = []
                for p in latest_frame.tracked_persons:
                    dur = latest_frame.person_visibility_durations.get(p.track_id, 0.0)
                    is_staff = getattr(p, "is_staff", False)
                    role = "Barista (Staff)" if is_staff else "Customer"

                    table_str = "None (Standing/Ordering)"
                    if latest_frame.spatial_association:
                        for tid, pids in latest_frame.spatial_association.table_persons.items():
                            if p.track_id in pids:
                                table_str = f"Table {tid}"
                                break

                    track_rows.append({
                        "Track ID": f"#{p.track_id}",
                        "Role": role,
                        "Associated Table": table_str,
                        "Confidence": f"{p.conf:.2f}",
                        "Dwell Time": f"{dur:.1f}s"
                    })
                df_track = pd.DataFrame(track_rows)
                st.dataframe(df_track, use_container_width=True, hide_index=True, height=210)
            else:
                st.caption("No active tracked persons.")

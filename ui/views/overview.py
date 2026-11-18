"""Overview View - Exact match to Stitch Image 2 with 100% functional interactive controls."""
import time
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import streamlit as st

from src.pipeline import CoffeeShopPipeline, ProcessedFrame
from ui.components import (
    get_video_stream_info,
    render_compact_table_list,
    render_legend_footer,
    render_overview_kpis,
    render_video_feed_header,
)


def render_overview_view(
    latest_frame: Optional[ProcessedFrame],
    pipeline: Optional[CoffeeShopPipeline],
    is_running: bool,
    events: List[str]
):
    """Render the Overview screen matching Stitch Image 2 with fully working controls."""
    metrics = latest_frame.metrics if latest_frame else None
    # Real customer count excluding Baristas/Staff
    active_customers = latest_frame.active_customers_count if latest_frame else 0

    # 4 Balanced KPI Cards
    render_overview_kpis(metrics=metrics, active_customers=active_customers)

    is_focused = st.session_state.get("is_focused", False)

    # Main Grid: Left Video Hero (8 cols or 12 cols if focused), Right Compact Table List
    if is_focused:
        col_video = st.container()
        col_table = None
    else:
        col_video, col_table = st.columns([8, 4], gap="medium")

    with col_video:
        v_source = st.session_state.get("video_source", "data/videos/kopisop.mp4")
        res_tag, fps_val = get_video_stream_info(v_source)

        # Video Header with Clickable Icon Buttons (layers, camera, fullscreen)
        st.markdown(
            '<div class="stitch-table-header" style="border:1px solid rgba(229,231,235,0.85);border-bottom:none;border-radius:12px 12px 0 0;background:#FFFFFF;padding:8px 16px;display:flex;align-items:center;justify-content:space-between;">',
            unsafe_allow_html=True
        )
        h_left, h_right = st.columns([8.2, 1.8], gap="small")
        with h_left:
            render_video_feed_header(
                cam_name="Ground Floor Central (CAM-04)",
                fps=fps_val,
                res=res_tag
            )
        with h_right:
            c_layers, c_cam, c_full = st.columns([1, 1, 1], gap="small")
            show_ov = st.session_state.get("show_overlays", True)
            if c_layers.button("⧉", key="hdr_ic_layers", help="Toggle AI Overlays (Bounding Boxes & Tables)"):
                st.session_state.show_overlays = not show_ov
                st.rerun()
            if c_cam.button("📷", key="hdr_ic_camera", help="Take High-Res Snapshot"):
                st.session_state.take_snapshot = True
                st.rerun()
            if c_full.button("⛶", key="hdr_ic_expand", help="Toggle Focus / Expanded View"):
                st.session_state.is_focused = not is_focused
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Video stream box
        video_placeholder = st.empty()
        if latest_frame is not None:
            raw_bgr = latest_frame.annotated_frame if (show_ov and latest_frame.annotated_frame is not None) else latest_frame.raw_frame
            rgb_frame = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        else:
            sample_img = "data/videos/coffee_shop.png" if Path("data/videos/coffee_shop.png").exists() else "coffee_shop.png"
            if Path(sample_img).exists():
                video_placeholder.image(sample_img, use_container_width=True)
            else:
                video_placeholder.image("https://images.unsplash.com/photo-1554118811-1e0d58224f24?auto=format&fit=crop&w=1200&q=80", use_container_width=True)

        # Snapshot Download Action (if triggered)
        if st.session_state.get("take_snapshot") and latest_frame is not None:
            target_bgr = latest_frame.annotated_frame if show_ov else latest_frame.raw_frame
            ok, buf = cv2.imencode(".png", target_bgr)
            if ok:
                st.download_button(
                    label="💾 Download Captured Frame (PNG)",
                    data=buf.tobytes(),
                    file_name=f"seatvision_cam04_{int(time.time())}.png",
                    mime="image/png",
                    use_container_width=True
                )
                st.session_state.take_snapshot = False

        # Clean Legend Footer directly beneath the video
        render_legend_footer()

    if col_table is not None:
        with col_table:
            table_states = latest_frame.table_states if latest_frame else {}
            layout = pipeline.layout if pipeline else None

            if "selected_table_id" not in st.session_state:
                st.session_state.selected_table_id = "T1"

            render_compact_table_list(
                table_states=table_states,
                layout=layout,
                selected_id=st.session_state.selected_table_id
            )

            # Quick selector pill buttons
            t_keys = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
            sel_cols = st.columns(7)
            for i, tid in enumerate(t_keys):
                if sel_cols[i].button(tid, key=f"ov_sel_{tid}", use_container_width=True):
                    st.session_state.selected_table_id = tid
                    st.rerun()

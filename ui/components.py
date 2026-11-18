"""Reusable UI Components implementing exact Google Stitch visual elements."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import streamlit as st

from src.metrics import OccupancyMetrics
from src.spatial import SeatingLayout
from src.state_machine import AvailabilityState, DurationWarningState, RegionState
from ui.styles import render_html


def render_top_header(
    title: str = "Overview",
    subtitle: str = "Coffee Shop Monitoring",
    is_running: bool = False,
    video_name: str = "kopisop.mp4"
):
    """Render top breadcrumb header matching Stitch Images 2-5."""
    render_top_header_banner(title, subtitle, is_running, video_name)


def render_top_header_banner(
    title: str = "Overview",
    subtitle: str = "Coffee Shop Monitoring",
    is_running: bool = False,
    video_name: str = "kopisop.mp4"
):
    """Render top breadcrumb header matching Stitch Images 2-5."""
    status_cls = "background:#F0FDF4;border:1px solid #BBF7D0;color:#15803D;" if is_running else "background:#F9FAFB;border:1px solid #E5E7EB;color:#6B7280;"
    dot_color = "#16A34A" if is_running else "#9CA3AF"
    status_label = "Analysis Running" if is_running else "Engine Ready"

    html = f"""<div style="display:flex;align-items:center;justify-content:space-between;padding:12px 0 20px 0;border-bottom:1px solid #F3F4F6;margin-bottom:20px;">
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:15px;font-weight:600;color:#111827;letter-spacing:-0.02em;">{title}</span>
    <span style="color:#D1D5DB;font-size:14px;">/</span>
    <span style="font-size:13px;color:#6B7280;font-weight:400;">{subtitle}</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <div style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:9999px;font-size:11px;font-weight:500;{status_cls}">
      <span style="width:6px;height:6px;border-radius:50%;background:{dot_color};"></span>
      <span>{status_label}</span>
    </div>
    <div style="display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:6px;border:1px solid #E5E7EB;background:#FFFFFF;font-size:12px;color:#374151;font-weight:500;">
      <span class="material-symbols-outlined" style="font-size:15px;color:#9CA3AF;">videocam</span>
      <span>Sample: {video_name}</span>
    </div>
  </div>
</div>"""
    render_html(html)


def get_overview_kpis_html(metrics: Optional[OccupancyMetrics], active_customers: int = 0) -> str:
    """Build the 4 KPI cards HTML matching Stitch Image 2."""
    if metrics:
        occ_rate = f"{metrics.seat_occupancy_rate:.1f}%"
        delta_eff = metrics.effective_seat_occupancy_rate - metrics.seat_occupancy_rate
        eff_str = f"+{delta_eff:.1f}%" if delta_eff > 0 else f"{metrics.effective_seat_occupancy_rate:.1f}%"
        avail_str = f"{metrics.available_seats} <span style='font-size:15px;color:#9CA3AF;font-weight:400;'>/ {metrics.total_seats}</span>"
        cap_pct = int((metrics.occupied_seats / max(metrics.total_seats, 1)) * 100)
        cap_sub = f"{cap_pct}% capacity"
        cust_str = str(active_customers)
        cust_sub = f"Across {metrics.occupied_tables} active tables"

        p_level = metrics.seating_pressure
        if p_level == "HIGH":
            p_badge = '<span style="display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:9999px;background:#FEF2F2;border:1px solid #FECACA;color:#B91C1C;font-size:12px;font-weight:600;"><span style="width:6px;height:6px;border-radius:50%;background:#DC2626;"></span>HIGH</span>'
            p_sub = "Critical capacity reached"
        elif p_level == "MEDIUM":
            p_badge = '<span style="display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:9999px;background:#FFFBEB;border:1px solid #FDE68A;color:#B45309;font-size:12px;font-weight:600;"><span style="width:6px;height:6px;border-radius:50%;background:#D97706;"></span>MEDIUM</span>'
            p_sub = "Walk-ins accommodated"
        else:
            p_badge = '<span style="display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:9999px;background:#F0FDF4;border:1px solid #BBF7D0;color:#15803D;font-size:12px;font-weight:600;"><span style="width:6px;height:6px;border-radius:50%;background:#16A34A;"></span>LOW</span>'
            p_sub = "Ample seating available"
    else:
        occ_rate = "38.5%"
        eff_str = "4.2%"
        avail_str = "5 <span style='font-size:15px;color:#9CA3AF;font-weight:400;'>/ 13</span>"
        cap_sub = "38% capacity"
        cust_str = "8"
        cust_sub = "Across 4 active tables"
        p_badge = '<span style="display:inline-flex;align-items:center;gap:6px;padding:2px 8px;border-radius:9999px;background:#FFFBEB;border:1px solid #FDE68A;color:#B45309;font-size:12px;font-weight:600;"><span style="width:6px;height:6px;border-radius:50%;background:#D97706;"></span>MEDIUM</span>'
        p_sub = "Walk-ins accommodated"

    return f"""<div class="stitch-kpi-grid">
  <div class="stitch-kpi-card">
    <div class="stitch-kpi-header">
      <span>Occupancy Rate</span>
      <span class="material-symbols-outlined stitch-kpi-icon">pie_chart</span>
    </div>
    <div class="stitch-kpi-body">
      <span class="stitch-kpi-val">{occ_rate}</span>
      <span class="stitch-kpi-delta-up">
        <span class="material-symbols-outlined" style="font-size:14px;">arrow_upward</span>{eff_str}
      </span>
    </div>
  </div>

  <div class="stitch-kpi-card">
    <div class="stitch-kpi-header">
      <span>Available Seats</span>
      <span class="material-symbols-outlined stitch-kpi-icon">event_seat</span>
    </div>
    <div class="stitch-kpi-body">
      <span class="stitch-kpi-val">{avail_str}</span>
      <span class="stitch-kpi-sub" style="font-family:'JetBrains Mono',monospace;">{cap_sub}</span>
    </div>
  </div>

  <div class="stitch-kpi-card">
    <div class="stitch-kpi-header">
      <span>Active Customers</span>
      <span class="material-symbols-outlined stitch-kpi-icon">person</span>
    </div>
    <div class="stitch-kpi-body">
      <span class="stitch-kpi-val">{cust_str}</span>
      <span class="stitch-kpi-sub">{cust_sub}</span>
    </div>
  </div>

  <div class="stitch-kpi-card">
    <div class="stitch-kpi-header">
      <span>Seating Pressure</span>
      <span class="material-symbols-outlined stitch-kpi-icon">speed</span>
    </div>
    <div class="stitch-kpi-body" style="align-items:center;">
      {p_badge}
      <span class="stitch-kpi-sub" style="font-size:11px;">{p_sub}</span>
    </div>
  </div>
</div>"""


def render_overview_kpis(metrics: Optional[OccupancyMetrics], active_customers: int = 0):
    """Render the 4 KPI cards matching Stitch Image 2."""
    html = get_overview_kpis_html(metrics, active_customers)
    render_html(html)


def get_video_duration(video_path: Optional[str]) -> float:
    """Extract real video duration in seconds directly from the media file."""
    if not video_path:
        return 10.0
    p = Path(video_path)
    if not p.exists():
        return 10.0
    try:
        import cv2
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 240.0
        cap.release()
        return max(1.0, cnt / fps)
    except Exception:
        return 10.0


def get_video_stream_info(video_path: Optional[str]) -> Tuple[str, float]:
    """Extract real video resolution tag (e.g. '720p', '1080p', '480p') and FPS from video file."""
    if not video_path:
        return "720p", 24.0
    p = Path(video_path)
    if not p.exists():
        return "720p", 24.0
    try:
        import cv2
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            return "720p", 24.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 24.0)
        cap.release()

        if h >= 2160 or w >= 3840:
            res_tag = "4K"
        elif h >= 1440 or w >= 2560:
            res_tag = "1440p"
        elif h >= 1000 or w >= 1900:
            res_tag = "1080p"
        elif h >= 700 or w >= 1200:
            res_tag = "720p"
        elif h >= 460:
            res_tag = "480p"
        elif h >= 340:
            res_tag = "360p"
        else:
            res_tag = f"{h}p"
        return res_tag, round(fps, 1)
    except Exception:
        return "720p", 24.0


def render_video_feed_header(cam_name="Ground Floor Central (CAM-04)", fps=24.0, res="720p"):
    """Render top left badge of the video player hero matching Stitch Image 2."""
    html = f"""<div style="display:flex;align-items:center;gap:8px;padding:2px 0;">
  <span style="width:7px;height:7px;border-radius:50%;background:#10B981;display:inline-block;"></span>
  <span style="font-size:13px;font-weight:600;color:#111827;">{cam_name}</span>
  <span style="color:#D1D5DB;">·</span>
  <span style="font-size:12px;color:#9CA3AF;font-family:'JetBrains Mono',monospace;">{res} · {fps:.1f} FPS</span>
</div>"""
    render_html(html)


def render_legend_footer():
    """Render video player legend beneath the playback bar matching Stitch Image 2."""
    html = """<div style="padding:8px 16px;background:#FAFAFA;border:1px solid rgba(229,231,235,0.85);border-top:1px solid #F3F4F6;border-radius:0 0 12px 12px;display:flex;align-items:center;justify-content:space-between;font-size:11px;color:#6B7280;">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <span style="display:inline-flex;align-items:center;gap:5px;"><span style="width:7px;height:7px;border-radius:50%;background:#16A34A;"></span>Available</span>
    <span style="display:inline-flex;align-items:center;gap:5px;"><span style="width:7px;height:7px;border-radius:50%;background:#DC2626;"></span>Occupied</span>
    <span style="display:inline-flex;align-items:center;gap:5px;"><span style="width:7px;height:7px;border-radius:50%;background:#D97706;"></span>Reserved</span>
    <span style="display:inline-flex;align-items:center;gap:5px;"><span style="width:7px;height:7px;border-radius:50%;background:#2563EB;"></span>Customer</span>
    <span style="display:inline-flex;align-items:center;gap:5px;"><span style="width:7px;height:7px;border-radius:50%;background:#8B5CF6;"></span>Barista (Staff)</span>
  </div>
  <span style="font-family:'JetBrains Mono',monospace;color:#9CA3AF;">ByteTrack YOLOv8</span>
</div>"""
    render_html(html)


def get_video_feed_footer_html(current_time="00:00", total_time="00:10") -> str:
    """Generate video player controls and legend HTML matching Stitch Image 2."""
    return f"""<div style="padding:8px 16px;background:#FAFAFA;border:1px solid rgba(229,231,235,0.85);border-top:1px solid #F3F4F6;border-radius:0 0 12px 12px;display:flex;align-items:center;justify-content:space-between;font-size:11px;color:#6B7280;">
  <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
    <span style="display:inline-flex;align-items:center;gap:5px;"><span style="width:7px;height:7px;border-radius:50%;background:#16A34A;"></span>Available</span>
    <span style="display:inline-flex;align-items:center;gap:5px;"><span style="width:7px;height:7px;border-radius:50%;background:#DC2626;"></span>Occupied</span>
    <span style="display:inline-flex;align-items:center;gap:5px;"><span style="width:7px;height:7px;border-radius:50%;background:#D97706;"></span>Reserved</span>
    <span style="display:inline-flex;align-items:center;gap:5px;"><span style="width:7px;height:7px;border-radius:50%;background:#2563EB;"></span>Customer</span>
    <span style="display:inline-flex;align-items:center;gap:5px;"><span style="width:7px;height:7px;border-radius:50%;background:#8B5CF6;"></span>Barista (Staff)</span>
  </div>
  <span style="font-family:'JetBrains Mono',monospace;color:#9CA3AF;">ByteTrack YOLOv8</span>
</div>"""


def render_video_feed_footer(current_time="00:00", total_time="00:10"):
    """Render video player controls and legend matching Stitch Image 2."""
    html = get_video_feed_footer_html(current_time, total_time)
    render_html(html)


def get_compact_table_list_html(
    table_states: Dict[str, RegionState],
    layout: Optional[SeatingLayout] = None,
    selected_id: str = "T1"
) -> str:
    """Generate compact table list & inspector card HTML matching Stitch Image 2 right column."""
    default_tables = [
        {"id": "T1", "type": "2 Seats", "status": "Occupied", "color": "#DC2626", "dot": "#DC2626", "dur": "04:32", "persons": "Person #1, Person #4", "items": "None"},
        {"id": "T2", "type": "2 Seats", "status": "Occupied", "color": "#DC2626", "dot": "#DC2626", "dur": "03:18", "persons": "Person #3", "items": "None"},
        {"id": "T3", "type": "Booth", "status": "Limit Alert", "color": "#D97706", "dot": "#D97706", "dur": "52:14", "persons": "Person #2, Person #8", "items": "Laptop, Backpack"},
        {"id": "T4", "type": "Bistro", "status": "Available", "color": "#16A34A", "dot": "#16A34A", "dur": "00:00", "persons": "None", "items": "None"},
        {"id": "T5", "type": "Lounge", "status": "Reserved", "color": "#D97706", "dot": "#D97706", "dur": "14:20", "persons": "Unattended", "items": "MacBook Pro"},
        {"id": "T6", "type": "Bar", "status": "Available", "color": "#16A34A", "dot": "#16A34A", "dur": "00:00", "persons": "None", "items": "None"},
        {"id": "T7", "type": "Patio", "status": "Overtime", "color": "#DC2626", "dot": "#DC2626", "dur": "72:18", "persons": "Person #5", "items": "None"},
    ]

    items_to_render = []
    if table_states:
        for tid, ts in table_states.items():
            t_obj = layout.tables.get(tid) if layout else None
            s_cnt = len(t_obj.seats) if t_obj else 2
            dur_sec = ts.occupancy_duration
            dur_str = f"{int(dur_sec//60):02d}:{int(dur_sec%60):02d}"

            if ts.duration_warning_state == DurationWarningState.OVERTIME:
                st_label, color, dot = "Overtime", "#DC2626", "#DC2626"
            elif ts.duration_warning_state == DurationWarningState.APPROACHING_LIMIT:
                st_label, color, dot = "Limit Alert", "#D97706", "#D97706"
            elif ts.availability_state == AvailabilityState.OCCUPIED:
                st_label, color, dot = "Occupied", "#DC2626", "#DC2626"
            elif ts.availability_state == AvailabilityState.RESERVED_BY_BELONGINGS:
                st_label, color, dot = "Reserved", "#D97706", "#D97706"
            else:
                st_label, color, dot = "Available", "#16A34A", "#16A34A"

            p_str = ", ".join(f"Person #{pid}" for pid in ts.active_person_ids) if ts.active_person_ids else "None"
            b_str = f"{ts.active_belongings_count} Item(s)" if ts.active_belongings_count > 0 else "None"

            items_to_render.append({
                "id": tid,
                "type": f"{s_cnt} Seats",
                "status": st_label,
                "color": color,
                "dot": dot,
                "dur": dur_str,
                "persons": p_str,
                "items": b_str
            })
    else:
        items_to_render = default_tables

    rows_html = []
    for item in items_to_render:
        is_sel = item["id"] == selected_id
        bg_style = "background:#F3F4F6;" if is_sel else "background:#FFFFFF;"
        rows_html.append(
            f"""<div class="stitch-table-row" style="{bg_style}">
  <div style="display:flex;align-items:center;gap:8px;">
    <span style="font-weight:600;color:#111827;">{item['id']}</span>
    <span style="color:#9CA3AF;font-size:11px;">{item['type']}</span>
  </div>
  <div style="display:flex;align-items:center;gap:12px;">
    <span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:500;color:{item['color']};">
      <span style="width:6px;height:6px;border-radius:50%;background:{item['dot']};"></span>
      {item['status']}
    </span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#6B7280;width:40px;text-align:right;">{item['dur']}</span>
  </div>
</div>"""
        )

    all_rows = "".join(rows_html)
    active_inspect = next((it for it in items_to_render if it["id"] == selected_id), items_to_render[0])

    return f"""<div class="stitch-table-card">
  <div class="stitch-table-header">
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="font-size:13px;font-weight:600;color:#111827;">Tables</span>
      <span style="font-size:11px;color:#9CA3AF;font-family:'JetBrains Mono',monospace;">{len(items_to_render)} Total</span>
    </div>
    <span style="font-size:11px;color:#9CA3AF;font-family:'JetBrains Mono',monospace;">Click to inspect</span>
  </div>

  <div style="display:flex;flex-direction:column;">
    {all_rows}
  </div>

  <div style="padding:16px;background:#FAFAFA;border-top:1px solid #F3F4F6;display:flex;flex-direction:column;gap:10px;">
    <div style="display:flex;align-items:center;justify-content:space-between;">
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:11px;font-weight:600;color:#6B7280;letter-spacing:0.04em;text-transform:uppercase;">INSPECTION</span>
        <span style="font-size:12px;font-weight:600;color:#111827;padding:1px 6px;background:#FFFFFF;border:1px solid #E5E7EB;border-radius:4px;">{active_inspect['id']}</span>
      </div>
      <span style="font-size:11px;font-weight:600;color:{active_inspect['color']};">{active_inspect['status']}</span>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:11px;padding-top:4px;">
      <div>
        <span style="color:#9CA3AF;display:block;">Configuration</span>
        <span style="color:#1F2937;font-weight:500;">{active_inspect['type']}</span>
      </div>
      <div>
        <span style="color:#9CA3AF;display:block;">Dwell Time</span>
        <span style="color:#1F2937;font-family:'JetBrains Mono',monospace;font-weight:500;">{active_inspect['dur']}</span>
      </div>
      <div>
        <span style="color:#9CA3AF;display:block;">Detected Patrons</span>
        <span style="color:#1F2937;font-weight:500;">{active_inspect['persons']}</span>
      </div>
      <div>
        <span style="color:#9CA3AF;display:block;">Unattended Items</span>
        <span style="color:#1F2937;font-weight:500;">{active_inspect['items']}</span>
      </div>
    </div>
  </div>
</div>"""


def render_compact_table_list(
    table_states: Dict[str, RegionState],
    layout: Optional[SeatingLayout] = None,
    selected_id: str = "T1"
):
    """Render compact table list & inspector card matching Stitch Image 2 right column."""
    html = get_compact_table_list_html(table_states, layout, selected_id)
    render_html(html)

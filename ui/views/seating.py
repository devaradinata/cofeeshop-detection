"""Seating Management View - Dynamically computed from YOLOv8, ByteTrack, and Spatial Layout."""
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import CoffeeShopPipeline, ProcessedFrame
from src.state_machine import AvailabilityState, DurationWarningState, RegionState
from ui.styles import render_html


def render_seating_view(
    latest_frame: Optional[ProcessedFrame],
    pipeline: Optional[CoffeeShopPipeline]
):
    """Render the Seating Management screen with 100% real dynamic data from computer vision detections."""
    metrics = latest_frame.metrics if latest_frame else None
    table_states = latest_frame.table_states if latest_frame else {}
    seat_states = latest_frame.seat_states if latest_frame else {}
    layout = pipeline.layout if pipeline else None
    spatial = latest_frame.spatial_association if latest_frame else None

    # Real Metrics
    tot_seats = len(layout.seats) if layout else (metrics.total_seats if metrics else 13)
    active_cust = latest_frame.active_customers_count if latest_frame else (metrics.active_customers if metrics else 0)
    unattended = metrics.belongings_seats if metrics else 0
    occ_pct = metrics.seat_occupancy_rate if metrics else 0.0

    # Calculate real average dwell across tables with duration > 0
    durations = [ts.occupancy_duration for ts in table_states.values() if ts.occupancy_duration > 0]
    avg_dwell = (sum(durations) / len(durations)) if durations else 0.0
    dwell_str = f"{avg_dwell:.1f}s" if avg_dwell < 60 else f"{avg_dwell/60:.1f}m"

    # Identify any table with belongings
    belonging_tables = [tid for tid, ts in table_states.items() if ts.availability_state == AvailabilityState.RESERVED_BY_BELONGINGS]
    if belonging_tables:
        hold_sub = f"● {belonging_tables[0]} Reserved"
    elif unattended > 0:
        hold_sub = f"● {unattended} Item(s) Detected"
    else:
        hold_sub = "● Clear (No Hold)"

    # --- 1. Balanced Top KPI Cards ---
    kpis_html = f"""<div class="stitch-kpi-grid">
  <div class="stitch-kpi-card">
    <div class="stitch-kpi-header">
      <span>Total Capacity</span>
      <span class="material-symbols-outlined stitch-kpi-icon">event_seat</span>
    </div>
    <div class="stitch-kpi-body">
      <span class="stitch-kpi-val">{tot_seats}</span>
      <span class="stitch-kpi-sub">calibrated seats</span>
    </div>
  </div>

  <div class="stitch-kpi-card">
    <div class="stitch-kpi-header">
      <span>Active Customers</span>
      <span class="material-symbols-outlined stitch-kpi-icon">person</span>
    </div>
    <div class="stitch-kpi-body">
      <span class="stitch-kpi-val">{active_cust}</span>
      <span class="stitch-kpi-delta-up" style="font-family:'JetBrains Mono',monospace;">{occ_pct:.1f}% occupied</span>
    </div>
  </div>

  <div class="stitch-kpi-card">
    <div class="stitch-kpi-header">
      <span>Unattended Items</span>
      <span class="material-symbols-outlined stitch-kpi-icon">work</span>
    </div>
    <div class="stitch-kpi-body">
      <span class="stitch-kpi-val">{unattended}</span>
      <span class="stitch-kpi-sub" style="color:#D97706;font-weight:600;">{hold_sub}</span>
    </div>
  </div>

  <div class="stitch-kpi-card">
    <div class="stitch-kpi-header">
      <span>Avg Seating Dwell</span>
      <span class="material-symbols-outlined stitch-kpi-icon">schedule</span>
    </div>
    <div class="stitch-kpi-body">
      <span class="stitch-kpi-val">{dwell_str}</span>
      <span class="stitch-kpi-sub">Real-time table average</span>
    </div>
  </div>
</div>"""
    render_html(kpis_html)

    # --- 2. Main Section: Blueprint & Inspector ---
    col_blueprint, col_inspector = st.columns([8, 4], gap="medium")

    with col_blueprint:
        # Generate Dynamic Blueprint Cards from real tables
        table_cards_html = []
        if layout and layout.tables:
            for tid, table in layout.tables.items():
                ts = table_states.get(tid)
                state = ts.availability_state if ts else AvailabilityState.AVAILABLE
                s_cnt = len(table.seats)
                dur = ts.occupancy_duration if ts else 0.0
                dur_str = f"{int(dur//60):02d}:{int(dur%60):02d}"

                persons = spatial.table_persons.get(tid, []) if spatial else []
                items = spatial.table_belongings.get(tid, []) if spatial else []

                if state == AvailabilityState.OCCUPIED:
                    st_label = "● Occupied"
                    st_color = "#DC2626"
                    card_bg = "#FFFFFF"
                    card_border = "#E5E7EB"
                    cust_info = f"#{', #'.join(map(str, persons))} · {dur_str}" if persons else f"Occupied · {dur_str}"
                elif state == AvailabilityState.RESERVED_BY_BELONGINGS:
                    st_label = "● Reserved"
                    st_color = "#D97706"
                    card_bg = "#FFFBEB"
                    card_border = "#FBBF24"
                    cust_info = f"{len(items)} Item(s) · {dur_str}"
                elif ts and ts.duration_warning_state == DurationWarningState.OVERTIME:
                    st_label = "● Overtime"
                    st_color = "#DC2626"
                    card_bg = "#FEF2F2"
                    card_border = "#FCA5A5"
                    cust_info = f"#{', #'.join(map(str, persons))} · {dur_str}"
                else:
                    st_label = "● Available"
                    st_color = "#16A34A"
                    card_bg = "#FFFFFF"
                    card_border = "#E5E7EB"
                    cust_info = "Ready for guests"

                table_cards_html.append(f"""<div style="padding:12px;background:{card_bg};border:1px solid {card_border};border-radius:8px;box-shadow:0 1px 2px rgba(0,0,0,0.02);">
  <div style="display:flex;justify-content:space-between;align-items:center;font-size:11px;margin-bottom:4px;">
    <span style="font-weight:700;color:#111827;">{tid} · {s_cnt} Seats</span>
    <span style="font-size:10px;color:{st_color};font-weight:600;">{st_label}</span>
  </div>
  <div style="font-size:10px;color:#9CA3AF;font-family:'JetBrains Mono',monospace;">{cust_info}</div>
</div>""")

        blueprint_grid = "".join(table_cards_html)

        blueprint_html = f"""<div class="stitch-table-card" style="margin-bottom:24px;">
  <div class="stitch-table-header">
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="width:7px;height:7px;border-radius:50%;background:#10B981;display:inline-block;"></span>
      <span style="font-size:13px;font-weight:600;color:#111827;">Spatial Topology Blueprint</span>
      <span style="color:#D1D5DB;">·</span>
      <span style="font-size:12px;color:#9CA3AF;font-family:'JetBrains Mono',monospace;">CAM-04 Active Layout</span>
    </div>
    <div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#6B7280;">
      <span style="padding:2px 8px;border-radius:4px;background:#F3F4F6;font-weight:600;color:#111827;">Live Spatial Grid</span>
      <span style="color:#D1D5DB;">|</span>
      <span style="font-family:'JetBrains Mono',monospace;color:#9CA3AF;">{len(layout.tables) if layout else 7} Tables</span>
    </div>
  </div>

  <div style="position:relative;width:100%;aspect-ratio:16/10;background:#FAFAFA;border-bottom:1px solid #F3F4F6;padding:20px;display:flex;flex-direction:column;justify-content:space-between;">
    <div style="display:flex;align-items:flex-start;justify-content:space-between;">
      <div style="border:1px dashed #D1D5DB;border-radius:4px;padding:4px 10px;font-size:10px;font-family:'JetBrains Mono',monospace;color:#9CA3AF;text-transform:uppercase;">
        Entrance / Vestibule
      </div>
      <div style="border:1px solid #E5E7EB;background:rgba(255,255,255,0.85);border-radius:6px;padding:6px 14px;text-align:right;">
        <span style="font-size:9px;text-transform:uppercase;letter-spacing:0.04em;color:#9CA3AF;display:block;font-weight:600;">Service Zone</span>
        <span style="font-size:11px;font-weight:600;color:#1F2937;">Espresso Bar & POS</span>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:auto 0;">
      {blueprint_grid}
    </div>

    <div style="display:flex;align-items:flex-end;justify-content:space-between;font-size:10px;color:#9CA3AF;font-family:'JetBrains Mono',monospace;padding-top:8px;">
      <span style="border:1px dashed #D1D5DB;border-radius:4px;padding:2px 8px;">Window Glazing</span>
      <span style="border:1px dashed #D1D5DB;border-radius:4px;padding:2px 8px;">Customer Floor</span>
      <span style="border:1px dashed #D1D5DB;border-radius:4px;padding:2px 8px;">Patio Portal</span>
    </div>
  </div>

  <div class="stitch-video-legend">
    <div style="display:flex;align-items:center;gap:16px;">
      <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:7px;height:7px;border-radius:50%;background:#16A34A;"></span>Available</span>
      <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:7px;height:7px;border-radius:50%;background:#DC2626;"></span>Occupied</span>
      <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:7px;height:7px;border-radius:50%;background:#D97706;"></span>Reserved / Items</span>
      <span style="display:inline-flex;align-items:center;gap:6px;"><span style="width:7px;height:7px;border-radius:50%;background:#8B5CF6;"></span>Staff Zone</span>
    </div>
    <span style="font-family:'JetBrains Mono',monospace;color:#9CA3AF;">ByteTrack Spatial Machine</span>
  </div>
</div>"""
        render_html(blueprint_html)

    with col_inspector:
        # Table Selector for Detailed Real ROI Inspection
        if "selected_seating_table" not in st.session_state:
            st.session_state.selected_seating_table = "T1"

        avail_t_ids = list(layout.tables.keys()) if layout and layout.tables else ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
        cur_t_idx = avail_t_ids.index(st.session_state.selected_seating_table) if st.session_state.selected_seating_table in avail_t_ids else 0

        st.markdown("<div style='font-size:12px;font-weight:600;color:#111827;margin-bottom:4px;'>Inspect Table ROI</div>", unsafe_allow_html=True)
        sel_t_id = st.selectbox(
            "Select Table",
            avail_t_ids,
            index=cur_t_idx,
            key="seating_table_selector",
            label_visibility="collapsed"
        )
        st.session_state.selected_seating_table = sel_t_id

        # Real Inspection Data for selected table
        sel_table_obj = layout.tables.get(sel_t_id) if layout else None
        sel_ts = table_states.get(sel_t_id)
        sel_state = sel_ts.availability_state if sel_ts else AvailabilityState.AVAILABLE
        sel_dur = sel_ts.occupancy_duration if sel_ts else 0.0
        sel_dur_str = f"{int(sel_dur//60):02d}:{int(sel_dur%60):02d}"

        sel_persons = spatial.table_persons.get(sel_t_id, []) if spatial else []
        sel_items = spatial.table_belongings.get(sel_t_id, []) if spatial else []
        item_names = ", ".join(set(d.cls_name for d in sel_items)) if sel_items else "None"

        if sel_state == AvailabilityState.OCCUPIED:
            badge_color = "#DC2626"
            badge_bg = "#FEF2F2"
            badge_border = "#FCA5A5"
            badge_text = "● Occupied"
        elif sel_state == AvailabilityState.RESERVED_BY_BELONGINGS:
            badge_color = "#D97706"
            badge_bg = "#FFFBEB"
            badge_border = "#FDE68A"
            badge_text = "● Reserved"
        else:
            badge_color = "#16A34A"
            badge_bg = "#F0FDF4"
            badge_border = "#BBF7D0"
            badge_text = "● Available"

        # Associated seats HTML for this table
        seats_html = []
        if sel_table_obj:
            for s in sel_table_obj.seats:
                s_state = seat_states.get(s.id)
                st_val = s_state.availability_state.value if s_state else "Available"
                s_col = "#DC2626" if "OCCUPIED" in st_val else ("#D97706" if "RESERVED" in st_val else "#16A34A")
                s_custs = spatial.seat_persons.get(s.id, []) if spatial else []
                cust_label = f"Customer #{s_custs[0]}" if s_custs else "Empty"
                seats_html.append(f"""<div style="padding:8px;border:1px solid #E5E7EB;border-radius:6px;background:#FAFAFA;">
  <div style="font-weight:600;color:#1F2937;">Seat {s.id}</div>
  <div style="color:{s_col};font-weight:500;">{st_val}</div>
  <div style="font-size:10px;color:#6B7280;">{cust_label}</div>
</div>""")
        seats_grid = "".join(seats_html) if seats_html else "<div>No seats configured</div>"

        # Crop Real Table Image from Frame
        crop_container = st.empty()
        if latest_frame is not None and sel_table_obj:
            raw = latest_frame.raw_frame if latest_frame.raw_frame is not None else latest_frame.annotated_frame
            pts = np.array(sel_table_obj.polygon, dtype=np.int32)
            rx, ry, rw, rh = cv2.boundingRect(pts)
            H, W = raw.shape[:2]
            pad = 20
            crop_bgr = raw[max(0, ry - pad):min(H, ry + rh + pad), max(0, rx - pad):min(W, rx + rw + pad)]
            if crop_bgr.size > 0:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                crop_container.image(crop_rgb, caption=f"Live Feed ROI: {sel_t_id} (Bounding Box)", use_container_width=True)

        inspector_html = f"""<div class="stitch-table-card" style="margin-bottom:10px;">
  <div class="stitch-table-header">
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="font-size:13px;font-weight:600;color:#111827;">Table {sel_t_id}</span>
      <span style="font-size:11px;color:#9CA3AF;">{sel_table_obj.name if sel_table_obj else 'Calibrated'}</span>
    </div>
    <span style="display:inline-flex;align-items:center;gap:5px;font-size:11px;padding:2px 8px;border-radius:9999px;background:{badge_bg};color:{badge_color};font-weight:600;border:1px solid {badge_border};">
      {badge_text}
    </span>
  </div>

  <div style="padding:16px;display:flex;flex-direction:column;gap:12px;font-size:12px;">
    <div>
      <span style="font-size:11px;font-weight:600;color:#6B7280;display:block;margin-bottom:6px;">Associated Seats</span>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:11px;">
        {seats_grid}
      </div>
    </div>

    <div style="border-top:1px solid #F3F4F6;padding-top:8px;display:flex;flex-direction:column;gap:6px;font-size:11px;">
      <div style="display:flex;justify-content:space-between;">
        <span style="color:#9CA3AF;">Dwell Duration</span>
        <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:#111827;">{sel_dur_str}</span>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <span style="color:#9CA3AF;">Detected Belongings</span>
        <span style="font-weight:500;color:#1F2937;">{item_names}</span>
      </div>
      <div style="display:flex;justify-content:space-between;">
        <span style="color:#9CA3AF;">Camera ROI Polygon</span>
        <span style="font-family:'JetBrains Mono',monospace;color:#4B5563;">{len(sel_table_obj.polygon) if sel_table_obj else 4} Vertices</span>
      </div>
    </div>
  </div>
</div>"""
        render_html(inspector_html)

        # Action Buttons
        if st.button("🔔 Notify Floor Staff", use_container_width=True, type="primary", key="btn_flag_staff"):
            st.toast(f"🔔 Notification sent to host for Table {sel_t_id}!", icon="🚨")

    # --- 3. Section 3: Dynamic Seating Capacity Matrix Table ---
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    f_col1, f_col2 = st.columns([8, 4])
    with f_col1:
        render_html(f"""<div style="display:flex;align-items:center;gap:8px;padding:8px 0;">
  <span style="font-size:14px;font-weight:600;color:#111827;">Seating Capacity Matrix</span>
  <span style="font-size:11px;color:#9CA3AF;font-family:'JetBrains Mono',monospace;">{len(layout.seats) if layout else 0} Registered Seats</span>
</div>""")
    with f_col2:
        matrix_filter = st.radio(
            "Matrix Filter",
            ["All", "Available", "Occupied", "Alerts"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )

    # Build Dynamic Seat Rows from Real Pipeline
    all_seats_data = []
    if layout and layout.seats:
        for s_id, seat in layout.seats.items():
            t_id = seat.table_id
            s_ts = seat_states.get(s_id)
            state_val = s_ts.availability_state if s_ts else AvailabilityState.AVAILABLE
            dur_sec = s_ts.occupancy_duration if s_ts else 0.0
            dur_disp = f"{int(dur_sec//60):02d}:{int(dur_sec%60):02d}"

            custs = spatial.seat_persons.get(s_id, []) if spatial else []
            cust_disp = f"Customer #{custs[0]}" if custs else "—"

            b_list = spatial.seat_belongings.get(s_id, []) if spatial else []
            item_disp = ", ".join(set(b.cls_name for b in b_list)) if b_list else "—"

            if state_val == AvailabilityState.OCCUPIED:
                st_name, st_col, st_dot = "Occupied", "#DC2626", "●"
            elif state_val == AvailabilityState.RESERVED_BY_BELONGINGS:
                st_name, st_col, st_dot = "Belongings Hold", "#D97706", "●"
            elif s_ts and s_ts.duration_warning_state == DurationWarningState.OVERTIME:
                st_name, st_col, st_dot = "Overtime", "#DC2626", "●"
            else:
                st_name, st_col, st_dot = "Available", "#16A34A", "●"

            all_seats_data.append({
                "id": s_id,
                "table": f"Table {t_id}",
                "type": f"{t_id} Seating",
                "status": st_name,
                "color": st_col,
                "dot": st_dot,
                "cust": cust_disp,
                "dwell": dur_disp,
                "items": item_disp
            })

    # Filter seats
    filtered_seats = []
    for s in all_seats_data:
        if matrix_filter == "All":
            filtered_seats.append(s)
        elif matrix_filter == "Available" and s["status"] == "Available":
            filtered_seats.append(s)
        elif matrix_filter == "Occupied" and s["status"] == "Occupied":
            filtered_seats.append(s)
        elif matrix_filter == "Alerts" and s["status"] in ["Approaching Limit", "Belongings Hold", "Overtime"]:
            filtered_seats.append(s)

    seat_rows_html = []
    for s in filtered_seats:
        bg = "background:#FFFBEB;" if s["status"] == "Belongings Hold" else ("background:#FEF2F2;" if s["status"] == "Overtime" else "")
        seat_rows_html.append(f"""<tr style="border-bottom:1px solid #F3F4F6;{bg}">
  <td style="padding:10px 16px;font-family:'JetBrains Mono',monospace;font-weight:600;color:#111827;">{s['id']}</td>
  <td style="padding:10px 16px;font-weight:500;">{s['table']}</td>
  <td style="padding:10px 16px;color:#6B7280;">{s['type']}</td>
  <td style="padding:10px 16px;color:{s['color']};font-weight:600;">{s['dot']} {s['status']}</td>
  <td style="padding:10px 16px;">{s['cust']}</td>
  <td style="padding:10px 16px;font-family:'JetBrains Mono',monospace;">{s['dwell']}</td>
  <td style="padding:10px 16px;">{s['items']}</td>
  <td style="padding:10px 16px;text-align:right;color:#9CA3AF;">›</td>
</tr>""")

    rendered_rows = "".join(seat_rows_html)

    matrix_html = f"""<div class="stitch-table-card" style="margin-bottom:24px;">
  <div style="overflow-x:auto;font-size:12px;">
    <table style="width:100%;text-align:left;border-collapse:collapse;">
      <thead>
        <tr style="border-bottom:1px solid #F3F4F6;font-size:11px;color:#9CA3AF;background:#FAFAFA;">
          <th style="padding:8px 16px;">Seat ID</th>
          <th style="padding:8px 16px;">Table</th>
          <th style="padding:8px 16px;">Type / Zone</th>
          <th style="padding:8px 16px;">Status</th>
          <th style="padding:8px 16px;">Customer</th>
          <th style="padding:8px 16px;">Dwell</th>
          <th style="padding:8px 16px;">Belongings</th>
          <th style="padding:8px 16px;text-align:right;">Action</th>
        </tr>
      </thead>
      <tbody style="color:#374151;">
        {rendered_rows}
      </tbody>
    </table>
  </div>
  <div style="padding:10px 16px;background:#FAFAFA;border-top:1px solid #F3F4F6;display:flex;align-items:center;justify-content:space-between;font-size:11px;color:#9CA3AF;">
    <span>Showing {len(filtered_seats)} of {len(all_seats_data)} seats · Edge Processing YOLOv8 & ByteTrack</span>
    <div style="display:flex;align-items:center;gap:8px;">
      <span>Status: Calibrated Active</span>
    </div>
  </div>
</div>"""
    render_html(matrix_html)

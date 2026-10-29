"""Spatial and Geometry Analysis Module (Phase 3).

Handles Seat and Table ROI management, geometric intersection (IoU, IoA,
centroid containment), and customer/belongings spatial association.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from shapely.geometry import Point, Polygon, box as shapely_box

from src.detector import Detection
from src.tracker import TrackedPerson


@dataclass
class SeatROI:
    id: str
    table_id: str
    polygon: List[List[float]]
    _shapely_poly: Polygon = field(init=False, repr=False)

    def __post_init__(self):
        self._shapely_poly = Polygon(self.polygon)

    @property
    def shapely(self) -> Polygon:
        return self._shapely_poly

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self._shapely_poly.bounds


@dataclass
class TableROI:
    id: str
    name: str
    polygon: List[List[float]]
    seats: List[SeatROI] = field(default_factory=list)
    _shapely_poly: Polygon = field(init=False, repr=False)

    def __post_init__(self):
        self._shapely_poly = Polygon(self.polygon)

    @property
    def shapely(self) -> Polygon:
        return self._shapely_poly

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self._shapely_poly.bounds


@dataclass
class SpatialAssociationResult:
    # Table level associations
    table_persons: Dict[str, List[int]] = field(default_factory=dict)         # table_id -> [track_id]
    table_belongings: Dict[str, List[Detection]] = field(default_factory=dict) # table_id -> [Detection]
    table_ambiguous: Dict[str, bool] = field(default_factory=dict)             # table_id -> bool
    
    # Seat level associations
    seat_persons: Dict[str, List[int]] = field(default_factory=dict)          # seat_id -> [track_id]
    seat_belongings: Dict[str, List[Detection]] = field(default_factory=dict)  # seat_id -> [Detection]
    seat_ambiguous: Dict[str, bool] = field(default_factory=dict)              # seat_id -> bool

    # Unassociated detections
    unassociated_persons: List[int] = field(default_factory=list)
    unassociated_belongings: List[Detection] = field(default_factory=list)


def box_to_polygon(xyxy: Tuple[float, float, float, float]) -> Polygon:
    x1, y1, x2, y2 = xyxy
    return shapely_box(x1, y1, x2, y2)


def box_centroid(xyxy: Tuple[float, float, float, float]) -> Point:
    x1, y1, x2, y2 = xyxy
    return Point((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_footpoint(xyxy: Tuple[float, float, float, float]) -> Point:
    """Bottom center of bounding box, often closest to ground/seat level."""
    x1, y1, x2, y2 = xyxy
    return Point((x1 + x2) / 2.0, y2)


def intersection_over_box_area(box_poly: Polygon, region_poly: Polygon) -> float:
    if not box_poly.intersects(region_poly):
        return 0.0
    inter_area = box_poly.intersection(region_poly).area
    return inter_area / max(box_poly.area, 1e-6)


def intersection_over_region_area(box_poly: Polygon, region_poly: Polygon) -> float:
    if not region_poly.intersects(box_poly):
        return 0.0
    inter_area = region_poly.intersection(box_poly).area
    return inter_area / max(region_poly.area, 1e-6)


class SeatingLayout:
    def __init__(self, config_path: Optional[str] = None):
        self.tables: Dict[str, TableROI] = {}
        self.seats: Dict[str, SeatROI] = {}
        self.video_resolution: Tuple[int, int] = (1280, 720)
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str):
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Seating ROI config not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.video_resolution = tuple(data.get("video_resolution", [1280, 720]))
        self.tables.clear()
        self.seats.clear()

        for t_data in data.get("tables", []):
            t_id = t_data["id"]
            seats_list: List[SeatROI] = []
            for s_data in t_data.get("seats", []):
                s_id = s_data["id"]
                seat = SeatROI(id=s_id, table_id=t_id, polygon=s_data["polygon"])
                seats_list.append(seat)
                self.seats[s_id] = seat

            table = TableROI(
                id=t_id,
                name=t_data.get("name", f"Table {t_id}"),
                polygon=t_data["polygon"],
                seats=seats_list
            )
            self.tables[t_id] = table

    def save_config(self, config_path: str):
        data = {
            "video_resolution": list(self.video_resolution),
            "tables": []
        }
        for table in self.tables.values():
            t_dict = {
                "id": table.id,
                "name": table.name,
                "polygon": table.polygon,
                "seats": [
                    {"id": s.id, "polygon": s.polygon} for s in table.seats
                ]
            }
            data["tables"].append(t_dict)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def associate_spatial(
        self,
        tracked_persons: List[TrackedPerson],
        detections: List[Detection],
        min_belonging_ioa: float = 0.10,
        min_person_ioa: float = 0.15,
        ambiguity_margin: float = 0.05
    ) -> SpatialAssociationResult:
        """Associate tracked persons and detected belongings with tables and seats."""
        result = SpatialAssociationResult()

        for t_id in self.tables:
            result.table_persons[t_id] = []
            result.table_belongings[t_id] = []
            result.table_ambiguous[t_id] = False

        for s_id in self.seats:
            result.seat_persons[s_id] = []
            result.seat_belongings[s_id] = []
            result.seat_ambiguous[s_id] = False

        # 1. Associate Tracked Persons
        for p in tracked_persons:
            p_poly = box_to_polygon(p.xyxy)
            p_centroid = box_centroid(p.xyxy)
            p_foot = box_footpoint(p.xyxy)
            associated_any = False

            # First check individual seats
            best_seat: Optional[Tuple[str, float]] = None
            second_seat_score = 0.0
            for s_id, seat in self.seats.items():
                ioa = intersection_over_box_area(p_poly, seat.shapely)
                point_in = 1.0 if (seat.shapely.contains(p_centroid) or seat.shapely.contains(p_foot)) else 0.0
                score = max(ioa, point_in * 0.5)
                if score >= min_person_ioa:
                    if best_seat is None or score > best_seat[1]:
                        if best_seat is not None:
                            second_seat_score = best_seat[1]
                        best_seat = (s_id, score)
                    elif score > second_seat_score:
                        second_seat_score = score

            if best_seat and best_seat[1] >= min_person_ioa:
                s_id = best_seat[0]
                result.seat_persons[s_id].append(p.track_id)
                t_id = self.seats[s_id].table_id
                if p.track_id not in result.table_persons[t_id]:
                    result.table_persons[t_id].append(p.track_id)
                associated_any = True
                if (best_seat[1] - second_seat_score) < ambiguity_margin and second_seat_score > 0:
                    result.seat_ambiguous[s_id] = True

            # If not associated with a seat, check table region directly
            if not associated_any:
                best_table: Optional[Tuple[str, float]] = None
                for t_id, table in self.tables.items():
                    ioa = intersection_over_box_area(p_poly, table.shapely)
                    point_in = 1.0 if (table.shapely.contains(p_centroid) or table.shapely.contains(p_foot)) else 0.0
                    score = max(ioa, point_in * 0.4)
                    if score >= min_person_ioa:
                        if best_table is None or score > best_table[1]:
                            best_table = (t_id, score)

                if best_table and best_table[1] >= min_person_ioa:
                    t_id = best_table[0]
                    result.table_persons[t_id].append(p.track_id)
                    associated_any = True

            if not associated_any:
                result.unassociated_persons.append(p.track_id)

        # 2. Associate Belongings
        belongings_dets = [d for d in detections if d.category == "belongings"]
        for b in belongings_dets:
            b_poly = box_to_polygon(b.xyxy)
            b_centroid = box_centroid(b.xyxy)
            associated_any = False

            # Belongings belong primarily to tables
            best_table: Optional[Tuple[str, float]] = None
            second_table_score = 0.0
            for t_id, table in self.tables.items():
                ioa = intersection_over_box_area(b_poly, table.shapely)
                centroid_in = 1.0 if table.shapely.contains(b_centroid) else 0.0
                score = max(ioa, centroid_in * 0.8)
                if score >= min_belonging_ioa:
                    if best_table is None or score > best_table[1]:
                        if best_table is not None:
                            second_table_score = best_table[1]
                        best_table = (t_id, score)
                    elif score > second_table_score:
                        second_table_score = score

            if best_table and best_table[1] >= min_belonging_ioa:
                t_id = best_table[0]
                result.table_belongings[t_id].append(b)
                associated_any = True
                if (best_table[1] - second_table_score) < ambiguity_margin and second_table_score > 0:
                    result.table_ambiguous[t_id] = True

                # Also associate with nearest seat at that table if overlapping
                for s in self.tables[t_id].seats:
                    s_ioa = intersection_over_box_area(b_poly, s.shapely)
                    if s_ioa >= min_belonging_ioa or s.shapely.contains(b_centroid):
                        result.seat_belongings[s.id].append(b)
                        break

            if not associated_any:
                result.unassociated_belongings.append(b)

        return result


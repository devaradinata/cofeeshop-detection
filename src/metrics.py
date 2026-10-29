"""Occupancy and Seating Pressure Metrics Module (Phase 4).

Computes seating capacity metrics, occupancy rates, duration warnings,
and rule-based seating pressure assessments.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.state_machine import AvailabilityState, DurationWarningState, RegionState


@dataclass
class OccupancyMetrics:
    # Seat metrics
    total_seats: int
    occupied_seats: int
    available_seats: int
    belongings_seats: int
    unknown_seats: int
    seat_occupancy_rate: float       # Occupied Seats / Total Configured Seats * 100%
    effective_seat_occupancy_rate: float # (Occupied + Belongings) / Total Seats * 100%

    # Table metrics
    total_tables: int
    occupied_tables: int
    available_tables: int
    belongings_tables: int
    unknown_tables: int
    table_occupancy_rate: float      # Occupied Tables / Total Configured Tables * 100%
    approaching_limit_tables: int
    overtime_tables: int

    # Seating Pressure Assessment
    seating_pressure: str            # 'LOW', 'MEDIUM', 'HIGH'
    pressure_score: float            # 0.0 to 1.0 scale
    pressure_reason: str             # Transparent explainable rule summary


class MetricsCalculator:
    def __init__(
        self,
        high_occupancy_threshold: float = 80.0,
        medium_occupancy_threshold: float = 50.0,
        min_available_tables_critical: int = 1
    ):
        self.high_occupancy_threshold = high_occupancy_threshold
        self.medium_occupancy_threshold = medium_occupancy_threshold
        self.min_available_tables_critical = min_available_tables_critical

    def compute(
        self,
        table_states: Dict[str, RegionState],
        seat_states: Dict[str, RegionState]
    ) -> OccupancyMetrics:
        # Seat metrics
        tot_seats = len(seat_states)
        occ_seats = sum(1 for s in seat_states.values() if s.availability_state == AvailabilityState.OCCUPIED)
        avail_seats = sum(1 for s in seat_states.values() if s.availability_state == AvailabilityState.AVAILABLE)
        bel_seats = sum(1 for s in seat_states.values() if s.availability_state == AvailabilityState.RESERVED_BY_BELONGINGS)
        unk_seats = sum(1 for s in seat_states.values() if s.availability_state == AvailabilityState.UNKNOWN)

        # Standard PRD formula: Occupied Seats / Total Configured Seats * 100%
        seat_occ_rate = (occ_seats / max(tot_seats, 1)) * 100.0
        eff_occ_rate = ((occ_seats + bel_seats) / max(tot_seats, 1)) * 100.0

        # Table metrics
        tot_tables = len(table_states)
        occ_tables = sum(1 for t in table_states.values() if t.availability_state == AvailabilityState.OCCUPIED)
        avail_tables = sum(1 for t in table_states.values() if t.availability_state == AvailabilityState.AVAILABLE)
        bel_tables = sum(1 for t in table_states.values() if t.availability_state == AvailabilityState.RESERVED_BY_BELONGINGS)
        unk_tables = sum(1 for t in table_states.values() if t.availability_state == AvailabilityState.UNKNOWN)
        
        approaching_tables = sum(
            1 for t in table_states.values()
            if t.duration_warning_state == DurationWarningState.APPROACHING_LIMIT
        )
        overtime_tables = sum(
            1 for t in table_states.values()
            if t.duration_warning_state == DurationWarningState.OVERTIME
        )
        table_occ_rate = (occ_tables / max(tot_tables, 1)) * 100.0

        # Rule-Based Seating Pressure Assessment
        pressure_level, score, reason = self._assess_seating_pressure(
            tot_seats, occ_seats, avail_seats, bel_seats, unk_seats, seat_occ_rate, eff_occ_rate,
            tot_tables, occ_tables, avail_tables, bel_tables, unk_tables, overtime_tables
        )

        return OccupancyMetrics(
            total_seats=tot_seats,
            occupied_seats=occ_seats,
            available_seats=avail_seats,
            belongings_seats=bel_seats,
            unknown_seats=unk_seats,
            seat_occupancy_rate=round(seat_occ_rate, 1),
            effective_seat_occupancy_rate=round(eff_occ_rate, 1),
            total_tables=tot_tables,
            occupied_tables=occ_tables,
            available_tables=avail_tables,
            belongings_tables=bel_tables,
            unknown_tables=unk_tables,
            table_occupancy_rate=round(table_occ_rate, 1),
            approaching_limit_tables=approaching_tables,
            overtime_tables=overtime_tables,
            seating_pressure=pressure_level,
            pressure_score=round(score, 2),
            pressure_reason=reason
        )

    def _assess_seating_pressure(
        self,
        tot_seats: int, occ_seats: int, avail_seats: int, bel_seats: int, unk_seats: int,
        seat_occ_rate: float, eff_occ_rate: float,
        tot_tables: int, occ_tables: int, avail_tables: int, bel_tables: int, unk_tables: int,
        overtime_tables: int
    ) -> Tuple[str, float, str]:
        """Explainable, transparent rule-based scoring."""
        reasons = []

        # High pressure rules
        if avail_tables == 0 and tot_tables > 0:
            reasons.append("No completely available tables remain for new parties.")
            return "HIGH", 0.95, " | ".join(reasons)

        if seat_occ_rate >= self.high_occupancy_threshold:
            reasons.append(f"Occupancy rate ({seat_occ_rate:.1f}%) meets/exceeds high threshold ({self.high_occupancy_threshold}%).")
            if avail_seats <= 2:
                reasons.append(f"Critical remaining seat capacity ({avail_seats} seats).")
            return "HIGH", 0.88, " | ".join(reasons)

        if eff_occ_rate >= self.high_occupancy_threshold:
            reasons.append(f"Effective occupancy ({eff_occ_rate:.1f}% including belongings) exceeds {self.high_occupancy_threshold}%.")
            return "HIGH", 0.82, " | ".join(reasons)

        # Medium pressure rules
        if seat_occ_rate >= self.medium_occupancy_threshold:
            reasons.append(f"Occupancy rate ({seat_occ_rate:.1f}%) in moderate range ({self.medium_occupancy_threshold}%-{self.high_occupancy_threshold}%).")
            if avail_tables <= 2:
                reasons.append(f"Limited tables available ({avail_tables}/{tot_tables}).")
            if overtime_tables > 0:
                reasons.append(f"{overtime_tables} table(s) currently overtime.")
            return "MEDIUM", 0.55, " | ".join(reasons)

        if avail_tables <= 2 and tot_tables >= 4:
            reasons.append(f"Available tables count low ({avail_tables}/{tot_tables}).")
            return "MEDIUM", 0.50, " | ".join(reasons)

        if bel_tables >= 2:
            reasons.append(f"Multiple tables reserved by belongings ({bel_tables} tables).")
            return "MEDIUM", 0.45, " | ".join(reasons)

        # Low pressure rule
        reasons.append(f"Capacity available ({avail_seats}/{tot_seats} seats, {avail_tables}/{tot_tables} tables available).")
        score = max(0.1, round(seat_occ_rate / 100.0, 2))
        return "LOW", score, " | ".join(reasons)


def compute_mae(ground_truth_durations: List[float], predicted_durations: List[float]) -> float:
    """Mean Absolute Error (MAE) for duration evaluation."""
    if not ground_truth_durations or not predicted_durations:
        return 0.0
    if len(ground_truth_durations) != len(predicted_durations):
        raise ValueError("Ground truth and prediction lists must have same length.")
    diffs = [abs(g - p) for g, p in zip(ground_truth_durations, predicted_durations)]
    return sum(diffs) / len(diffs)


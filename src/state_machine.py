"""Temporal State Machine and Duration Monitoring Module (Phase 3).

Maintains availability states, session lifecycles, grace period buffering,
duration calculations, and duration limit warnings for tables and seats.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from src.detector import Detection
from src.spatial import SpatialAssociationResult


class AvailabilityState(str, Enum):
    AVAILABLE = "AVAILABLE"
    OCCUPIED = "OCCUPIED"
    RESERVED_BY_BELONGINGS = "RESERVED_BY_BELONGINGS"
    UNKNOWN = "UNKNOWN"


class DurationWarningState(str, Enum):
    NORMAL = "NORMAL"
    APPROACHING_LIMIT = "APPROACHING_LIMIT"
    OVERTIME = "OVERTIME"


@dataclass
class OccupancySession:
    session_id: str
    region_id: str
    start_time: float
    last_active_time: float
    end_time: Optional[float] = None
    associated_person_ids: Set[int] = field(default_factory=set)
    total_occupied_seconds: float = 0.0
    is_active: bool = True

    def duration(self, current_time: float) -> float:
        if self.is_active:
            return max(0.0, current_time - self.start_time)
        return max(0.0, (self.end_time or self.last_active_time) - self.start_time)


@dataclass
class RegionState:
    region_id: str
    is_table: bool
    availability_state: AvailabilityState = AvailabilityState.AVAILABLE
    duration_warning_state: DurationWarningState = DurationWarningState.NORMAL
    occupancy_duration: float = 0.0
    belongings_duration: float = 0.0
    active_person_ids: List[int] = field(default_factory=list)
    active_belongings_count: int = 0
    current_session: Optional[OccupancySession] = None
    completed_sessions: List[OccupancySession] = field(default_factory=list)
    
    # Internal temporal tracking
    last_seen_person_time: Optional[float] = None
    last_seen_belongings_time: Optional[float] = None
    belongings_start_time: Optional[float] = None
    empty_since_time: Optional[float] = None
    unknown_since_time: Optional[float] = None
    session_counter: int = 0


class SeatingStateMachine:
    def __init__(
        self,
        table_ids: List[str],
        seat_ids: List[str],
        duration_limit_seconds: float = 60.0,   # Configurable threshold (e.g. 60s for demo, 7200s for 2h)
        approaching_ratio: float = 0.90,        # 90% of duration limit
        grace_period_seconds: float = 3.0,      # Grace period for temporary customer absence (e.g. 3s for 10s video, 120s for real)
        confirmation_seconds: float = 0.5       # Temporal smoothing buffer to confirm state changes
    ):
        self.duration_limit_seconds = duration_limit_seconds
        self.approaching_ratio = approaching_ratio
        self.grace_period_seconds = grace_period_seconds
        self.confirmation_seconds = confirmation_seconds

        self.table_states: Dict[str, RegionState] = {
            t_id: RegionState(region_id=t_id, is_table=True) for t_id in table_ids
        }
        self.seat_states: Dict[str, RegionState] = {
            s_id: RegionState(region_id=s_id, is_table=False) for s_id in seat_ids
        }

        # Track person visibility duration across all frames (independent of seating)
        self.person_first_seen: Dict[int, float] = {}
        self.person_last_seen: Dict[int, float] = {}

    def get_person_visibility_durations(self, current_time: float) -> Dict[int, float]:
        """Distinguish person visibility duration from table/seat occupancy duration."""
        return {
            pid: max(0.0, self.person_last_seen.get(pid, current_time) - self.person_first_seen[pid])
            for pid in self.person_first_seen
        }

    def _update_region(
        self,
        state: RegionState,
        person_ids: List[int],
        belongings: List[Detection],
        is_ambiguous: bool,
        current_time: float
    ):
        has_person = len(person_ids) > 0
        has_belongings = len(belongings) > 0
        state.active_person_ids = list(person_ids)
        state.active_belongings_count = len(belongings)

        # 1. Update temporal markers
        if has_person:
            state.last_seen_person_time = current_time
            state.empty_since_time = None
        if has_belongings:
            state.last_seen_belongings_time = current_time
            if state.belongings_start_time is None:
                state.belongings_start_time = current_time
        else:
            state.belongings_start_time = None

        if not has_person and not has_belongings:
            if state.empty_since_time is None:
                state.empty_since_time = current_time

        # 2. Ambiguity check: if detections are contradictory or marked ambiguous
        if is_ambiguous:
            if state.unknown_since_time is None:
                state.unknown_since_time = current_time
            # If ambiguous for longer than confirmation window, mark UNKNOWN
            if (current_time - state.unknown_since_time) >= self.confirmation_seconds:
                state.availability_state = AvailabilityState.UNKNOWN
                return
        else:
            state.unknown_since_time = None

        # 3. State transitions
        if has_person:
            # Condition 1: Customer associated with table -> OCCUPIED
            state.availability_state = AvailabilityState.OCCUPIED

            # Start or maintain occupancy session
            if state.current_session is None or not state.current_session.is_active:
                state.session_counter += 1
                state.current_session = OccupancySession(
                    session_id=f"{state.region_id}_sess_{state.session_counter}",
                    region_id=state.region_id,
                    start_time=current_time,
                    last_active_time=current_time,
                    associated_person_ids=set(person_ids),
                    is_active=True
                )
            else:
                state.current_session.last_active_time = current_time
                state.current_session.associated_person_ids.update(person_ids)

            state.occupancy_duration = state.current_session.duration(current_time)
            state.belongings_duration = 0.0

        else:
            # No customer currently visible
            time_since_person = (
                current_time - state.last_seen_person_time
                if state.last_seen_person_time is not None
                else float("inf")
            )

            if time_since_person <= self.grace_period_seconds and state.current_session is not None and state.current_session.is_active:
                # Customer absent temporarily within grace period (e.g. restroom/order pickup)
                if has_belongings:
                    # Belongings left behind during grace period
                    state.availability_state = AvailabilityState.RESERVED_BY_BELONGINGS
                else:
                    # Still in grace period, maintain OCCUPIED to prevent single-frame flickering
                    state.availability_state = AvailabilityState.OCCUPIED

                state.occupancy_duration = state.current_session.duration(current_time)
                if state.belongings_start_time is not None:
                    state.belongings_duration = current_time - state.belongings_start_time
                else:
                    state.belongings_duration = 0.0
            else:
                # Grace period expired or never occupied
                if state.current_session is not None and state.current_session.is_active:
                    # Terminate the occupancy session
                    state.current_session.is_active = False
                    state.current_session.end_time = state.last_seen_person_time or current_time
                    state.completed_sessions.append(state.current_session)
                    state.current_session = None

                state.occupancy_duration = 0.0

                if has_belongings:
                    # Condition 2: No customer, but belongings consistently present -> RESERVED_BY_BELONGINGS
                    if state.belongings_start_time is not None:
                        state.belongings_duration = current_time - state.belongings_start_time
                    state.availability_state = AvailabilityState.RESERVED_BY_BELONGINGS
                else:
                    # Condition 3: Empty confirmation
                    state.belongings_duration = 0.0
                    empty_duration = (
                        current_time - state.empty_since_time
                        if state.empty_since_time is not None
                        else float("inf")
                    )
                    if empty_duration >= self.confirmation_seconds:
                        state.availability_state = AvailabilityState.AVAILABLE

        # 4. Update duration warning state (NORMAL, APPROACHING_LIMIT, OVERTIME)
        # Note: Non-mutually exclusive with availability state.
        if state.availability_state == AvailabilityState.OCCUPIED:
            if state.occupancy_duration >= self.duration_limit_seconds:
                state.duration_warning_state = DurationWarningState.OVERTIME
            elif state.occupancy_duration >= (self.duration_limit_seconds * self.approaching_ratio):
                state.duration_warning_state = DurationWarningState.APPROACHING_LIMIT
            else:
                state.duration_warning_state = DurationWarningState.NORMAL
        else:
            state.duration_warning_state = DurationWarningState.NORMAL

    def update(
        self,
        association: SpatialAssociationResult,
        current_time: float,
        visible_person_ids: List[int]
    ):
        """Update state machine for all tables and seats for current timestamp."""
        # Update person overall visibility tracking
        for pid in visible_person_ids:
            if pid not in self.person_first_seen:
                self.person_first_seen[pid] = current_time
            self.person_last_seen[pid] = current_time

        # Update Tables
        for t_id, t_state in self.table_states.items():
            p_ids = association.table_persons.get(t_id, [])
            belongings = association.table_belongings.get(t_id, [])
            is_ambig = association.table_ambiguous.get(t_id, False)
            self._update_region(t_state, p_ids, belongings, is_ambig, current_time)

        # Update Seats
        for s_id, s_state in self.seat_states.items():
            p_ids = association.seat_persons.get(s_id, [])
            belongings = association.seat_belongings.get(s_id, [])
            is_ambig = association.seat_ambiguous.get(s_id, False)
            self._update_region(s_state, p_ids, belongings, is_ambig, current_time)


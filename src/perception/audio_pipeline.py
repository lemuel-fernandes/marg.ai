"""Acoustic attention module (ROADMAP #27b, skeleton).

Simulates siren/horn detection and direction-of-arrival (DOA) so the
occlusion-aware planner coupling can be validated end-to-end without real
audio hardware. Scenarios declare synthetic acoustic events; this node turns
them into `AttentionCue`s on the message bus, with the same semantics a real
mic-array frontend would produce:

- detection gate (SNR threshold) models faint/inaudible events,
- persistence models reverberation/decay after the event ends,
- mounting miscalibration shifts reported azimuth,
- confidence is derived from SNR.

The physical policy (map cues -> speed caps / creep release) lives in
`AudioPlannerPolicy` so planner coupling stays decoupled and unit-testable.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

from src.common.types.sensor import SensorFrame


# Acoustic event classes (kept as plain strings: no ML class enum yet).
CLASS_SIREN = "siren"
CLASS_HORN = "horn"
CLASS_RICKSHAW_HOOTER = "rickshaw_hooter"

# Events below this SNR (dB) are treated as undetectable (detection gate).
DETECT_SNR_DB = 6.0
# Cues persist this long after an event ends (reverb/decay model).
PERSISTENCE_S = 2.0
# Confidence mapping: SNR in [DETECT_SNR_DB, 30] -> [0.55, 0.99].
MIN_CONFIDENCE = 0.55
MAX_CONFIDENCE = 0.99


@dataclass(frozen=True)
class AcousticEvent:
    """Scenario-level synthetic acoustic ground truth.

    t_onset/t_end: active window in sim seconds.
    azimuth_rad: direction of arrival in base_link at onset (0 = forward,
        positive = left), held constant for the event.
    snr_db: signal-to-noise ratio; events below DETECT_SNR_DB are not heard.
    """

    t_onset: float
    t_end: float
    acoustic_class: str
    azimuth_rad: float
    snr_db: float = 20.0

    def __post_init__(self):
        if self.t_end < self.t_onset:
            raise ValueError("AcousticEvent requires t_end >= t_onset")


@dataclass(frozen=True)
class AttentionCue:
    """Published acoustic attention cue (base_link frame)."""

    t: float
    acoustic_class: str
    azimuth_rad: float
    confidence: float
    t_onset: float
    is_persisted: bool = False


def _confidence_from_snr(snr_db: float) -> float:
    """Deterministic SNR -> confidence mapping."""
    lo, hi = DETECT_SNR_DB, 30.0
    frac = max(0.0, min(1.0, (snr_db - lo) / (hi - lo)))
    return MIN_CONFIDENCE + frac * (MAX_CONFIDENCE - MIN_CONFIDENCE)


class AcousticPerceptionNode:
    """Synthetic acoustic frontend: events -> attention cues.

    Implements the `PerceptionNode` call shape (`process(frame)`) but returns
    a list of `AttentionCue`s rather than `PerceptionOutput` — acoustic cues
    are attention metadata, not obstacle hypotheses, so they never enter the
    costmap.
    """

    def __init__(self, events: Optional[List[AcousticEvent]] = None,
                 persistence_s: float = PERSISTENCE_S,
                 detect_snr_db: float = DETECT_SNR_DB):
        self.events = list(events or [])
        self.persistence_s = persistence_s
        self.detect_snr_db = detect_snr_db
        # Mounting miscalibration (rad): added to every reported azimuth.
        self._yaw_offset_rad = 0.0
        # Cache of last-seen cues per (class, azimuth-key) for persistence.
        self._persisting: List[dict] = []

    def update_imu_yaw_offset(self, yaw_offset_rad: float) -> None:
        """Mirror of the vision pipeline's miscalibration compensation hook."""
        self._yaw_offset_rad = float(yaw_offset_rad)

    def process(self, frame: SensorFrame) -> List[AttentionCue]:
        t = frame.header.stamp

        cues: List[AttentionCue] = []
        for ev in self.events:
            # Detection gate: too faint -> never detected.
            if ev.snr_db < self.detect_snr_db:
                continue
            if ev.t_onset <= t <= ev.t_end:
                cues.append(AttentionCue(
                    t=t,
                    acoustic_class=ev.acoustic_class,
                    # Positive yaw offset = sensor mounted rotated left.
                    azimuth_rad=ev.azimuth_rad + self._yaw_offset_rad,
                    confidence=_confidence_from_snr(ev.snr_db),
                    t_onset=ev.t_onset,
                    is_persisted=False,
                ))
            elif ev.t_end < t <= ev.t_end + self.persistence_s:
                # Persistence: cue survives after the sound decays.
                cues.append(AttentionCue(
                    t=t,
                    acoustic_class=ev.acoustic_class,
                    azimuth_rad=ev.azimuth_rad + self._yaw_offset_rad,
                    confidence=_confidence_from_snr(ev.snr_db) * 0.8,
                    t_onset=ev.t_onset,
                    is_persisted=True,
                ))
        return cues


class AudioPlannerPolicy:
    """Maps attention cues to planner-relevant signals.

    Encodes the #27b coupling contract:
    - a siren cue imposes a defensive speed cap,
    - a horn/siren cue with NO matching visual track in its azimuth cone
      releases the occlusion creep-cap (something audible is approaching but
      not yet visible — creeping blind is the wrong response).
    """

    # Defensive cap while a siren is audible (m/s).
    SIREN_SPEED_CAP = 2.0
    # Half-angle of the azimuth cone for matching visual tracks (rad ~15 deg).
    CONE_HALF_ANGLE = 0.26

    def __init__(self, cone_half_angle: float = CONE_HALF_ANGLE,
                 siren_speed_cap: float = SIREN_SPEED_CAP):
        self.cone_half_angle = cone_half_angle
        self.siren_speed_cap = siren_speed_cap

    def siren_cap(self, cues: List[AttentionCue]) -> Optional[float]:
        """Speed cap to apply, or None if no siren cue is active."""
        if any(c.acoustic_class == CLASS_SIREN for c in cues):
            return self.siren_speed_cap
        return None

    @staticmethod
    def track_azimuth(track_xy: tuple, ego_pose) -> float:
        """Azimuth of a visual track (map xy) relative to ego heading."""
        dx = track_xy[0] - ego_pose.x
        dy = track_xy[1] - ego_pose.y
        return math.atan2(dy, dx) - ego_pose.heading

    def matching_visual_track(self, cue: AttentionCue,
                              visual_tracks_azimuth: List[float]) -> bool:
        """True when some visual track lies inside the cue's azimuth cone."""
        for az in visual_tracks_azimuth:
            delta = math.atan2(math.sin(az - cue.azimuth_rad),
                               math.cos(az - cue.azimuth_rad))
            if abs(delta) <= self.cone_half_angle:
                return True
        return False

    def creep_release(self, cues: List[AttentionCue],
                      visual_tracks_azimuth: List[float]) -> bool:
        """Release the occlusion creep-cap?

        True when an audible cue has no matching visual track: the vehicle
        should NOT creep blind while something audible approaches unseen.
        """
        audible = [c for c in cues
                   if c.acoustic_class in (CLASS_SIREN, CLASS_HORN,
                                           CLASS_RICKSHAW_HOOTER)]
        if not audible:
            return False
        return not any(self.matching_visual_track(c, visual_tracks_azimuth)
                       for c in audible)

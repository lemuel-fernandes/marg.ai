"""Unit tests for the acoustic attention skeleton (ROADMAP #27b)."""
import math
from types import SimpleNamespace

from src.common.types.base import FrameId, Header
from src.common.types.sensor import SensorFrame
from src.perception.audio_pipeline import (
    AcousticEvent, AcousticPerceptionNode, AudioPlannerPolicy,
    AttentionCue, CLASS_HORN, CLASS_SIREN,
)


def _frame(t: float) -> SensorFrame:
    return SensorFrame(header=Header(t, FrameId.SENSOR_FRONT, "test"))


def _event(t_onset=1.0, t_end=4.0, cls=CLASS_SIREN, az=1.2, snr=20.0):
    return AcousticEvent(t_onset=t_onset, t_end=t_end, acoustic_class=cls,
                         azimuth_rad=az, snr_db=snr)


def test_event_within_window_produces_cue():
    node = AcousticPerceptionNode([_event()])
    cues = node.process(_frame(2.0))
    assert len(cues) == 1
    cue = cues[0]
    assert cue.acoustic_class == CLASS_SIREN
    assert math.isclose(cue.azimuth_rad, 1.2, abs_tol=1e-9)
    assert not cue.is_persisted
    assert 0.5 < cue.confidence < 1.0


def test_detection_gate_blocks_faint_events():
    node = AcousticPerceptionNode([_event(snr=3.0)])  # below 6 dB gate
    assert node.process(_frame(2.0)) == []


def test_persistence_after_event_end():
    node = AcousticPerceptionNode([_event(t_end=4.0)], persistence_s=2.0)
    assert node.process(_frame(4.3))[0].is_persisted
    assert node.process(_frame(5.9))[0].is_persisted
    assert node.process(_frame(6.1)) == []  # expired


def test_yaw_miscalibration_shifts_azimuth():
    node = AcousticPerceptionNode([_event(az=0.5)])
    node.update_imu_yaw_offset(0.1)
    assert math.isclose(node.process(_frame(2.0))[0].azimuth_rad, 0.6,
                        abs_tol=1e-9)


def test_pipeline_publishes_cues_on_bus():
    from src.perception.audio_pipeline import AcousticEvent, AcousticPerceptionNode
    from src.integration.message_bus import Topic, TypedMessageBus

    received = []
    bus = TypedMessageBus()
    bus.subscribe(Topic.ACOUSTIC_CUE, received.append)

    node = AcousticPerceptionNode([AcousticEvent(0.0, 10.0, CLASS_SIREN, 0.0)])
    cue = node.process(_frame(1.0))[0]
    bus.publish(Topic.ACOUSTIC_CUE, cue)
    assert received == [cue]


def test_policy_siren_cap():
    policy = AudioPlannerPolicy()
    cue = SimpleNamespace(acoustic_class=CLASS_SIREN)
    assert policy.siren_cap([cue]) == policy.siren_speed_cap
    horn = SimpleNamespace(acoustic_class=CLASS_HORN)
    assert policy.siren_cap([horn]) is None
    assert policy.siren_cap([]) is None


def test_policy_creep_release_requires_unmatched_cue():
    policy = AudioPlannerPolicy()
    cue = AttentionCue(t=1.0, acoustic_class=CLASS_HORN, azimuth_rad=0.0,
                       confidence=0.9, t_onset=0.5)

    # Visual track inside the cone -> no release (nothing unseen approaching).
    assert not policy.creep_release([cue], visual_tracks_azimuth=[0.1])
    # No visual track near the cue azimuth -> release.
    assert policy.creep_release([cue], visual_tracks_azimuth=[2.5])
    # No cues at all -> no release.
    assert not policy.creep_release([], visual_tracks_azimuth=[])


def test_policy_track_azimuth_math():
    policy = AudioPlannerPolicy()
    ego_pose = SimpleNamespace(x=0.0, y=0.0, heading=math.pi / 2)
    # Object straight to the left of the ego -> azimuth 0 in base_link.
    az = policy.track_azimuth((0.0, 5.0), ego_pose)
    assert math.isclose(az, 0.0, abs_tol=1e-9)

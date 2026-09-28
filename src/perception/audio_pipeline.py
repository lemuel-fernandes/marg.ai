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
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from src.common.types.sensor import SensorFrame


# Acoustic event classes (kept as plain strings: no ML class enum yet).
CLASS_SIREN = "siren"
CLASS_HORN = "horn"
CLASS_RICKSHAW_HOOTER = "rickshaw_hooter"

# Indian-specific acoustic classes
CLASS_AUTO_RICKSHAW = "auto_rickshaw"
CLASS_CATTLE_BELL = "cattle_bell"
CLASS_LEVEL_CROSSING_BELL = "level_crossing_bell"
CLASS_MUSICAL_HORN = "musical_horn"
CLASS_TWO_WHEELER = "two_wheeler"
CLASS_CONSTRUCTION = "construction"

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


# Speed of sound in m/s at 20°C
SPEED_OF_SOUND = 343.0


class MicrophoneArray:
    """Microphone array abstraction for DOA estimation via GCC-PHAT.
    
    Supports 2-8 microphones in arbitrary 2D geometry. Provides
    Generalized Cross-Correlation with Phase Transform (GCC-PHAT)
    for Time Difference of Arrival (TDOA) estimation.
    """

    def __init__(self, mic_positions: List[Tuple[float, float]], 
                 sample_rate: int = 16000,
                 fft_size: int = 512,
                 speed_of_sound: float = SPEED_OF_SOUND):
        """
        Args:
            mic_positions: List of (x, y) positions in meters relative to array center.
                          x = forward, y = left (base_link convention)
            sample_rate: Audio sample rate in Hz
            fft_size: FFT size for cross-correlation
            speed_of_sound: Speed of sound in m/s
        """
        self.mic_positions = np.array(mic_positions, dtype=np.float64)
        self.num_mics = len(mic_positions)
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.speed_of_sound = speed_of_sound
        
        # Precompute microphone pair indices
        self.mic_pairs = [(i, j) for i in range(self.num_mics) 
                          for j in range(i + 1, self.num_mics)]
        
        # Frequency bins
        self.freq_bins = np.fft.rfftfreq(fft_size, 1.0 / sample_rate)
        
        # PHAT weighting (inverse magnitude spectrum)
        self.phat_weight = 1.0 / (np.abs(self.freq_bins) + 1e-10)

    def gcc_phat(self, sig1: np.ndarray, sig2: np.ndarray, 
                 max_tau: Optional[float] = None) -> Tuple[float, float]:
        """
        Compute GCC-PHAT between two microphone signals.
        
        Args:
            sig1: First microphone signal
            sig2: Second microphone signal
            max_tau: Maximum time delay to search (seconds)
            
        Returns:
            (tau_est, peak_value): Estimated time delay and correlation peak
        """
        n = self.fft_size
        
        # Compute FFT
        SIG1 = np.fft.rfft(sig1, n=n)
        SIG2 = np.fft.rfft(sig2, n=n)
        
        # Cross-power spectrum with PHAT weighting
        R = SIG1 * np.conj(SIG2)
        R = R / (np.abs(R) + 1e-10)  # PHAT normalization
        
        # Inverse FFT to get cross-correlation
        cc = np.fft.irfft(R, n=n)
        
        # Search range for time delay
        max_samples = int(max_tau * self.sample_rate) if max_tau else n // 2
        max_samples = min(max_samples, n // 2)
        
        # Find peak in [-max_samples, max_samples]
        cc_shifted = np.concatenate([cc[-max_samples:], cc[:max_samples+1]])
        peak_idx = np.argmax(cc_shifted)
        tau_est = (peak_idx - max_samples) / self.sample_rate
        peak_val = cc_shifted[peak_idx]
        
        return tau_est, peak_val

    def estimate_doa(self, signals: List[np.ndarray], 
                     max_tau: Optional[float] = None) -> Tuple[float, float]:
        """
        Estimate Direction of Arrival using SRP-PHAT (Steered Response Power).
        
        Args:
            signals: List of microphone signals (one per mic)
            max_tau: Maximum time delay to consider
            
        Returns:
            (azimuth_rad, confidence): DOA azimuth in base_link frame (0=forward, +left)
        """
        if len(signals) != self.num_mics:
            raise ValueError(f"Expected {self.num_mics} signals, got {len(signals)}")
        
        # Grid search over azimuth angles
        az_grid = np.linspace(-np.pi, np.pi, 361)  # 1-degree resolution
        srp_map = np.zeros_like(az_grid)
        
        for i, az in enumerate(az_grid):
            power = 0.0
            for mic_i, mic_j in self.mic_pairs:
                # Expected TDOA for this azimuth
                xi, yi = self.mic_positions[mic_i]
                xj, yj = self.mic_positions[mic_j]
                
                # Direction vector (forward = 0, left = +)
                dx = math.cos(az)
                dy = math.sin(az)
                
                # Project mic baseline onto direction
                baseline_x = xi - xj
                baseline_y = yi - yj
                expected_tau = (baseline_x * dx + baseline_y * dy) / self.speed_of_sound
                
                # Get GCC-PHAT for this pair
                tau_est, peak = self.gcc_phat(signals[mic_i], signals[mic_j], max_tau)
                
                # Weight by how well measured TDOA matches expected
                if abs(expected_tau) <= max_tau if max_tau else True:
                    power += peak * math.exp(-0.5 * ((tau_est - expected_tau) / 0.001)**2)
            
            srp_map[i] = power
        
        # Find peak
        peak_idx = np.argmax(srp_map)
        best_az = az_grid[peak_idx]
        confidence = float(srp_map[peak_idx] / (np.max(srp_map) + 1e-10))
        
        return best_az, confidence

    def compute_tdoa_matrix(self, signals: List[np.ndarray],
                            max_tau: Optional[float] = None) -> np.ndarray:
        """
        Compute TDOA matrix for all microphone pairs.
        
        Returns:
            Matrix of shape (num_mics, num_mics) with TDOA estimates
        """
        tdoa = np.zeros((self.num_mics, self.num_mics))
        for i, j in self.mic_pairs:
            tau, _ = self.gcc_phat(signals[i], signals[j], max_tau)
            tdoa[i, j] = tau
            tdoa[j, i] = -tau
        return tdoa


class SoundSourceTracker:
    """Multi-source acoustic tracker using Kalman filtering on DOA estimates."""
    
    def __init__(self, max_sources: int = 5, process_noise: float = 0.1,
                 measurement_noise: float = 0.05):
        self.max_sources = max_sources
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Track state: list of dicts with keys:
        # - id: track ID
        # - az: current azimuth estimate (rad)
        # - az_var: azimuth variance
        # - class_probs: dict of class -> probability
        # - last_update: timestamp
        # - snr_db: estimated SNR
        self.tracks = []
        self._next_id = 1

    def predict(self, dt: float) -> None:
        """Predict step: propagate track states forward in time."""
        for track in self.tracks:
            # Azimuth random walk
            track['az_var'] += self.process_noise * dt

    def update(self, measurements: List[Tuple[float, float, str, float]], 
               current_time: float) -> None:
        """
        Update tracks with new measurements.
        
        Args:
            measurements: List of (azimuth_rad, confidence, acoustic_class, snr_db)
            current_time: Current simulation time
        """
        # Simple nearest-neighbor association
        used_tracks = set()
        
        for az_meas, conf, cls, snr in measurements:
            best_track = None
            best_dist = float('inf')
            
            for i, track in enumerate(self.tracks):
                if i in used_tracks:
                    continue
                # Angular distance
                delta = abs(math.atan2(math.sin(az_meas - track['az']),
                                       math.cos(az_meas - track['az'])))
                if delta < best_dist and delta < 0.5:  # ~30 deg gate
                    best_dist = delta
                    best_track = i
            
            if best_track is not None:
                # Update existing track
                track = self.tracks[best_track]
                used_tracks.add(best_track)
                
                # Kalman-like update
                K = track['az_var'] / (track['az_var'] + self.measurement_noise)
                track['az'] = track['az'] + K * math.atan2(
                    math.sin(az_meas - track['az']),
                    math.cos(az_meas - track['az']))
                track['az_var'] = (1 - K) * track['az_var']
                track['last_update'] = current_time
                
                # Update class probabilities (exponential smoothing)
                alpha = 0.3
                if cls not in track['class_probs']:
                    track['class_probs'][cls] = 0.0
                track['class_probs'][cls] = (1 - alpha) * track['class_probs'][cls] + alpha * conf
                
                # Update SNR estimate
                track['snr_db'] = 0.9 * track['snr_db'] + 0.1 * snr
            else:
                # Create new track
                if len(self.tracks) < self.max_sources:
                    self.tracks.append({
                        'id': self._next_id,
                        'az': az_meas,
                        'az_var': self.measurement_noise,
                        'class_probs': {cls: conf},
                        'last_update': current_time,
                        'snr_db': snr,
                    })
                    self._next_id += 1
        
        # Remove stale tracks (not updated for > 2 seconds)
        self.tracks = [t for t in self.tracks 
                       if current_time - t['last_update'] <= 2.0]

    def get_active_tracks(self, current_time: float, 
                          min_confidence: float = 0.5) -> List[dict]:
        """Get tracks that are currently active and confident."""
        active = []
        for track in self.tracks:
            if current_time - track['last_update'] <= 0.5:  # recent update
                best_class = max(track['class_probs'], key=track['class_probs'].get)
                best_conf = track['class_probs'][best_class]
                if best_conf >= min_confidence:
                    active.append({
                        'id': track['id'],
                        'azimuth_rad': track['az'],
                        'acoustic_class': best_class,
                        'confidence': best_conf,
                        'snr_db': track['snr_db'],
                    })
        return active


def compute_propagation_loss(distance: float, frequency: float = 1000.0,
                             humidity: float = 0.5, temperature: float = 20.0) -> float:
    """
    Compute atmospheric attenuation of sound.
    
    Simplified model: spherical spreading + atmospheric absorption.
    
    Args:
        distance: Propagation distance in meters
        frequency: Sound frequency in Hz
        humidity: Relative humidity (0-1)
        temperature: Temperature in Celsius
        
    Returns:
        Attenuation in dB
    """
    if distance <= 0:
        return 0.0
    
    # Spherical spreading loss
    spreading_loss = 20 * math.log10(distance)
    
    # Atmospheric absorption (ISO 9613-1 simplified)
    # Absorption coefficient in dB/m
    f = frequency / 1000.0  # kHz
    alpha = 0.01 * f**1.7 * (1 + humidity * 0.1) * (temperature / 20.0)**0.5
    absorption_loss = alpha * distance
    
    return spreading_loss + absorption_loss


def compute_doppler_shift(source_velocity: Tuple[float, float],
                          observer_velocity: Tuple[float, float],
                          source_position: Tuple[float, float],
                          observer_position: Tuple[float, float],
                          emitted_freq: float) -> float:
    """
    Compute Doppler frequency shift for moving source and observer.
    
    Args:
        source_velocity: (vx, vy) of source in m/s
        observer_velocity: (vx, vy) of observer in m/s
        source_position: (x, y) of source in m
        observer_position: (x, y) of observer in m
        emitted_freq: Emitted frequency in Hz
        
    Returns:
        Observed frequency in Hz
    """
    # Vector from observer to source
    dx = source_position[0] - observer_position[0]
    dy = source_position[1] - observer_position[1]
    dist = math.hypot(dx, dy)
    
    if dist == 0:
        return emitted_freq
    
    # Unit vector from observer to source
    ux, uy = dx / dist, dy / dist
    
    # Radial velocities
    vr_source = source_velocity[0] * ux + source_velocity[1] * uy
    vr_observer = observer_velocity[0] * ux + observer_velocity[1] * uy
    
    # Doppler shift: f_observed = f_emitted * (c + vr_observer) / (c - vr_source)
    observed_freq = emitted_freq * (SPEED_OF_SOUND + vr_observer) / (SPEED_OF_SOUND - vr_source)
    
    return observed_freq


def estimate_snr_at_distance(source_level_db: float, distance: float,
                             ambient_noise_db: float = 40.0,
                             frequency: float = 1000.0) -> float:
    """
    Estimate SNR at a given distance from source.
    
    Args:
        source_level_db: Source sound pressure level at 1m (dB SPL)
        distance: Distance from source in meters
        ambient_noise_db: Ambient noise floor (dB SPL)
        frequency: Dominant frequency for absorption calculation
        
    Returns:
        SNR in dB
    """
    loss = compute_propagation_loss(distance, frequency)
    received_level = source_level_db - loss
    snr = received_level - ambient_noise_db
    return max(snr, -20.0)  # Floor at -20 dB


class AcousticPerceptionNode:
    """Synthetic acoustic frontend: events -> attention cues.

    Implements the `PerceptionNode` call shape (`process(frame)`) but returns
    a list of `AttentionCue`s rather than `PerceptionOutput` — acoustic cues
    are attention metadata, not obstacle hypotheses, so they never enter the
    costmap.
    """

    def __init__(self, events: Optional[List[AcousticEvent]] = None,
                 persistence_s: float = PERSISTENCE_S,
                 detect_snr_db: float = DETECT_SNR_DB,
                 mic_array: Optional[MicrophoneArray] = None,
                 enable_tracking: bool = False):
        self.events = list(events or [])
        self.persistence_s = persistence_s
        self.detect_snr_db = detect_snr_db
        # Mounting miscalibration (rad): added to every reported azimuth.
        self._yaw_offset_rad = 0.0
        
        # Microphone array for DOA estimation (optional)
        self.mic_array = mic_array or MicrophoneArray([
            (-0.04, -0.04), (0.04, -0.04),  # Front pair
            (-0.04, 0.04), (0.04, 0.04),    # Rear pair
        ])
        
        # Multi-source tracker
        self.tracker = SoundSourceTracker() if enable_tracking else None
        
        # Ambient noise estimate (dB SPL)
        self.ambient_noise_db = 40.0
        
        # Source levels for different classes (dB SPL at 1m)
        self.source_levels = {
            CLASS_SIREN: 120.0,
            CLASS_HORN: 110.0,
            CLASS_RICKSHAW_HOOTER: 100.0,
            CLASS_AUTO_RICKSHAW: 95.0,
            CLASS_CATTLE_BELL: 85.0,
            CLASS_LEVEL_CROSSING_BELL: 105.0,
            CLASS_MUSICAL_HORN: 115.0,
            CLASS_TWO_WHEELER: 90.0,
            CLASS_CONSTRUCTION: 100.0,
        }
        
        # Dominant frequencies for different classes (Hz)
        self.source_frequencies = {
            CLASS_SIREN: 800.0,
            CLASS_HORN: 500.0,
            CLASS_RICKSHAW_HOOTER: 600.0,
            CLASS_AUTO_RICKSHAW: 150.0,  # Engine fundamental
            CLASS_CATTLE_BELL: 400.0,
            CLASS_LEVEL_CROSSING_BELL: 800.0,
            CLASS_MUSICAL_HORN: 440.0,
            CLASS_TWO_WHEELER: 200.0,
            CLASS_CONSTRUCTION: 1000.0,
        }

    def update_imu_yaw_offset(self, yaw_offset_rad: float) -> None:
        """Mirror of the vision pipeline's miscalibration compensation hook."""
        self._yaw_offset_rad = float(yaw_offset_rad)

    def update_ambient_noise(self, noise_db: float) -> None:
        """Update ambient noise estimate."""
        self.ambient_noise_db = noise_db

    def set_mic_array(self, mic_array: MicrophoneArray) -> None:
        """Set microphone array geometry."""
        self.mic_array = mic_array

    def process(self, frame: SensorFrame) -> List[AttentionCue]:
        t = frame.header.stamp

        cues: List[AttentionCue] = []
        measurements = []  # For tracker
        
        for ev in self.events:
            # Detection gate: too faint -> never detected.
            if ev.snr_db < self.detect_snr_db:
                continue
            
            if ev.t_onset <= t <= ev.t_end:
                # Compute Doppler shift if we had velocity info
                # (simplified: just use the event's azimuth and SNR)
                az = ev.azimuth_rad + self._yaw_offset_rad
                conf = _confidence_from_snr(ev.snr_db)
                
                cues.append(AttentionCue(
                    t=t,
                    acoustic_class=ev.acoustic_class,
                    azimuth_rad=az,
                    confidence=conf,
                    t_onset=ev.t_onset,
                    is_persisted=False,
                ))
                measurements.append((az, conf, ev.acoustic_class, ev.snr_db))
                
            elif ev.t_end < t <= ev.t_end + self.persistence_s:
                # Persistence: cue survives after the sound decays.
                az = ev.azimuth_rad + self._yaw_offset_rad
                conf = _confidence_from_snr(ev.snr_db) * 0.8
                
                cues.append(AttentionCue(
                    t=t,
                    acoustic_class=ev.acoustic_class,
                    azimuth_rad=az,
                    confidence=conf,
                    t_onset=ev.t_onset,
                    is_persisted=True,
                ))
                measurements.append((az, conf, ev.acoustic_class, ev.snr_db))
        
        # Update tracker if enabled
        if self.tracker is not None:
            self.tracker.predict(0.1)  # dt = 0.1s
            self.tracker.update(measurements, t)
            
            # Add tracked sources that aren't in current events
            tracked = self.tracker.get_active_tracks(t)
            for trk in tracked:
                # Check if already covered by direct event
                covered = any(abs(math.atan2(
                    math.sin(c.azimuth_rad - trk['azimuth_rad']),
                    math.cos(c.azimuth_rad - trk['azimuth_rad']))) < 0.2
                    for c in cues if c.acoustic_class == trk['acoustic_class'])
                if not covered:
                    cues.append(AttentionCue(
                        t=t,
                        acoustic_class=trk['acoustic_class'],
                        azimuth_rad=trk['azimuth_rad'],
                        confidence=trk['confidence'],
                        t_onset=t - 0.1,  # approximate
                        is_persisted=True,
                    ))
        
        return cues

    def process_with_mic_array(self, frame: SensorFrame,
                               mic_signals: List[np.ndarray]) -> List[AttentionCue]:
        """
        Process using actual microphone array signals for DOA estimation.
        
        Args:
            frame: Sensor frame with timestamp
            mic_signals: List of microphone audio signals (one per mic)
            
        Returns:
            List of AttentionCue with DOA estimated from mic array
        """
        t = frame.header.stamp
        
        # Estimate DOA from microphone array
        if self.mic_array is not None and len(mic_signals) == self.mic_array.num_mics:
            az, confidence = self.mic_array.estimate_doa(mic_signals)
            az += self._yaw_offset_rad
            
            # For synthetic events, we still use event-based classification
            # In a real system, this would be replaced by a CNN classifier
            cues = []
            for ev in self.events:
                if ev.t_onset <= t <= ev.t_end and ev.snr_db >= self.detect_snr_db:
                    # Use array-estimated DOA but event-provided class
                    cues.append(AttentionCue(
                        t=t,
                        acoustic_class=ev.acoustic_class,
                        azimuth_rad=az,
                        confidence=confidence * _confidence_from_snr(ev.snr_db),
                        t_onset=ev.t_onset,
                        is_persisted=False,
                    ))
            return cues
        
        # Fallback to event-based
        return self.process(frame)


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
                                           CLASS_RICKSHAW_HOOTER,
                                           CLASS_MUSICAL_HORN,
                                           CLASS_TWO_WHEELER)]
        if not audible:
            return False
        return not any(self.matching_visual_track(c, visual_tracks_azimuth)
                       for c in audible)

    def get_azimuth_based_speed_cap(self, cues: List[AttentionCue]) -> Optional[float]:
        """Get speed cap based on azimuth - stricter for rear/side threats."""
        for c in cues:
            if c.acoustic_class in (CLASS_SIREN, CLASS_LEVEL_CROSSING_BELL):
                # Rear threats (ambulance overtaking, train crossing)
                if abs(c.azimuth_rad) > math.pi / 2:  # Behind
                    return 1.0  # Very conservative
                else:
                    return self.siren_speed_cap
            elif c.acoustic_class in (CLASS_HORN, CLASS_MUSICAL_HORN, CLASS_TWO_WHEELER):
                # Side/rear horns indicate overtaking intent
                if abs(c.azimuth_rad) > math.pi / 4:  # Not directly ahead
                    return 2.5
        return None


class HornIntentAnalyzer:
    """Analyzes horn signals to infer intent of other road users.
    
    Indian traffic context: Horns are used as communication, not just warning.
    - Short beep (0.1-0.3s): "I'm here" / greeting / light warning
    - Sustained (0.5-2s): "Warning" / "Move" / "Overtaking"
    - Repeated short: "Urgent" / "Aggressive overtaking"
    - Musical horn: "Aggressive" / "Make way"
    - Two-wheeler beep: "Lane splitting" / "Passing"
    """
    
    # Intent types
    INTENT_WARNING = "warning"           # Defensive: "Watch out, I'm here"
    INTENT_OVERTAKING = "overtaking"     # Assertive: "I'm passing, make space"
    INTENT_CROSSING = "crossing"         # "I'm crossing your path"
    INTENT_AGGRESSIVE = "aggressive"     # "Get out of my way"
    INTENT_LANE_SPLIT = "lane_split"     # Two-wheeler passing between lanes
    INTENT_EMERGENCY = "emergency"       # Siren: emergency vehicle
    INTENT_UNKNOWN = "unknown"
    
    def __init__(self):
        # Track recent horn events for pattern analysis
        self.horn_history = []  # List of (t, class, azimuth, duration)
        self.max_history = 10
    
    def analyze_horn_intent(self, cues: List[AttentionCue], 
                            visual_tracks: List[Dict] = None,
                            ego_speed: float = 0.0) -> Dict[str, Any]:
        """
        Analyze horn cues and infer intent.
        
        Args:
            cues: Current AttentionCues
            visual_tracks: List of visual tracks with 'azimuth', 'distance', 'class', 'speed'
            ego_speed: Current ego vehicle speed (m/s)
            
        Returns:
            Dict with 'primary_intent', 'confidence', 'recommended_action', 'details'
        """
        horn_cues = [c for c in cues 
                      if c.acoustic_class in (CLASS_HORN, CLASS_MUSICAL_HORN, 
                                              CLASS_TWO_WHEELER, CLASS_RICKSHAW_HOOTER,
                                              CLASS_SIREN, CLASS_LEVEL_CROSSING_BELL)]
        
        if not horn_cues:
            return {
                'primary_intent': self.INTENT_UNKNOWN,
                'confidence': 0.0,
                'recommended_action': None,
                'details': {}
            }
        
        # Analyze each horn cue
        intents = []
        for cue in horn_cues:
            intent_result = self._analyze_single_horn(cue, visual_tracks, ego_speed)
            intents.append(intent_result)
            
            # Add to history
            self.horn_history.append({
                't': cue.t,
                'class': cue.acoustic_class,
                'azimuth': cue.azimuth_rad,
                'duration': getattr(cue, 'duration', 0.5),  # Default 0.5s
                'intent': intent_result['intent'],
            })
        
        # Keep history bounded
        if len(self.horn_history) > self.max_history:
            self.horn_history = self.horn_history[-self.max_history:]
        
        # Determine primary intent (highest confidence)
        primary = max(intents, key=lambda x: x['confidence'])
        
        return {
            'primary_intent': primary['intent'],
            'confidence': primary['confidence'],
            'recommended_action': primary['action'],
            'details': {
                'all_intents': intents,
                'horn_count': len(horn_cues),
                'azimuths': [c.azimuth_rad for c in horn_cues],
                'classes': [c.acoustic_class for c in horn_cues],
            }
        }
    
    def _analyze_single_horn(self, cue: AttentionCue, 
                             visual_tracks: List[Dict] = None,
                             ego_speed: float = 0.0) -> Dict[str, Any]:
        """Analyze a single horn cue for intent."""
        az = cue.azimuth_rad
        horn_class = cue.acoustic_class
        confidence = cue.confidence
        
        # Determine azimuth sector
        # Behind: |az| > 135 deg (2.35 rad)
        # Rear-side: 90-135 deg (1.57-2.35 rad) 
        # Side: 45-90 deg (0.78-1.57 rad)
        # Front-side: 0-45 deg (0-0.78 rad)
        # Front: 0 deg
        abs_az = abs(az)
        is_behind = abs_az > 2.35
        is_rear_side = abs_az > 1.57
        is_side = abs_az > 0.78
        
        # Check for visual track in horn direction
        visual_in_sector = False
        visual_distance = float('inf')
        visual_speed = 0.0
        if visual_tracks:
            for vt in visual_tracks:
                vt_az = vt.get('azimuth', 0)
                vt_dist = vt.get('distance', float('inf'))
                if abs(vt_az - az) < 0.5:  # Within ~30 deg
                    visual_in_sector = True
                    visual_distance = min(visual_distance, vt_dist)
                    visual_speed = vt.get('speed', 0.0)
        
        # Analyze based on horn type and direction
        if horn_class == CLASS_SIREN:
            return {
                'intent': self.INTENT_EMERGENCY,
                'confidence': 0.95,
                'action': 'pull_over_stop',
                'details': {'reason': 'Emergency vehicle approaching'}
            }
        
        elif horn_class == CLASS_LEVEL_CROSSING_BELL:
            return {
                'intent': self.INTENT_WARNING,
                'confidence': 0.9,
                'action': 'stop_before_crossing',
                'details': {'reason': 'Train crossing ahead'}
            }
        
        elif horn_class == CLASS_MUSICAL_HORN:
            # Musical horn = aggressive, usually from behind/side
            if is_behind or is_rear_side:
                return {
                    'intent': self.INTENT_AGGRESSIVE,
                    'confidence': 0.85,
                    'action': 'yield_lane_maintain_speed',
                    'details': {'reason': 'Aggressive overtaking from behind', 'sector': 'rear'}
                }
            else:
                return {
                    'intent': self.INTENT_AGGRESSIVE,
                    'confidence': 0.7,
                    'action': 'yield_slow_down',
                    'details': {'reason': 'Aggressive crossing from front/side'}
                }
        
        elif horn_class == CLASS_TWO_WHEELER:
            # Two-wheeler = lane splitting or close passing
            if is_side or is_rear_side:
                return {
                    'intent': self.INTENT_LANE_SPLIT,
                    'confidence': 0.8,
                    'action': 'hold_lane_create_gap',
                    'details': {'reason': 'Two-wheeler lane splitting', 'sector': 'side'}
                }
            elif is_behind:
                return {
                    'intent': self.INTENT_OVERTAKING,
                    'confidence': 0.75,
                    'action': 'maintain_lane',
                    'details': {'reason': 'Two-wheeler overtaking from behind'}
                }
            else:
                return {
                    'intent': self.INTENT_CROSSING,
                    'confidence': 0.7,
                    'action': 'slow_down_yield',
                    'details': {'reason': 'Two-wheeler crossing from front'}
                }
        
        elif horn_class == CLASS_RICKSHAW_HOOTER:
            # Auto-rickshaw = usually warning or overtaking
            if is_behind:
                return {
                    'intent': self.INTENT_OVERTAKING,
                    'confidence': 0.7,
                    'action': 'maintain_lane_allow_pass',
                    'details': {'reason': 'Auto-rickshaw overtaking'}
                }
            else:
                return {
                    'intent': self.INTENT_WARNING,
                    'confidence': 0.6,
                    'action': 'caution',
                    'details': {'reason': 'Auto-rickshaw warning'}
                }
        
        else:  # CLASS_HORN (regular horn)
            # Duration-based analysis (would need cue.duration attribute)
            # Default to warning for front, overtaking for rear
            if is_behind:
                # Check for visual track behind
                if visual_in_sector and visual_distance < 30.0:
                    if visual_speed > ego_speed + 2.0:
                        return {
                            'intent': self.INTENT_OVERTAKING,
                            'confidence': 0.8,
                            'action': 'maintain_lane_allow_pass',
                            'details': {'reason': 'Faster vehicle overtaking from behind'}
                        }
                return {
                    'intent': self.INTENT_OVERTAKING,
                    'confidence': 0.6,
                    'action': 'maintain_lane',
                    'details': {'reason': 'Possible overtaking from behind'}
                }
            elif is_rear_side:
                return {
                    'intent': self.INTENT_OVERTAKING,
                    'confidence': 0.65,
                    'action': 'hold_lane_create_gap',
                    'details': {'reason': 'Vehicle approaching from rear-side'}
                }
            elif is_side:
                return {
                    'intent': self.INTENT_CROSSING,
                    'confidence': 0.6,
                    'action': 'slow_down_yield',
                    'details': {'reason': 'Vehicle crossing from side'}
                }
            else:  # Front
                return {
                    'intent': self.INTENT_WARNING,
                    'confidence': 0.5,
                    'action': 'caution',
                    'details': {'reason': 'Warning from front'}
                }
    
    def get_recommended_planner_params(self, intent_analysis: Dict) -> Dict[str, float]:
        """Convert intent analysis to planner parameter adjustments."""
        intent = intent_analysis.get('primary_intent', self.INTENT_UNKNOWN)
        confidence = intent_analysis.get('confidence', 0.0)
        
        if confidence < 0.4:
            return {}  # Low confidence, no action
        
        params = {}
        
        if intent == self.INTENT_EMERGENCY:
            params['speed_cap'] = 0.5  # Near stop
            params['lateral_buffer'] = 3.0  # Maximum clearance
            params['urgency'] = 'critical'
            
        elif intent == self.INTENT_AGGRESSIVE:
            params['speed_cap'] = 2.0  # Slow down
            params['lateral_buffer'] = 1.5  # Give space
            params['allow_lane_change'] = False  # Don't change lanes
            params['urgency'] = 'high'
            
        elif intent == self.INTENT_OVERTAKING:
            params['speed_cap'] = 3.0  # Maintain moderate speed
            params['lateral_buffer'] = 1.0  # Standard clearance
            params['allow_lane_change'] = False
            params['urgency'] = 'medium'
            
        elif intent == self.INTENT_LANE_SPLIT:
            params['speed_cap'] = 2.5
            params['lateral_buffer'] = 0.5  # Minimal gap for two-wheeler
            params['allow_lane_change'] = False
            params['urgency'] = 'medium'
            
        elif intent == self.INTENT_CROSSING:
            params['speed_cap'] = 1.5
            params['lateral_buffer'] = 1.5
            params['urgency'] = 'high'
            
        elif intent == self.INTENT_WARNING:
            params['speed_cap'] = 4.0
            params['lateral_buffer'] = 1.0
            params['urgency'] = 'low'
            
        elif intent == self.INTENT_CROSSING:
            params['speed_cap'] = 0.0  # Stop
            params['lateral_buffer'] = 2.0
            params['urgency'] = 'critical'
        
        return params


# Add horn duration tracking to AttentionCue
# (We'll extend the dataclass with a duration field)
# For now, we'll use a simple heuristic: if confidence > 0.8 and is_persisted,
# it's likely a sustained horn


class AcousticSceneClassifier:
    """
    Classifies the acoustic scene/environment based on detected sound events.
    
    Indian acoustic scenes:
    - HIGHWAY: Continuous vehicle noise, occasional horns
    - URBAN: Mixed traffic, frequent horns, pedestrian sounds
    - MARKET: High pedestrian density, vendor calls, two-wheelers
    - VILLAGE: Low traffic, animal sounds, tractor
    - INDUSTRIAL: Construction, heavy vehicles
    - RESIDENTIAL: Quiet, occasional vehicles
    - EMERGENCY: Siren present
    - LEVEL_CROSSING: Warning bells
    """
    
    SCENES = [
        'highway', 'urban', 'market', 'village', 
        'industrial', 'residential', 'emergency', 'level_crossing'
    ]
    
    # Scene signatures: event class -> (min_count, max_count, weight)
    SCENE_SIGNATURES = {
        'highway': {
            'two_wheeler': (2, 10, 1.0),
            'car': (3, 15, 1.2),
            'truck': (1, 5, 1.5),
            'horn': (0, 3, 0.5),
        },
        'urban': {
            'car': (2, 10, 1.0),
            'auto_rickshaw': (2, 8, 1.5),
            'two_wheeler': (3, 12, 1.2),
            'horn': (2, 8, 1.0),
            'pedestrian': (1, 5, 0.5),
        },
        'market': {
            'auto_rickshaw': (1, 5, 1.0),
            'two_wheeler': (2, 10, 1.2),
            'pedestrian': (3, 15, 1.5),
            'horn': (3, 10, 1.0),
            'vendor_call': (1, 5, 2.0),  # Not implemented yet
        },
        'village': {
            'tractor': (1, 3, 2.0),
            'cattle_bell': (1, 5, 2.0),
            'car': (0, 3, 0.5),
            'two_wheeler': (1, 5, 0.8),
        },
        'industrial': {
            'construction': (2, 8, 2.0),
            'truck': (2, 6, 1.5),
            'horn': (1, 4, 0.8),
        },
        'residential': {
            'car': (1, 5, 1.0),
            'two_wheeler': (1, 4, 0.8),
            'dog': (1, 3, 0.5),
            'horn': (0, 2, 0.3),
        },
        'emergency': {
            'siren': (1, 3, 3.0),
        },
        'level_crossing': {
            'level_crossing_bell': (1, 2, 3.0),
            'horn': (0, 2, 0.5),
        },
    }
    
    def __init__(self, window_duration: float = 10.0):
        self.window_duration = window_duration
        self.event_history = []  # List of (t, acoustic_class, azimuth, confidence)
    
    def add_event(self, cue: 'AttentionCue'):
        """Add an acoustic cue to history."""
        self.event_history.append({
            't': cue.t,
            'class': cue.acoustic_class,
            'azimuth': cue.azimuth_rad,
            'confidence': cue.confidence,
        })
        # Prune old events
        self._prune_history(cue.t)
    
    def _prune_history(self, current_time: float):
        """Remove events older than window_duration."""
        cutoff = current_time - self.window_duration
        self.event_history = [e for e in self.event_history if e['t'] >= cutoff]
    
    def classify_scene(self, current_time: float) -> Dict[str, float]:
        """
        Classify current acoustic scene.
        
        Returns:
            Dict mapping scene name -> probability (0-1)
        """
        self._prune_history(current_time)
        
        if not self.event_history:
            return {'residential': 1.0}  # Default to quiet
        
        # Count events by class in current window
        class_counts = {}
        for e in self.event_history:
            cls = e['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Score each scene
        scene_scores = {}
        for scene, signature in self.SCENE_SIGNATURES.items():
            score = 0.0
            total_weight = 0.0
            for cls, (min_c, max_c, weight) in signature.items():
                count = class_counts.get(cls, 0)
                if count >= min_c:
                    # Score based on how close to expected range
                    if count <= max_c:
                        score += weight * 1.0
                    else:
                        # Too many - reduce score
                        score += weight * max(0.0, 1.0 - (count - max_c) / max_c)
                    total_weight += weight
            
            if total_weight > 0:
                scene_scores[scene] = score / total_weight
            else:
                scene_scores[scene] = 0.0
        
        # Normalize to probabilities
        total = sum(scene_scores.values())
        if total > 0:
            scene_scores = {k: v / total for k, v in scene_scores.items()}
        else:
            scene_scores = {'residential': 1.0}
        
        return scene_scores
    
    def get_dominant_scene(self, current_time: float) -> str:
        """Get the most likely scene."""
        scores = self.classify_scene(current_time)
        return max(scores, key=scores.get)
    
    def get_scene_parameters(self, scene: str) -> Dict:
        """Get planner parameters for a given scene."""
        params = {
            'highway': {'target_speed': 22.22, 'following_distance': 3.0, 'lane_changes': True},
            'urban': {'target_speed': 11.11, 'following_distance': 2.0, 'lane_changes': True},
            'market': {'target_speed': 5.56, 'following_distance': 1.0, 'lane_changes': False},
            'village': {'target_speed': 8.33, 'following_distance': 1.5, 'lane_changes': True},
            'industrial': {'target_speed': 8.33, 'following_distance': 2.0, 'lane_changes': True},
            'residential': {'target_speed': 8.33, 'following_distance': 2.0, 'lane_changes': True},
            'emergency': {'target_speed': 0.5, 'following_distance': 5.0, 'lane_changes': False},
            'level_crossing': {'target_speed': 0.0, 'following_distance': 10.0, 'lane_changes': False},
        }
        return params.get(scene, params['urban'])
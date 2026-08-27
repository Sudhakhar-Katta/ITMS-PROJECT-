import math

import numpy as np


class DistractionMonitor:
    """
    Driver distraction / looking-away monitor.

    Works in two modes:

    1. gaze_plus_head_pose:
       - Uses gaze score when eyes are visible.
       - Also uses head pose.

    2. head_pose_only:
       - Used when sunglasses/glare/occlusion make gaze unreliable.
       - Uses face direction only.

    Requirement:
        Trigger if driver is looking away for >= distraction_seconds.
    """

    def __init__(
        self,
        gaze_thresh=0.20,
        lookaway_yaw_thresh=20.0,
        lookaway_pitch_thresh=18.0,
        severe_yaw_thresh=30.0,
        severe_pitch_thresh=25.0,
        severe_roll_thresh=25.0,
        distraction_seconds=5.0,
        max_signal_gap_seconds=0.25,
        smoothing_alpha=0.35,
        calibration_seconds=2.0,
    ):
        self.gaze_thresh = gaze_thresh

        self.lookaway_yaw_thresh = lookaway_yaw_thresh
        self.lookaway_pitch_thresh = lookaway_pitch_thresh

        self.severe_yaw_thresh = severe_yaw_thresh
        self.severe_pitch_thresh = severe_pitch_thresh
        self.severe_roll_thresh = severe_roll_thresh

        self.distraction_seconds = distraction_seconds
        self.max_signal_gap_seconds = max_signal_gap_seconds
        self.smoothing_alpha = smoothing_alpha
        self.calibration_seconds = calibration_seconds

        self.calibration_start = None
        self.calibration_samples = []
        self.calibrated = False

        self.base_roll = 0.0
        self.base_pitch = 0.0
        self.base_yaw = 0.0

        self.smooth_roll = None
        self.smooth_pitch = None
        self.smooth_yaw = None

        self.away_start = None
        self.last_away_seen = None
        self.alerted_for_episode = False

    def reset_active_continuity(self):
        """Reset only the current lookaway episode after face loss."""
        self.away_start = None
        self.last_away_seen = None
        self.alerted_for_episode = False

    def _to_float(self, value):
        try:
            value = float(np.ravel(value)[0])
            if math.isfinite(value):
                return value
            return None
        except Exception:
            return None

    def _calibrate_or_get_relative_pose(self, t_now, roll, pitch, yaw):
        roll = self._to_float(roll)
        pitch = self._to_float(pitch)
        yaw = self._to_float(yaw)

        if roll is None or pitch is None or yaw is None:
            return None, None, None, False

        if self.calibration_start is None:
            self.calibration_start = t_now

        if not self.calibrated:
            self.calibration_samples.append((roll, pitch, yaw))

            elapsed = t_now - self.calibration_start

            if elapsed >= self.calibration_seconds and len(self.calibration_samples) >= 5:
                arr = np.array(self.calibration_samples)

                self.base_roll = float(np.median(arr[:, 0]))
                self.base_pitch = float(np.median(arr[:, 1]))
                self.base_yaw = float(np.median(arr[:, 2]))

                self.calibrated = True

            return 0.0, 0.0, 0.0, self.calibrated

        rel_roll = roll - self.base_roll
        rel_pitch = pitch - self.base_pitch
        rel_yaw = yaw - self.base_yaw

        raw_away_pose = (
            abs(rel_yaw) >= self.lookaway_yaw_thresh
            or abs(rel_pitch) >= self.lookaway_pitch_thresh
            or abs(rel_roll) >= self.severe_roll_thresh
        )

        if self.smooth_roll is None:
            self.smooth_roll = rel_roll
            self.smooth_pitch = rel_pitch
            self.smooth_yaw = rel_yaw
        elif raw_away_pose:
            self.smooth_roll = rel_roll
            self.smooth_pitch = rel_pitch
            self.smooth_yaw = rel_yaw
        else:
            a = self.smoothing_alpha
            self.smooth_roll = a * rel_roll + (1 - a) * self.smooth_roll
            self.smooth_pitch = a * rel_pitch + (1 - a) * self.smooth_pitch
            self.smooth_yaw = a * rel_yaw + (1 - a) * self.smooth_yaw

        return self.smooth_roll, self.smooth_pitch, self.smooth_yaw, True

    def update(
        self,
        t_now,
        roll,
        pitch,
        yaw,
        gaze_score,
        gaze_reliable,
    ):
        alerts = []

        rel_roll, rel_pitch, rel_yaw, calibrated = self._calibrate_or_get_relative_pose(
            t_now=t_now,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )

        if not calibrated:
            return {
                "away_signal": False,
                "looking_away": False,
                "distracted": False,
                "away_duration": 0.0,
                "distraction_reason": "calibrating_forward_pose",
                "distraction_mode": "calibrating",
                "rel_roll": rel_roll,
                "rel_pitch": rel_pitch,
                "rel_yaw": rel_yaw,
            }, alerts

        gaze_value = self._to_float(gaze_score)

        gaze_away = False
        head_away = False
        severe_pose = False

        if gaze_reliable and gaze_value is not None:
            gaze_away = gaze_value >= self.gaze_thresh

        if abs(rel_yaw) >= self.lookaway_yaw_thresh:
            head_away = True

        if abs(rel_pitch) >= self.lookaway_pitch_thresh:
            head_away = True

        if abs(rel_yaw) >= self.severe_yaw_thresh:
            severe_pose = True

        if abs(rel_pitch) >= self.severe_pitch_thresh:
            severe_pose = True

        if abs(rel_roll) >= self.severe_roll_thresh:
            severe_pose = True

        if gaze_reliable:
            away_signal = gaze_away or head_away or severe_pose
            mode_reason = "gaze_or_head_pose"
        else:
            away_signal = head_away or severe_pose
            mode_reason = "head_pose_only"

        if gaze_away:
            reason = "gaze_away"
        elif severe_pose:
            reason = "severe_head_pose"
        elif head_away:
            reason = "head_away"
        else:
            reason = "forward"

        if away_signal:
            if self.away_start is None:
                self.away_start = t_now
                self.alerted_for_episode = False

            self.last_away_seen = t_now
        else:
            if (
                self.away_start is not None
                and self.last_away_seen is not None
                and (t_now - self.last_away_seen) <= self.max_signal_gap_seconds
            ):
                away_signal = True
            else:
                self.away_start = None
                self.last_away_seen = None
                self.alerted_for_episode = False

        away_duration = 0.0

        if away_signal and self.away_start is not None:
            away_duration = t_now - self.away_start

        looking_away = away_duration >= self.distraction_seconds
        distracted = looking_away

        if looking_away and not self.alerted_for_episode:
            trigger_time = self.away_start + self.distraction_seconds
            detection_delay = t_now - trigger_time

            alerts.append(
                {
                    "event": "driver_distraction",
                    "time_sec": round(t_now, 3),
                    "away_duration": round(away_duration, 3),
                    "detection_delay": round(detection_delay, 3),
                    "reason": reason,
                    "mode": mode_reason,
                    "rel_yaw": round(rel_yaw, 3),
                    "rel_pitch": round(rel_pitch, 3),
                    "rel_roll": round(rel_roll, 3),
                }
            )

            self.alerted_for_episode = True

        states = {
            "away_signal": bool(away_signal),
            "looking_away": looking_away,
            "distracted": distracted,
            "away_duration": round(away_duration, 3),
            "distraction_reason": reason,
            "distraction_mode": mode_reason,
            "rel_roll": round(rel_roll, 3),
            "rel_pitch": round(rel_pitch, 3),
            "rel_yaw": round(rel_yaw, 3),
        }

        return states, alerts

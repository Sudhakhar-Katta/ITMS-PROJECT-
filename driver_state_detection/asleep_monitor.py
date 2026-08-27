import math
import numpy as np


class AsleepMonitor:
    """
    Final ASLEEP decision.

    ASLEEP triggers from either:
    1. Drowsiness active: eyes closed continuously >= 3 sec
    2. Tired + head-down posture held for >= 3 sec

    This fixes cases where the driver sleeps with head down but EAR/gaze
    are unreliable due to glasses, sunglasses, or face angle.
    """

    def __init__(
        self,
        head_down_pitch_thresh=18.0,
        side_slump_roll_thresh=28.0,
        head_down_seconds=3.0,
        max_signal_gap_seconds=0.25,
    ):
        self.head_down_pitch_thresh = head_down_pitch_thresh
        self.side_slump_roll_thresh = side_slump_roll_thresh
        self.head_down_seconds = head_down_seconds
        self.max_signal_gap_seconds = max_signal_gap_seconds

        self.head_down_start = None
        self.last_head_down_seen = None
        self.last_reason = "not_asleep"

    def reset_active_continuity(self):
        """Reset only the in-progress sleeping-pose timer after face loss."""
        self.head_down_start = None
        self.last_head_down_seen = None
        self.last_reason = "not_asleep"

    def _to_float(self, value):
        try:
            value = float(np.ravel(value)[0])
            if math.isfinite(value):
                return value
            return None
        except Exception:
            return None

    def update(
        self, t_now, drowsiness_active, tired, rel_pitch, rel_roll=None, rel_yaw=None
    ):
        alerts = []

        rel_pitch_value = self._to_float(rel_pitch)
        rel_roll_value = self._to_float(rel_roll)

        # Case 1: normal sleep detection from eyes closed >= 3 sec
        if drowsiness_active:
            self.last_reason = "eyes_closed_3s"
            return {
                "asleep": True,
                "sleep_reason": self.last_reason,
                "sleep_pose_signal": False,
                "sleep_pose_duration": 0.0,
                "head_down_duration": 0.0,
            }, alerts

        # Case 2: fallback sleep detection from sustained sleeping posture.
        head_down = False
        side_slump = False

        if rel_pitch_value is not None:
            head_down = abs(rel_pitch_value) >= self.head_down_pitch_thresh

        if rel_roll_value is not None:
            side_slump = abs(rel_roll_value) >= self.side_slump_roll_thresh

        down_side_slump = head_down and side_slump
        sleep_posture_signal = bool(head_down or side_slump)

        if down_side_slump:
            pose_reason = "down_side_slump_sleep_pose"
        elif side_slump:
            pose_reason = "side_slump_sleep_pose"
        elif head_down:
            pose_reason = "head_down_sleep_pose"
        else:
            pose_reason = "not_asleep"

        if sleep_posture_signal:
            if self.head_down_start is None:
                self.head_down_start = t_now

            self.last_head_down_seen = t_now

        else:
            if (
                self.head_down_start is not None
                and self.last_head_down_seen is not None
                and (t_now - self.last_head_down_seen) <= self.max_signal_gap_seconds
            ):
                sleep_posture_signal = True
            else:
                self.head_down_start = None
                self.last_head_down_seen = None

        head_down_duration = 0.0

        if sleep_posture_signal and self.head_down_start is not None:
            head_down_duration = t_now - self.head_down_start

        asleep = head_down_duration >= self.head_down_seconds

        if asleep:
            self.last_reason = pose_reason
        elif sleep_posture_signal:
            self.last_reason = "sleep_pose_counting"
        else:
            self.last_reason = "not_asleep"

        return {
            "asleep": asleep,
            "sleep_reason": self.last_reason,
            "sleep_pose_signal": sleep_posture_signal,
            "sleep_pose_duration": round(head_down_duration, 3),
            "head_down_duration": round(head_down_duration, 3),
        }, alerts

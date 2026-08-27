import math

import numpy as np


class TiredDisplayFilter:
    """
    Tired alert filter.

    Behavior:
    - Normal blink should not show TIRED.
    - Slow blink / short sleep should show TIRED.
    - Heavy eyelids should show TIRED.
    - TIRED should clear quickly after eyes reopen.
    """

    def __init__(
        self,
        tired_on_perclos=0.14,
        tired_off_perclos=0.07,
        critical_perclos=0.30,
        perclos_activation_seconds=1.0,
        slow_closure_seconds=0.55,
        blink_ignore_seconds=0.35,
        heavy_ear_thresh=0.30,
        heavy_ear_seconds=1.0,
        clear_after_open_seconds=1.0,
        min_active_seconds=0.8,
    ):
        self.tired_on_perclos = tired_on_perclos
        self.tired_off_perclos = tired_off_perclos
        self.critical_perclos = critical_perclos
        self.perclos_activation_seconds = perclos_activation_seconds
        self.slow_closure_seconds = slow_closure_seconds
        self.blink_ignore_seconds = blink_ignore_seconds
        self.heavy_ear_thresh = heavy_ear_thresh
        self.heavy_ear_seconds = heavy_ear_seconds
        self.clear_after_open_seconds = clear_after_open_seconds
        self.min_active_seconds = min_active_seconds

        self.active = False
        self.active_start = None

        self.high_perclos_start = None
        self.heavy_ear_start = None
        self.eyes_open_start = None

        self.last_reason = "not_tired"

    def _to_float(self, value):
        try:
            value = float(np.ravel(value)[0])
            if math.isfinite(value):
                return value
            return None
        except Exception:
            return None

    def update(self, t_now, perclos_score, eyes_closed, closed_duration, ear=None):
        perclos = self._to_float(perclos_score)
        ear_value = self._to_float(ear)

        if perclos is None:
            self.last_reason = "no_perclos"
            return self.active

        normal_blink = eyes_closed and closed_duration < self.blink_ignore_seconds
        slow_closure = eyes_closed and closed_duration >= self.slow_closure_seconds

        heavy_ear = False
        if ear_value is not None:
            heavy_ear = ear_value <= self.heavy_ear_thresh

        if eyes_closed and not normal_blink:
            self.eyes_open_start = None
        else:
            if self.eyes_open_start is None:
                self.eyes_open_start = t_now

        if perclos >= self.tired_on_perclos:
            if self.high_perclos_start is None:
                self.high_perclos_start = t_now
        else:
            self.high_perclos_start = None

        high_perclos_duration = 0.0
        if self.high_perclos_start is not None:
            high_perclos_duration = t_now - self.high_perclos_start

        if heavy_ear:
            if self.heavy_ear_start is None:
                self.heavy_ear_start = t_now
        else:
            self.heavy_ear_start = None

        heavy_ear_duration = 0.0
        if self.heavy_ear_start is not None:
            heavy_ear_duration = t_now - self.heavy_ear_start

        # Turn TIRED on
        if not self.active:
            # Best trigger: slow blink / short sleep / waking immediately
            if slow_closure:
                self.active = True
                self.active_start = t_now
                self.last_reason = "slow_eye_closure"

            # Heavy eyelids: EAR is low for long enough
            elif heavy_ear_duration >= self.heavy_ear_seconds:
                self.active = True
                self.active_start = t_now
                self.last_reason = "heavy_eyelids"

            # Critical PERCLOS only counts if the eyes also look bad right now
            elif perclos >= self.critical_perclos and (eyes_closed or heavy_ear):
                self.active = True
                self.active_start = t_now
                self.last_reason = "critical_perclos_with_eye_signal"

            # Do NOT activate TIRED from sustained PERCLOS alone
            else:
                self.last_reason = "not_tired"
        else:
            active_duration = t_now - self.active_start if self.active_start else 0.0
            open_duration = t_now - self.eyes_open_start if self.eyes_open_start else 0.0

            if active_duration >= self.min_active_seconds:
                if (
                    open_duration >= self.clear_after_open_seconds
                    and perclos <= self.critical_perclos
                    and not heavy_ear
                ):
                    self.active = False
                    self.active_start = None
                    self.high_perclos_start = None
                    self.heavy_ear_start = None
                    self.last_reason = "cleared_eyes_open"
                elif perclos <= self.tired_off_perclos and not eyes_closed:
                    self.active = False
                    self.active_start = None
                    self.high_perclos_start = None
                    self.heavy_ear_start = None
                    self.last_reason = "cleared_low_perclos"
                else:
                    if slow_closure:
                        self.last_reason = "slow_eye_closure"
                    elif heavy_ear:
                        self.last_reason = "heavy_eyelids"
                    else:
                        self.last_reason = "active_hold"

        return self.active


TiredFilter = TiredDisplayFilter

from collections import deque


class DriverStateMonitor:
    def __init__(
        self,
        ear_thresh=0.22,
        drowsy_seconds=3.0,
        micro_closure_seconds=2.0,
        micro_closure_count=3,
        micro_window_seconds=60.0,
        max_open_gap_seconds=0.15,
        yawn_mar_thresh=0.55,
        yawn_seconds=1.2,
        yawn_pattern_count=3,
        yawn_window_seconds=300.0,
        alert_cooldown_seconds=5.0,
    ):
        self.ear_thresh = ear_thresh
        self.drowsy_seconds = drowsy_seconds
        self.micro_closure_seconds = micro_closure_seconds
        self.micro_closure_count = micro_closure_count
        self.micro_window_seconds = micro_window_seconds
        self.max_open_gap_seconds = max_open_gap_seconds
        self.yawn_mar_thresh = yawn_mar_thresh
        self.yawn_seconds = yawn_seconds
        self.yawn_pattern_count = yawn_pattern_count
        self.yawn_window_seconds = yawn_window_seconds
        self.alert_cooldown_seconds = alert_cooldown_seconds

        self.eye_closed_start = None
        self.last_seen_closed = None
        self.drowsy_alerted_for_episode = False

        self.closure_events = deque()
        self.last_micro_alert_time = -999999.0

        self.yawn_start = None
        self.yawn_alerted_for_episode = False
        self.yawn_events = deque()
        self.last_yawn_pattern_alert_time = -999999.0

    def reset_active_continuity(self):
        """Reset only in-progress signals after a sustained missing-face gap."""
        self.eye_closed_start = None
        self.last_seen_closed = None
        self.drowsy_alerted_for_episode = False
        self.yawn_start = None
        self.yawn_alerted_for_episode = False

    def update(self, t_now, ear, mar=None):
        alerts = []

        eyes_closed_raw = ear is not None and ear <= self.ear_thresh

        if eyes_closed_raw:
            if self.eye_closed_start is None:
                self.eye_closed_start = t_now
                self.drowsy_alerted_for_episode = False

            self.last_seen_closed = t_now
            eyes_closed_effective = True

        else:
            if (
                self.eye_closed_start is not None
                and self.last_seen_closed is not None
                and (t_now - self.last_seen_closed) <= self.max_open_gap_seconds
            ):
                eyes_closed_effective = True
            else:
                eyes_closed_effective = False

        closed_duration = 0.0

        if eyes_closed_effective and self.eye_closed_start is not None:
            closed_duration = t_now - self.eye_closed_start

            if (
                closed_duration >= self.drowsy_seconds
                and not self.drowsy_alerted_for_episode
            ):
                trigger_time = self.eye_closed_start + self.drowsy_seconds
                detection_delay = t_now - trigger_time

                alerts.append(
                    {
                        "event": "driver_drowsiness",
                        "time_sec": round(t_now, 3),
                        "closed_duration": round(closed_duration, 3),
                        "detection_delay": round(detection_delay, 3),
                    }
                )

                self.drowsy_alerted_for_episode = True

        if not eyes_closed_effective and self.eye_closed_start is not None:
            episode_end = self.last_seen_closed if self.last_seen_closed else t_now
            episode_duration = episode_end - self.eye_closed_start

            if episode_duration >= self.micro_closure_seconds:
                self.closure_events.append(episode_end)

            self.eye_closed_start = None
            self.last_seen_closed = None
            self.drowsy_alerted_for_episode = False

        while self.closure_events and (t_now - self.closure_events[0]) > self.micro_window_seconds:
            self.closure_events.popleft()

        if (
            len(self.closure_events) >= self.micro_closure_count
            and (t_now - self.last_micro_alert_time) >= self.alert_cooldown_seconds
        ):
            alerts.append(
                {
                    "event": "micro_sleep",
                    "time_sec": round(t_now, 3),
                    "closures_in_window": len(self.closure_events),
                }
            )

            self.last_micro_alert_time = t_now
            self.closure_events.clear()

        single_yawn_active = False

        if mar is not None and mar >= self.yawn_mar_thresh:
            if self.yawn_start is None:
                self.yawn_start = t_now
                self.yawn_alerted_for_episode = False

            yawn_duration = t_now - self.yawn_start

            if yawn_duration >= self.yawn_seconds:
                single_yawn_active = True

                if not self.yawn_alerted_for_episode:
                    self.yawn_events.append(t_now)
                    alerts.append(
                        {
                            "event": "yawning",
                            "time_sec": round(t_now, 3),
                            "duration_sec": round(yawn_duration, 3),
                            "reason": "single_yawn",
                            "mode": "mar",
                            "detection_delay": 0.0,
                            "mar": round(mar, 3),
                        }
                    )

                    self.yawn_alerted_for_episode = True

        else:
            self.yawn_start = None
            self.yawn_alerted_for_episode = False

        while (
            self.yawn_events
            and (t_now - self.yawn_events[0]) > self.yawn_window_seconds
        ):
            self.yawn_events.popleft()

        current_yawn_count = len(self.yawn_events)
        yawning_pattern_active = current_yawn_count >= self.yawn_pattern_count

        if (
            yawning_pattern_active
            and (t_now - self.last_yawn_pattern_alert_time) >= self.alert_cooldown_seconds
        ):
            pattern_start = self.yawn_events[0]
            alerts.append(
                {
                    "event": "yawning_pattern",
                    "time_sec": round(t_now, 3),
                    "start_sec": round(pattern_start, 3),
                    "duration_sec": round(t_now - pattern_start, 3),
                    "yawn_count": current_yawn_count,
                    "reason": "three_yawns",
                    "mode": "mar",
                    "detection_delay": 0.0,
                }
            )
            self.last_yawn_pattern_alert_time = t_now
            self.yawn_events.clear()

        states = {
            "eyes_closed": eyes_closed_effective,
            "closed_duration": round(closed_duration, 3),
            "drowsiness_active": closed_duration >= self.drowsy_seconds,
            "micro_closure_count": len(self.closure_events),
            "single_yawn_active": single_yawn_active,
            "yawning": single_yawn_active,
            "yawn_alert": yawning_pattern_active,
            "yawn_count": current_yawn_count,
            "yawn_count_in_window": current_yawn_count,
        }

        return states, alerts

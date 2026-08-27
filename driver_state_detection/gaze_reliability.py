import math
from collections import deque

import numpy as np


LEFT_EYE_LANDMARKS = [
    33, 133, 159, 145, 153, 154, 155, 173, 157, 158, 160, 161, 246
]

RIGHT_EYE_LANDMARKS = [
    362, 263, 386, 374, 380, 381, 382, 398, 384, 385, 387, 388, 466
]


class GazeReliabilityGate:
    """
    Adaptive gaze gate.

    Goal:
    - Normal eyes / clear glasses / transparent glasses -> use gaze.
    - Dark sunglasses / heavy reflection / bad iris tracking -> use head pose only.
    """

    def __init__(
        self,
        ear_min_for_gaze=0.18,
        max_gaze_score=1.20,
        max_gaze_jump=0.45,
        jitter_window=8,
        max_jitter=0.35,
        bad_frames_to_disable=2,
        good_frames_to_enable=5,

        # Relaxed visibility thresholds
        max_dark_ratio=0.65,
        max_glare_ratio=0.35,
        min_eye_contrast=4.0,
        min_eye_to_face_brightness_ratio=0.42,
    ):
        self.ear_min_for_gaze = ear_min_for_gaze
        self.max_gaze_score = max_gaze_score
        self.max_gaze_jump = max_gaze_jump
        self.jitter_window = jitter_window
        self.max_jitter = max_jitter
        self.bad_frames_to_disable = bad_frames_to_disable
        self.good_frames_to_enable = good_frames_to_enable

        self.max_dark_ratio = max_dark_ratio
        self.max_glare_ratio = max_glare_ratio
        self.min_eye_contrast = min_eye_contrast
        self.min_eye_to_face_brightness_ratio = min_eye_to_face_brightness_ratio

        self.gaze_history = deque(maxlen=jitter_window)
        self.prev_gaze = None

        self.bad_count = 0
        self.good_count = 0
        self.gaze_enabled = True
        self.last_reason = "initial"

    def _to_float(self, value):
        try:
            value = float(np.ravel(value)[0])
            if math.isfinite(value):
                return value
            return None
        except Exception:
            return None

    def _is_number(self, value):
        return self._to_float(value) is not None

    def _landmarks_to_pixels(self, landmarks, frame_width, frame_height):
        pts = np.asarray(landmarks)

        if np.nanmax(pts[:, :2]) <= 2.0:
            pts_px = pts.copy()
            pts_px[:, 0] = pts[:, 0] * frame_width
            pts_px[:, 1] = pts[:, 1] * frame_height
            return pts_px

        return pts

    def _patch_stats(self, patch):
        if patch is None or patch.size == 0:
            return None

        return {
            "brightness": float(np.mean(patch)),
            "contrast": float(np.std(patch)),
            "dark_ratio": float(np.mean(patch < 45)),
            "glare_ratio": float(np.mean(patch > 235)),
        }

    def _crop_eye_patch(self, gray, pts_px, frame_width, frame_height, eye_indices):
        eye_pts = pts_px[eye_indices, :2]

        x_min = int(np.min(eye_pts[:, 0]))
        x_max = int(np.max(eye_pts[:, 0]))
        y_min = int(np.min(eye_pts[:, 1]))
        y_max = int(np.max(eye_pts[:, 1]))

        pad_x = max(8, int((x_max - x_min) * 0.55))
        pad_y = max(8, int((y_max - y_min) * 1.40))

        x_min = max(0, x_min - pad_x)
        x_max = min(frame_width - 1, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(frame_height - 1, y_max + pad_y)

        if x_max <= x_min or y_max <= y_min:
            return None

        return gray[y_min:y_max, x_min:x_max]

    def _face_reference_brightness(self, gray, pts_px, frame_width, frame_height):
        """
        Uses a central face patch as brightness reference.
        This avoids falsely marking normal dark eyes as sunglasses.
        """
        x_min = int(np.min(pts_px[:, 0]))
        x_max = int(np.max(pts_px[:, 0]))
        y_min = int(np.min(pts_px[:, 1]))
        y_max = int(np.max(pts_px[:, 1]))

        face_w = x_max - x_min
        face_h = y_max - y_min

        if face_w <= 0 or face_h <= 0:
            return None

        # Central cheek/nose/mouth skin-ish region, below eyes
        rx1 = int(x_min + 0.25 * face_w)
        rx2 = int(x_min + 0.75 * face_w)
        ry1 = int(y_min + 0.38 * face_h)
        ry2 = int(y_min + 0.78 * face_h)

        rx1 = max(0, rx1)
        rx2 = min(frame_width - 1, rx2)
        ry1 = max(0, ry1)
        ry2 = min(frame_height - 1, ry2)

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        patch = gray[ry1:ry2, rx1:rx2]

        if patch.size == 0:
            return None

        return float(np.mean(patch))

    def _eyes_visible(self, gray_frame, landmarks, frame_size):
        if gray_frame is None or landmarks is None or frame_size is None:
            return True, "no_visibility_check"

        frame_width, frame_height = frame_size

        if len(gray_frame.shape) == 3:
            gray = gray_frame[:, :, 0]
        else:
            gray = gray_frame

        pts_px = self._landmarks_to_pixels(landmarks, frame_width, frame_height)

        left_patch = self._crop_eye_patch(
            gray, pts_px, frame_width, frame_height, LEFT_EYE_LANDMARKS
        )
        right_patch = self._crop_eye_patch(
            gray, pts_px, frame_width, frame_height, RIGHT_EYE_LANDMARKS
        )

        left_stats = self._patch_stats(left_patch)
        right_stats = self._patch_stats(right_patch)

        if left_stats is None or right_stats is None:
            return False, "eye_patch_missing"

        eye_brightness = (left_stats["brightness"] + right_stats["brightness"]) / 2.0
        eye_contrast = (left_stats["contrast"] + right_stats["contrast"]) / 2.0
        dark_ratio = (left_stats["dark_ratio"] + right_stats["dark_ratio"]) / 2.0
        glare_ratio = (left_stats["glare_ratio"] + right_stats["glare_ratio"]) / 2.0

        face_brightness = self._face_reference_brightness(
            gray, pts_px, frame_width, frame_height
        )

        if face_brightness is None or face_brightness <= 1:
            brightness_ratio = 1.0
        else:
            brightness_ratio = eye_brightness / face_brightness

        # Dark sunglasses: eye region is much darker than face AND mostly dark.
        if (
            dark_ratio > self.max_dark_ratio
            and brightness_ratio < self.min_eye_to_face_brightness_ratio
        ):
            return False, "sunglasses_dark"

        # Strong glass reflection / glare
        if glare_ratio > self.max_glare_ratio:
            return False, "eye_glare_high"

        # Too little texture/detail, but only if the eye is also dark relative to face.
        if (
            eye_contrast < self.min_eye_contrast
            and brightness_ratio < 0.55
        ):
            return False, "eye_detail_low"

        return True, "eyes_visible"

    def _check_gaze_signal(self, gaze_score):
        if gaze_score is None or not self._is_number(gaze_score):
            return False, "no_gaze_score"

        gaze_value = self._to_float(gaze_score)

        if gaze_value < 0 or gaze_value > self.max_gaze_score:
            return False, "gaze_out_of_range"

        if self.prev_gaze is not None:
            if abs(gaze_value - self.prev_gaze) > self.max_gaze_jump:
                self.prev_gaze = gaze_value
                return False, "gaze_jump"

        self.gaze_history.append(gaze_value)

        if len(self.gaze_history) >= self.jitter_window:
            jitter = float(np.std(self.gaze_history))

            if jitter > self.max_jitter:
                self.prev_gaze = gaze_value
                return False, "gaze_jitter"

            self.prev_gaze = gaze_value
            return True, "gaze_stable"

        self.prev_gaze = gaze_value
        return True, "warming_up"

    def update(self, ear, gaze_score, gray_frame=None, landmarks=None, frame_size=None):
        if ear is None or not self._is_number(ear):
            is_good = False
            reason = "no_ear"

        elif self._to_float(ear) < self.ear_min_for_gaze:
            is_good = False
            reason = "eyes_too_closed"

        else:
            eyes_visible, visibility_reason = self._eyes_visible(
                gray_frame=gray_frame,
                landmarks=landmarks,
                frame_size=frame_size,
            )

            if not eyes_visible:
                is_good = False
                reason = visibility_reason
            else:
                is_good, reason = self._check_gaze_signal(gaze_score)

                # Preserve useful visibility information during debugging
                if is_good and visibility_reason == "eyes_visible":
                    reason = reason

        if is_good:
            self.good_count += 1
            self.bad_count = 0
        else:
            self.bad_count += 1
            self.good_count = 0

        if self.bad_count >= self.bad_frames_to_disable:
            self.gaze_enabled = False

        if self.good_count >= self.good_frames_to_enable:
            self.gaze_enabled = True

        self.last_reason = reason

        if self.gaze_enabled:
            return True, reason, "gaze_plus_head_pose"

        return False, reason, "head_pose_only"
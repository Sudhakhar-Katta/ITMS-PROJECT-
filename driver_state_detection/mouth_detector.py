import numpy as np
from numpy import linalg as LA


def get_MAR(landmarks):
    """
    Mouth Aspect Ratio:
    MAR = mouth opening / mouth width

    Higher MAR means mouth is open wider.
    """

    try:
        left_mouth = landmarks[61, :2]
        right_mouth = landmarks[291, :2]
        upper_lip = landmarks[13, :2]
        lower_lip = landmarks[14, :2]

        mouth_width = LA.norm(left_mouth - right_mouth)
        mouth_opening = LA.norm(upper_lip - lower_lip)

        if mouth_width == 0:
            return None

        mar = mouth_opening / mouth_width
        return mar

    except Exception:
        return None
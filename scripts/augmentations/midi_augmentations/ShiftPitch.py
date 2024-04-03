from scripts.augmentations.base import AugmentationBase
from symusic.core import ScoreTick
from random import randint, random

class ShiftPitch(AugmentationBase):
    def __init__(self, proba: float=0.1):
        self.proba = proba

    def __call__(self, score: ScoreTick):
        if random() < self.proba:
            shift_value = randint(-12, 12)
            score = score.shift_pitch(shift_value)
        return score
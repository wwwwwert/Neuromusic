from scripts.augmentations.base import AugmentationBase
from symusic.core import ScoreTick
from random import randint, random

class ShiftVelocity(AugmentationBase):
    def __init__(self, proba: float=0.1):
        self.proba = proba

    def __call__(self, score: ScoreTick):
        if random() < self.proba:
            shift_value = randint(-4, 4)
            score = score.shift_velocity(shift_value)
        return score
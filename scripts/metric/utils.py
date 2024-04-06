from editdistance import eval
from sklearn.metrics import accuracy_score
from torch import Tensor


def calc_ter(target_sequence: Tensor, predicted_sequence: Tensor) -> float:
    """Tokens error rate.
    Motivation. Model can skip token that does not affect the sample overall. However in this case accuracy will be small.
    It is better to calculate edit distance between sequences.
    """
    if target_sequence.shape[0] == 0:
        if predicted_sequence.shape[0] == 0:
            return 1.
        return 0.
    return eval(target_sequence, predicted_sequence) / len(target_sequence)


def calc_accuracy_score(target_sequence: Tensor, predicted_sequence: Tensor) -> float:
    """Its better to use accuracy only on train. Use calc_ter to check inference."""
    return accuracy_score(target_sequence, predicted_sequence)

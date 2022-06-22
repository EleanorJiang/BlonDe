import math
import numpy as np
from typing import Sequence, Tuple, Dict, Any, Union, Optional, Sequence
from collections import Counter, defaultdict
# Declaration
ProcessedSent = Dict[str, Any]
ProcessedDoc = Sequence[ProcessedSent]
ProcessedCorpus = Sequence[ProcessedDoc]
Counts = Dict[str, Union[Sequence[Counter], Counter]]
Weight = Dict[str, Union[Tuple[float], float]]


# math
def normalize(lst):
    Z = sum(lst)
    return [item/Z for item in lst]

def safe_devide(a, b):
    try:
        value = a/b
    except ZeroDivisionError:
        value = float('nan')
    return value


def compute_F1(r: float, p: float) -> float:
    if not math.isnan(r) and not math.isnan(p):
        return safe_devide(2*r*p, r+p)
    elif r != float('nan'):
        return r
    else:
        return p


def my_log(num: float, epsilon: float=0.00001) -> float:
    """
    Floors the log function

    :param num: the number
    :return: log(num) floored to a very low number
    """
    num = max(num, epsilon)
    return math.log(num)

# Counter Union
def union(counter1, counter2):
    all_ngram = list(set(counter1.keys()).union(set(counter2.keys())))
    return all_ngram

# intersect of lists
def intersect(lst1, lst2):
    return list(set(lst1).intersection(set(lst2)))

# diff of lists
def diff(lst1, lst2):
    return list(set(lst1) - set(lst2))


# averaging
def geo_average(scores, weights=None):
    scores = np.array(scores)
    if weights is None:
        return np.exp(np.nansum([my_log(score) for score in scores]) / sum(~np.isnan(scores)))
    else:
        return np.exp(np.nansum([my_log(score) * weight for score, weight in zip(scores, weights)]) / sum(~np.isnan(scores)))


def arm_average(scores, weights=None):
    scores = np.array(scores)
    if weights is None:
        return np.nanmean(scores)
    else:
        return np.nanmean([my_log(score) * weight for score, weight in zip(scores, weights)])


# I/O
def args_to_dict(args, prefix: str, strip_prefix: bool = False):
    """Filters argparse's `Namespace` into dictionary with arguments
    beginning with the given prefix."""
    prefix += '_'
    d = {}
    for k, v in args.__dict__.items():
        if k.startswith(prefix):
            k = k.replace(prefix, '') if strip_prefix else k
            d[k] = v
    return d
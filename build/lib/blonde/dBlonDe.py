"""dBlonDe score implementation."""
from __future__ import division

import logging
import math
from collections import Counter, defaultdict
from typing import Sequence, Tuple, Dict, Any
from operator import itemgetter
from .utils import Weight, Counts, compute_F1, safe_devide, union, intersect, diff, geo_average, arm_average
from operator import itemgetter
import numpy as np

def sim(sent_s_count: Counts, max_sent_r_count: Counts, weights: Weight, categories: dict):
    """
    Calculate the similarity between Un_s and Un_r.
    :param sent_s_count: the counts in hypothesis, corresponds to processed_sent['count'].
                    See the description in ``process_corpus``
    :param max_sent_r_count: the maximum count of references. Same as sent_s_count.
    :param weights: Weight. See the description in BlonDe.
    :return similarity: the similarity score, aka the total number of matched checkpofloats. float
    :return numerator: a dict[float], the numbers of matched checkpofloats per category.
    :return denominator_r: a dict[float], the numbers of reference checkpofloats per category.
    :return denominator_s: a dict[float], the numbers of system checkpofloats per category.
    """
    similarity = 0
    numerator, denominator_r, denominator_s = defaultdict(float), defaultdict(float), defaultdict(float)
    for category, weight_list in weights.items():
        sys_counts = sent_s_count[category]
        ref_counts = max_sent_r_count[category]
        if category == "n-gram" or category == "plus":
            for i, n in enumerate(categories[category]):
                # a counter of all i-grams
                # for each ngrams, we find the max counter
                all_ngram = union(ref_counts[i], sys_counts[i])
                for ngram in all_ngram:
                    # ref_count, sys_count = ref_counts[i][ngram], sys_counts[i][ngram]
                    match_count = min(ref_counts[i][ngram], sys_counts[i][ngram])
                    numerator[n] += weight_list[i] * match_count
                    denominator_r[n] += weight_list[i] * ref_counts[i][ngram]
                    denominator_s[n] += weight_list[i] * sys_counts[i][ngram]
                    similarity += numerator[n]
        elif category == "entity":
            for i, _ in enumerate(categories[category]):
                # a counter of all i-grams
                # for each ngrams, we find the max counter
                all_ngram = union(ref_counts[i], sys_counts[i])
                for ngram in all_ngram:
                    match_count = min(ref_counts[i][ngram], sys_counts[i][ngram])
                    numerator[category] += weight_list[i] * match_count
                    denominator_r[category] += weight_list[i] * ref_counts[i][ngram]
                    denominator_s[category] += weight_list[i] * sys_counts[i][ngram]
                    similarity += numerator[category]
        else:
            for i, type in enumerate(categories[category]):
                match_count = min(ref_counts[type], sys_counts[type])
                numerator[category] += weight_list[i] * match_count
                denominator_r[category] += weight_list[i] * ref_counts[type]
                denominator_s[category] += weight_list[i] * sys_counts[type]
                similarity += weight_list[i] * match_count
    return similarity, numerator, denominator_r, denominator_s


def scoring(doc_s_count: Sequence[Counts], max_doc_r_count: Sequence[Counts], weights: Weight, categories: dict,
            max_order=None, effective_order: bool=False, average_method: str = 'geometric',
            smooth_method: str = 'none', smooth_value=None
            ) -> Tuple[float, float, float, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Calculate the match score between the system document and the reference documents.
    :param doc_s_count: the counts in hypothesis document, a list of processed_sent['count'].
                    See the description in ``process_corpus``
    :param max_doc_r_count: the maximum count of references document. Same as doc_s_count.
    :param weights: see the description in BlonDe.
    :return recall/precision/F1: float.
    :return recalls/precisions/F1s: a dict, recall/precision/F1 per category.
    """
    total_similarity, total_r, total_s = 0, 0, 0
    total_numerator, total_denominator_r, total_denominator_s = \
        defaultdict(float), defaultdict(float), defaultdict(float)
    recalls, precisions, F1s = {}, {}, {}
    for sent_s_count, max_sent_r_count in zip(doc_s_count, max_doc_r_count):
        tmp_similarity, tmp_numerator, tmp_denominator_r, tmp_denominator_s = sim(
            sent_s_count, max_sent_r_count, weights, categories)
        total_similarity += tmp_similarity
        for key in tmp_numerator.keys():
            total_numerator[key] += tmp_numerator[key]
            total_denominator_r[key] += tmp_denominator_r[key]
            total_denominator_s[key] += tmp_denominator_s[key]

    # smoothing
    do_smoothing = (smooth_method is not None and 'n-gram' in weights.keys() and max_order is not None)
    def get_ngram(a, orders):
        return itemgetter(*orders)(a)
    if do_smoothing:
        orders = list(range(1, max_order+1))
        ngram_recalls, eff_order_r = smoothing(get_ngram(total_numerator, orders),
                                               get_ngram(total_denominator_r, orders),
                                               orders, smooth_method, smooth_value)
        ngram_precisions, eff_order_p = smoothing(get_ngram(total_numerator, orders),
                                                  get_ngram(total_denominator_s, orders),
                                                  orders, smooth_method, smooth_value)
        # update eff_order based on smooth_method
        eff_order_F1 = min(eff_order_r, eff_order_p)

    # now compute every subcategory:
    for key in total_numerator.keys():
        if do_smoothing and key in orders:
            if effective_order:
                if key in range(1, eff_order_r+1):
                    recalls[key] = ngram_recalls[key-1]
                if key in range(1, eff_order_p+1):
                    precisions[key] = ngram_recalls[key-1]
                if key in range(1, eff_order_F1+1):
                    F1s[key] = compute_F1(recalls[key], precisions[key])
            else:
                recalls[key] = ngram_recalls[key-1]
                precisions[key] = ngram_precisions[key-1]
                F1s[key] = compute_F1(recalls[key], precisions[key])
        else:  # no smoothing
            recalls[key] = safe_devide(total_numerator[key], total_denominator_r[key])
            precisions[key] = safe_devide(total_numerator[key], total_denominator_s[key])
            F1s[key] = compute_F1(recalls[key], precisions[key])

    # average over all the subcategory
    def get_sum_results(recall_values, precision_values):
        if average_method == 'geometric':
            recall = geo_average(recall_values)
            precision = geo_average(precision_values)
        elif average_method == 'arithmetic':
            recall = arm_average(recall_values)
            precision = arm_average(precision_values)
        F1 = compute_F1(recall, precision)
        return recall, precision, F1

    R, P, F1 = get_sum_results(list(recalls.values()), list(precisions.values()))

    # For dBlonDe
    assert len(diff(recalls.keys(), precisions.keys())) == 0, "Oh no! recalls and precisions are having different keys!"
    keys = intersect(recalls.keys(), ["tense", "pronoun", "entity", "dm"])
    recalls['dBlonDe'], precisions['dBlonDe'], F1s['dBlonDe'] = get_sum_results(
        itemgetter(*keys)(recalls), itemgetter(*keys)(precisions))
    # For BlonDe when BlonDe_plus is computed
    keys = diff(recalls.keys(), ["ambiguity", "ellipsis"])
    recalls['sBlonDe'], precisions['sBlonDe'], F1s['sBlonDe'] = get_sum_results(
        itemgetter(*keys)(recalls), itemgetter(*keys)(precisions))

    return R, P, F1, recalls, precisions, F1s


def smoothing(correct, total, orders, smooth_method, smooth_value):
    """
    Smoothing methods (citing "A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU",
        Boxing Chen and Colin Cherry, WMT 2014: http://aclweb.org/anthology/W14-3346)

        - floor: Method 1 (requires small positive value (0.1 in the paper) to be set)
        - add-k: Method 2 (Generalizing Lin and Och, 2004)
        - exp: Method 3 (NIST smoothing method i.e. in use with mteval-v13a.pl)
    """
    numerator = [0.0 for _ in orders]
    smooth_mteval = 1.
    for n in range(1, len(numerator) + 1):
        if smooth_method == 'add-k' and n > 1:
            correct[n - 1] += smooth_value
            total[n - 1] += smooth_value

        if total[n - 1] == 0:
            break

        eff_order = n

        if correct[n - 1] == 0:
            if smooth_method == 'exp':
                smooth_mteval *= 2
                numerator[n - 1] = safe_devide(1., (smooth_mteval * total[n - 1]))
            elif smooth_method == 'floor':
                numerator[n - 1] = safe_devide(smooth_value,  total[n - 1])
        else:
            numerator[n - 1] = safe_devide(correct[n - 1], total[n - 1])
    return numerator, eff_order


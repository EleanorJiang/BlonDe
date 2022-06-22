"""
The implementation of the BlonDe metric.
Adapted from Sacreblonde.
"""

import math
import logging
from importlib import import_module
from . import WEIGHTS, CATEGORIES
from .processing import process_corpus, add_blonde_plus_categories, refine_NER
from .dBlonDe import scoring
from typing import Sequence, Tuple, Dict, Any, Union, Optional, Sequence
from .utils import normalize, Weight, Counts, ProcessedSent, ProcessedDoc, ProcessedCorpus
from .base import Score, Signature, Metric
from collections import Counter, defaultdict

logger = logging.getLogger('BlonDe')


class BLONDESignature(Signature):
    """A convenience class to represent the reproducibility signature for BlonDe.

    :param args: key-value dictionary passed from the actual metric instance.
    """
    def __init__(self, args: dict):
        """`BLONDESignature` initializer."""
        super().__init__(args)

        self._abbr.update({
            'case': 'c',
            'eff': 'e',
            # 'tok': 'tok',
            # 'smooth': 's',
        })

        self.info.update({
            'case': 'lc' if args['lowercase'] else 'mixed',
            'eff': 'yes' if args['effective_order'] else 'no',
        })


class BLONDEScore(Score):
    """A convenience class to represent BLONDE scores.

    :param score: The BLONDE score.
    :param recall: the overall recall
    :param precision: the overall recall
    :param F1: the overall F1 measure
    :param verbose: a dict, where the items are:
            `recalls`: a dict of recall per category, where each item is (category: str, score: float)
            `precisions`: a dict of precision per category, where each item is (category: str, score: float)
            `F1s`: a dict of F1 measure per category, where each item is (category: str, score: float)
    _verbose: a string that shows "recall", "precision" and "F1" per category.
    """
    def __init__(self, recall: float, precision: float, F1: float,
                 verbose: Optional[Dict[str, Dict[str, float]]]=None
                 ):
        """`BLONDEScore` initializer."""
        super().__init__('BLONDE', F1)

        self.recall = recall
        self.precision = precision
        self.F1 = F1
        self.detail = verbose
        # The verbose part of BLONDE
        # for method in ("recall", "precision", "F1"):
        self.verbose = f"\nR: {recall:.2%}\tP: {precision:.2%}\tF1: {F1:.2%}\n"
        if verbose is not None:
            for method, values in verbose.items():
                self.verbose += f"{method}\n\t"
                for category, score in values.items():
                    self.verbose += f"{category}: {score:.2%}\t"
                self.verbose += "\n"


class BLONDE(Metric):
    """Computes the BLONDE metric given hypotheses and references.

    :param categories: A dict where the keys are chosen from ('tense', 'pronoun', 'entity', 'n-gram')
                        and the keys are the names of features in different categories, Dict[str, Sequence[str]]
                        If `None`, ('tense', 'pronoun', 'entity', 'n-gram')
    :param weights: The weights of the aerformentioned features, Dict[str, Sequence[float]]
                        If `None`, uniform weights
    :param plus_categories: The human annotated categories, e.g. ('ambiguity', 'ellipsis') (default: None)
    :param plus_weights: The weights of the human annotated categories (default: None)
    :param weight_normalize: Whether to reweight to 1 (default: False)
    :param lowercase: If True, lowercased BLONDE is computed.
    :param average_method: The average method to use ('geometric', 'arithmetic').
    :param smooth_method: The smoothing method to use ('floor', 'add-k', 'exp' or 'none').
    :param smooth_value: The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
    :param max_ngram_order: If given, it overrides the maximum n-gram order (default: 4).
    :param effective_order: If `True`, stop including n-gram orders for which score is 0. This should be
    `True`, if sentence-level BLONDE will be computed.
    :param references: A sequence of reference documents with document being
    defined as a sequence of reference strings. If given, the reference n-grams
    and lengths will be pre-computed and cached for faster BLONDE computation
    across many systems.
    """


    WEIGHTS_DEFAULTS = WEIGHTS

    SMOOTH_DEFAULTS: Dict[str, Optional[float]] = {
        # The defaults for `floor` and `add-k` are obtained from the following paper
        # A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU
        # Boxing Chen and Colin Cherry
        # http://aclweb.org/anthology/W14-3346
        'none': None,   # No value is required
        'floor': 0.1,
        'add-k': 1,
        'exp': None,    # No value is required
    }

    _SIGNATURE_TYPE = BLONDESignature



    def __init__(self, weights: Weight=WEIGHTS_DEFAULTS,
                 weight_normalize: bool = False,
                 average_method: str = 'geometric',
                 categories: dict = CATEGORIES,
                 plus_categories=None,  # ("ambiguity", "ellipsis")
                 plus_weights=(1, 1),
                 lowercase: bool = False,
                 smooth_method: str = 'exp',
                 smooth_value: Optional[float] = None,
                 effective_order: bool = False,
                 references: Optional[Sequence[Sequence[Sequence[str]]]] = None,
                 annotation: Sequence[Sequence[str]] = None,
                 ner_refined: Sequence[Sequence[str]] = None):
        """`BLONDE` initializer."""
        super().__init__()
        self.set_weight(weights, weight_normalize)
        self.set_smooth(smooth_method, smooth_value)
        self.lowercase = lowercase
        self.average_method = average_method
        self.categories = categories
        self.plus_categories = plus_categories
        if plus_categories is not None:
            self.add_categories(plus_categories, plus_weights)
        self.effective_order = effective_order
        if weight_normalize:
            self._normalize_weight(self.weight)
        if references is not None:
            # Pre-compute reference ngrams and lengths
            self._ref_cache = self._cache_references(references, annotation, ner_refined)


    def set_weight(self, weights: Weight, weight_normalize: bool = False):
        self.weights = weights
        self.weight_normalize = weight_normalize
        if weight_normalize:
            self._normalize_weight(self.weight)


    def reset_weight(self):
        """
        revert weights to the default choice
        """
        self.weights = self.WEIGHTS_DEFAULTS


    def set_smooth(self, smooth_method, smooth_value):
        assert smooth_method in BLONDE.SMOOTH_DEFAULTS.keys(), \
            "Unknown smooth_method {smooth_method!r}"
        self.smooth_method = smooth_method
        # Fetch the default value for floor and add-k
        if smooth_value is None:
            self.smooth_value = BLONDE.SMOOTH_DEFAULTS[smooth_method]


    def _normalize_weight(self, weights):
        """
        normalize per category, make the sum of all types on a certain category sum to 1
        """
        for cat, weight in weights.items():
            if type(weight) is tuple or type(weight) is list:
                weights[cat] = normalize(weight)
            else:
                weights[cat] = 1


    def add_categories(self, plus_categories, weights):
        self.categories["plus"] = plus_categories
        if self.weight_normalize:
            self._normalize_weight(weights)
        self.weights["plus"] = weights


    def compute_blonde(self, s_count: Sequence[Counts],
                      max_r_count: Sequence[Counts],
                      weights: Weight=None) -> BLONDEScore:
        """Computes BLONDE score from its sufficient statistics with smoothing.
        :return: A `BLONDEScore` instance.
        """

        # Fetch the default value for floor and add-k
        if weights is None:
            weights = BLONDE.WEIGHTS_DEFAULTS

        max_order = None
        if 'n-gram' in weights.keys():
            if type(weights['n-gram']) is tuple or type(weights['n-gram']) is list:
                max_order = len(weights['n-gram'])
            else:
                max_order = weights['n-gram']

        recall, precision, F1, recalls, precisions, F1s = scoring(s_count, max_r_count, weights,
                                                                  categories=self.categories,
                                                                  max_order=max_order,
                                                                  effective_order=self.effective_order,
                                                                  average_method=self.average_method,
                                                                  smooth_method=self.smooth_method,
                                                                  smooth_value=self.smooth_value)
        verbose = {'recalls': recalls, 'precisions': precisions, 'F1s': F1s}
        return BLONDEScore(recall, precision, F1, verbose)

    def _compute_score_from_stats(self, stats: Tuple[Sequence[Counts], Sequence[Counts]]) -> BLONDEScore:
        s_count = stats[0]
        max_r_count = stats[1]
        return self.compute_blonde(s_count, max_r_count, weights=self.weights)

    def _aggregate_and_compute(self, stats: Tuple[Sequence[Sequence[Counts]], Sequence[Sequence[Counts]]]) -> BLONDEScore:
        """Computes the final BLEU score given the pre-computed corpus statistics.

        :param stats: A list of segment-level statistics
        :return: A `BLEUScore` instance.
        """
        s_count = [item for doc_stats in stats[0] for item in doc_stats]
        max_r_count = [item for doc_stats in stats[1] for item in doc_stats]
        return self.compute_blonde(s_count, max_r_count, weights=self.weights)

    def _extract_max_reference_sent(self, list_of_sent_r: Sequence[ProcessedSent]) -> Counts:
        """
        extract the max reference count at sentence level
        :param list_of_sent_r: a list of references, where every reference is a dict `processed_sent`.
        """
        max_sent_r_count = {}  # correspond to sent_r["count"]
        for category, types in self.categories.items():
            max_sent_r_count[category] = {}
            list_of_count_r = [sent_r["count"][category] for sent_r in list_of_sent_r]  # length of k
            if category in ["entity", "n-gram", "plus"]:
                for n, _ in enumerate(types):
                    # a counter of all i-grams
                    # for each ngrams, we find the max counter
                    max_sent_r_count[category][n] = None
                    for ref_counts in list_of_count_r:
                        # iterate over k, each ref_counts is a counter of one reference for one sentence
                        # ref_counts for 1-gram would be ["I":1, "love":3, "you":4]
                        if max_sent_r_count[category][n] is None:
                            # if we don't have any
                            max_sent_r_count[category][n] = ref_counts[n]
                        else:
                            for ngram, count in ref_counts[n].items():
                                max_sent_r_count[category][n][ngram] = max(max_sent_r_count[category][n][ngram], count)
            else:
                for i, type in enumerate(types):
                    max_sent_r_count[category][type] = max([ref_counts[type] for ref_counts in list_of_count_r])
        return max_sent_r_count

    def _extract_max_reference_doc(self, list_of_doc_r: Sequence[ProcessedDoc]) -> Sequence[Counts]:
        """
        extract the max reference count at document level
        :param list_of_doc_r: a list of references, where every reference is a list `processed_doc`.
        :return max_doc_r_count: corresponds to a list of processed_sent["count"].
        """
        max_doc_r_count = []
        for i in range(len(list_of_doc_r[0])):
            list_of_sent_r = [doc_r[i] for doc_r in list_of_doc_r]
            max_doc_r_count.append(self._extract_max_reference_sent(list_of_sent_r))
        return max_doc_r_count

    def _extract_max_reference(self, list_of_corpus_r: Sequence[ProcessedCorpus]) -> Sequence[Sequence[Counts]]:
        """
        extract the max reference count at corpus level
        :param list_of_doc_r: a list of references, where every reference is a list `processed_corpus`.
        :return max_corpus_r_count: corresponds to a list of lists of processed_sent["count"].
        """
        max_corpus_r_count = []
        for i in range(len(list_of_corpus_r[0])):
            list_of_doc_r = [doc_r[i] for doc_r in list_of_corpus_r]
            max_corpus_r_count.append(self._extract_max_reference_doc(list_of_doc_r))
        return max_corpus_r_count

    def _cache_references(self, references: Optional[Sequence[Sequence[Sequence[str]]]],
                          annotation: Sequence[Sequence[str]]=None, ner_refined: Sequence[Sequence[str]]=None) -> Sequence[Sequence[Counter]]:
        """Given the full set of document references, extract the counts
            (or other necessary information) for caching purposes.

        :param references: A list of reference corpora.
        :param annotation: A list of annotation docs, where each annotation doc is a list of annotation lines
                            corresponding to the sentences in that document.
        :param ner_refined: A list of human annotated NER results,
                            where each NER result is a list of NER lines
                            corresponding to the sentences in that document.
        :return: A list of lists of counters, each counter correspond to a sentence.
        """
        # process all the reference corpora one by one
        all_processed_reference = []
        for reference in references:
            processed_reference = process_corpus(reference, self.categories, lowercase=self.lowercase)
            if annotation is not None:
                assert len(annotation) == len(processed_reference)
                for i, (ref_doc, lines_an) in enumerate(zip(processed_reference, annotation)):
                    assert len(ref_doc) == len(lines_an), f"{i}-th doc: On no! len(ref_doc) != len(lines_an)"
                    add_blonde_plus_categories(ref_doc, lines_an)
            if ner_refined is not None:
                assert len(ner_refined) == len(processed_reference)
                for i, (ref_doc, lines_ner) in enumerate(zip(processed_reference, ner_refined)):
                    assert len(ref_doc) == len(lines_ner), f"{i}-th doc: On no! len(ref_doc) != len(lines_ner)"
                    refine_NER(ref_doc, lines_ner)
            all_processed_reference.append(processed_reference)
        # get the max reference corpus, which in the exact same format as processed_hypotheses
        max_corpus_r_count = self._extract_max_reference(all_processed_reference)
        return max_corpus_r_count

    def _extract_corpus_statistics(self, hypotheses: Sequence[Sequence[str]],
                                   references: Optional[Sequence[Sequence[Sequence[str]]]],
                                   annotation: Sequence[Sequence[str]]=None, ner_refined: Sequence[Sequence[str]]=None
                                   ) -> Tuple[Sequence[Sequence[Counter]], Sequence[Sequence[Counter]]]:
        """Reads the corpus and returns sentence-level match statistics for
        faster re-computations esp. during statistical tests.

        :param hypotheses: A hypothesis corpus.
        :param references: A list of reference corpora.
        :param annotation: A list of annotation docs, where each annotation doc is a list of annotation lines
                            corresponding to the sentences in that document.
        :param ner_refined: A list of human annotated NER results,
                            where each NER result is a list of NER lines
                            corresponding to the sentences in that document.
        :return: A tuple where each sublist corresponds to corpus_s_count/max_corpus_r_count.
        """
        # Pre-compute references
        # Don't store the cache as the user is explicitly passing refs
        if references:
            max_corpus_r_count = self._cache_references(references, annotation, ner_refined)
        elif self._ref_cache:
            max_corpus_r_count = self._ref_cache
        else:
            raise RuntimeError('No references provided and the cache is empty.')

        # process the hypothesis corpus
        processed_hypotheses = process_corpus(hypotheses, self.categories, lowercase=self.lowercase)

        # add annotation
        if annotation is not None:
            for i, (hyp_doc, lines_an) in enumerate(zip(processed_hypotheses, annotation)):
                assert len(hyp_doc) == len(lines_an), f"{i}-th doc: On no! len(hyp_doc) != len(lines_an)"
                add_blonde_plus_categories(hyp_doc, lines_an)

        corpus_s_count = [[sent_s['count'] for sent_s in doc_s] for doc_s in processed_hypotheses]

        return corpus_s_count, max_corpus_r_count

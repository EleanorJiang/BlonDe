"""The base `Score`, `Metric` and `Signature` classes to derive from.

`Metric` is an abstract class that enforces the implementation of a set
of abstract methods. This way, a correctly implemented metric will work
seamlessly with the rest of the codebase.

Adapted from Sacrebleu.
"""

import json
import logging
import statistics
from typing import Sequence, Sequence, Any, Optional, Dict, Tuple
from abc import ABCMeta, abstractmethod
from .utils import Counts

sacrelogger = logging.getLogger('sacrebleu')


class Score:
    """A base score class to derive from.

    :param name: The name of the underlying metric.
    :param score: A floating point number for the final metric.
    """
    def __init__(self, name: str, score: float):
        """`Score` initializer."""
        self.name = name
        self.score = score

        # Statistical test related fields
        self.mean = -1.0
        self.ci = -1.0

        # More info can be added right after the score
        self.verbose = ''

    def format(self, width: int = 4, score_only: bool = False,
               signature: str = '', is_json: bool = False) -> str:
        """Returns a pretty representation of the score.
        :param width: Floating point decimal precision width.
        :param score_only: If `True`, and the format is not `json`,
        returns a single score string.
        :param signature: A string representation of the given `Signature`
        instance.
        :param is_json: If `True`, will output the score in JSON string.
        :return: A plain or JSON-formatted string representation.
        """
        d = {
            'name': self.name,
            'score': float(f'{self.score:.{width}f}'),
            'signature': signature,
        }

        sc = f'{self.score:.{width}f}'

        if self.mean > 0:
            confidence_mean = f'{self.mean:.{width}f}'
            confidence_var = f'{self._ci:.{width}f}'
            confidence_str = f'μ = {confidence_mean} ± {confidence_var}'

            sc += f' ({confidence_str})'
            if is_json:
                d['confidence_mean'] = float(confidence_mean)
                d['confidence_var'] = float(confidence_var)
                d['confidence'] = confidence_str

        # Construct full score line
        full_score = f"{self.name}|{signature}" if signature else self.name
        full_score = f"{full_score} = {sc}"
        if self.verbose:
            full_score += f' {self.verbose}'
            d['verbose_score'] = self.verbose

        if score_only:
            return sc

        if is_json:
            for param in signature.split('|'):
                key, value = param.split(':')
                d[key] = value
            return json.dumps(d, indent=1, ensure_ascii=False)

        return full_score

    def estimate_ci(self, scores: Sequence['Score']):
        """Takes a list of scores and stores mean, stdev and 95% confidence
        interval around the mean.

        :param scores: A list of `Score` objects obtained from bootstrap
        resampling for example.
        """
        # Sort the scores
        raw_scores = sorted([x.score for x in scores])
        n = len(raw_scores)

        # Get CI bounds (95%, i.e. 1/40 from left)
        lower_idx = n // 40
        upper_idx = n - lower_idx - 1
        lower, upper = raw_scores[lower_idx], raw_scores[upper_idx]
        self._ci = 0.5 * (upper - lower)
        self.mean = statistics.mean(raw_scores)

    def __repr__(self):
        """Returns a human readable score string."""
        return self.format()


class Signature:
    """A convenience class to represent sacreBLEU reproducibility signatures.

    :param args: key-value dictionary passed from the actual metric instance.
    """
    def __init__(self, args: dict):
        """`Signature` initializer."""
        # Global items that are shared across all metrics
        self._abbr = {
            'nrefs': '#',
            'test': 't',
            'lang': 'l',
            'subset': 'S',
            'origlang': 'o',
            'bs': 'bs',     # Bootstrap resampling trials
            'ar': 'ar',     # Approximate randomization trials
            'seed': 'rs',   # RNG's seed
        }

        if 'num_refs' not in args:
            num_refs = 1
        else:
            num_refs = args['num_refs']
        if num_refs == -1:
            # Detect variable number of refs
            num_refs = 'var'

        # Global items that are shared across all metrics
        # None's will be ignored
        self.info = {
            'nrefs': num_refs,
            'bs': args.get('n_bootstrap', None),
            'ar': None,
            'seed': args.get('seed', None),
            'test': args.get('test_set', None),
            'lang': args.get('langpair', None),
            'origlang': args.get('origlang', None),
            'subset': args.get('subset', None),
        }

    def format(self, short: bool = False) -> str:
        """Returns a string representation of the signature.

        :param short: If True, shortened signature is produced.
        :return: A string representation of the signature.
        """
        pairs = []
        keys = list(self.info.keys())
        # keep version always at end
        for name in keys:
            value = self.info[name]
            if value is not None:
                if isinstance(value, bool):
                    # Replace True/False with yes/no
                    value = 'yes' if value else 'no'
                final_name = self._abbr[name] if short else name
                pairs.append(f'{final_name}:{value}')

        return '|'.join(pairs)

    def update(self, key: str, value: Any):
        """Add a new item or update an existing one.

        :param key: The key to use in the dictionary.
        :param value: The associated value for the `key`.
        """
        self.info[key] = value

    def __str__(self):
        """Returns a human-readable signature string."""
        return self.format()

    def __repr__(self):
        """Returns a human-readable signature string."""
        return self.format()


class Metric(metaclass=ABCMeta):
    """A base class for all metrics that ensures the implementation of some
    methods. Much of the common functionality is moved to this base class
    from other metrics."""

    # Each metric should define its Signature class' name here
    _SIGNATURE_TYPE = Signature

    def __init__(self):
        """`Metric` initializer."""
        # The pre-computed reference cache
        self._ref_cache = None

        # only useful for BLEU tokenized warnings. Set to True so that
        # warnings are not issued for other metrics.
        self._force = True

        # Will be used by the signature when bootstrap resampling
        self.n_bootstrap = None
        self.seed = None


    def _check_corpus_score_args(self, hyps: Sequence[Sequence[str]],
                                 refs: Optional[Sequence[Sequence[Sequence[str]]]]):
        """Performs sanity checks on `corpus_score` method's arguments.

        :param hypses: A sequence of hypothesis strings.
        :param refs: A sequence of reference documents with document being
        defined as a sequence of reference strings. If `None`, cached references
        will be used.
        """

        prefix = self.__class__.__name__
        err_msg = None

        if not isinstance(hyps, Sequence):
            err_msg = "`hyps` should be a corpus."
        elif not isinstance(hyps[0], Sequence):
            err_msg = 'Each element of `hyps` should be a document.'
        elif not isinstance(hyps[0][0], str):
            err_msg = 'Each element of `hyps[0]` should be a sentence.'

        if refs is not None:
            if not isinstance(refs, Sequence):
                err_msg = "`refs` should be a list of corpora."
            elif not isinstance(refs[0], Sequence):
                err_msg = "Each element of `refs` should be a corpus."
            elif not isinstance(refs[0][0], Sequence):
                err_msg = "Each element of `refs[0]` should be a document."
            elif not isinstance(refs[0][0][0], str):
                err_msg = "`Each element of `refs[0][0]` should be a sentence."

        if err_msg:
            raise RuntimeError(f'{prefix}: {err_msg}')


    @abstractmethod
    def _compute_score_from_stats(self, stats: Tuple[Sequence[Counts], Sequence[Counts]]) -> Any:
        """Computes the final score from already aggregated statistics.

        :param stats: A list or numpy array of segment-level statistics.
        :return: A `Score` object.
        """
        pass

    @abstractmethod
    def _aggregate_and_compute(self, stats: Sequence[Tuple[Sequence[Counts], Sequence[Counts]]]) -> Any:
        """Computes the final score given the pre-computed match statistics.

        :param stats: A list of segment-level statistics.
        :return: A `Score` instance.
        """
        pass


    def corpus_score(self, hypotheses: Sequence[Sequence[str]],
                     references: Optional[Sequence[Sequence[Sequence[str]]]] = None,
                     n_bootstrap: int = 1) -> Any:
        """Compute the metric for a corpus against a single (or multiple) reference(s).

        :param hypotheses: A hypothesis corpus.
        :param references: A list of reference corpora.
        defined as a sequence of reference strings. If `None`, cached references
        will be used.
        :param n_bootstrap: If > 1, provides 95% confidence interval around true mean
        using bootstrap resampling with `n_bootstrap` samples.
        :return: A `Score` object.
        """
        self._check_corpus_score_args(hypotheses, references)

        # Collect corpus stats
        corpus_s_count, max_corpus_r_count = self._extract_corpus_statistics(hypotheses, references)
        stats = (corpus_s_count, max_corpus_r_count)
        # Compute the actual system score
        actual_score = self._aggregate_and_compute(stats)

        if n_bootstrap > 1:
            # Compute bootstrap estimate as well
            # Delayed import is to escape from numpy import if bootstrap
            # is not requested.
            from significance import _bootstrap_resample

            self.n_bootstrap = n_bootstrap
            self.seed, bs_scores = _bootstrap_resample(stats, self, n_bootstrap)
            actual_score.estimate_ci(bs_scores)

        return actual_score

    def get_signature(self) -> Signature:
        """Creates and returns the signature for the metric. The creation
        of signatures is delayed as the number of references is resolved
        only at the point of reference caching."""
        return self._SIGNATURE_TYPE(self.__dict__)

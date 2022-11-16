"""
This script provide a simple mapping from a list of sentences to a document for coreference clusters and entity lists.
"""
import copy
from typing import DefaultDict, List, Optional, Iterator, Set, Tuple, Dict, Sequence
from collections import defaultdict


def get_attr(obj):
    """
    Iterate over object attributes in python and filter out the methods,
    """
    return [a for a in dir(obj) if not a.startswith('__') and not callable(getattr(obj, a))]


def merge_dicts(list_of_dicts):
    return {k: v for d in list_of_dicts for k, v in d.items()}


def align_sent2doc(list_of_tokens):
    """
    For each j-th token in the i-th sentence in `sentences`, i.e. sentences[i][j], return the position in the document.
    the max return value should be the number of total tokens in this document - 1 .
    :param sentences: List[List[str]]. A list of lists, where each element list is a list of tokens that corresponds to
                                       a sentence
    :return A position matrix. List[List[int]]. pos[i][j] is the position in the total document.
    """
    k = 0
    pos_matrix = []
    for tokens in list_of_tokens:
        pos_array = [k + i for i in range(len(tokens)+1)]
        pos_matrix.append(pos_array)
        k += len(tokens)
    return pos_matrix


def map_span(span: Sequence[int], pos_matrix: List[List[str]], sent_index: int) -> Sequence[int]:
    """
    :param span: List[int]. e.g. [0, 2].
    :return span: List[int]. e.g. [12, 14].
    """
    pos_array = pos_matrix[sent_index]
    start = pos_array[span[0]]
    end = pos_array[span[1]]
    assert end - start == span[1] - span[0]
    return [start, end]


def merge_clusters(list_of_clusters, pos_matrix):
    """
    Merge the list of cluster dicts that correspond to sentences into a single dict of clusters.
    The same function could also be applied to list_of_pronouns, which is in the same shape of List[Dict[int, List[Span]]].
    :param list_of_clusters: List[Dict[int, List[Span]]] a list of cluster dicts that correspond to sentences
    :param pos_matrix: List[List[int]]. the output of `align_sent2doc`.
    :return single_clusters: A single cluster Dict[int, List[Span]].
            new_list_of_clusters: A list_of_clusters of the same shape but the positions in spans are global.
    """
    assert len(list_of_clusters) == len(list_of_clusters), \
        "`list_of_clusters` and `pos_matrix` should have the same length (the number of sentences)."
    new_list_of_clusters = copy.deepcopy(list_of_clusters)
    single_clusters = defaultdict(list)
    for key in list_of_clusters[0].keys():
        single_clusters[key] = []
    for i, (clusters, pos_array) in enumerate(zip(list_of_clusters, pos_matrix)):
        for entity_id, spans in clusters.items():
            for j, span in enumerate(spans):
                new_list_of_clusters[i][entity_id][j] = map_span(span, pos_matrix, i)
                single_clusters[entity_id].append(new_list_of_clusters[i][entity_id][j])
    return dict(single_clusters), new_list_of_clusters


def merge_quotes(list_of_quotes, pos_matrix):
    single_quotes = []
    for i, (quotes, pos_array) in enumerate(zip(list_of_quotes, pos_matrix)):
        for quote in quotes:
            new_quote = (quote[0], map_span(quote[-1], pos_matrix, i))
            single_quotes.append(new_quote)
    return single_quotes



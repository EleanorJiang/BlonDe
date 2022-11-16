from BWB import *
import codecs
from collections import Counter, defaultdict
import json
import pandas as pd
from BWB import BWB

KEYS = ["tokens", "sents", "pronouns", "omitted_pronouns", "distinct_terms", "total_terms",
        "distinct_PER", "total_PER", "distinct_FAC", "total_FAC",  "distinct_GPE", "total_GPE",
        "distinct_LOC", "total_LOC", "distinct_VEH", "total_VEH",  "distinct_ORG", "total_ORG",
        "distinct_entities", "total_entities", "coref_chains", "mentions", "quotes", "speakers"]


def _local_counter(sentence, speakers, entity_types):
    """
    :param sentence: a BWBSentence object
    """
    entity_counts = Counter()
    counts = Counter()
    counts["tokens"] = len(sentence.words)
    counts["tokens"] = len(sentence.words)
    counts["sents"] = 1
    counts["pronouns"] = len(sentence.pronouns['P'])
    counts["omitted_pronouns"] = len(sentence.pronouns['O'])
    for entity_id, types in sentence.entities.items():
        num_mentions = len(sentence.clusters[entity_id])
        entity_counts[f'{sentence.document_id}-{entity_id}'] += 1
        if entity_id not in entity_types.keys():
            entity_types[entity_id] = types
        # if types != entity_types[entity_id]:
        #     raise RuntimeError(f'entity_id: {entity_id} has multiple type annotations: {types} and {entity_types[entity_id]}')
        if types[1] == 'T':
            counts["distinct_terms"] += 1
            counts["total_terms"] += num_mentions
        counts[f"distinct_{types[0]}"] += 1
        counts[f"total_{types[0]}"] += num_mentions
        counts["distinct_entities"] += 1
        counts["total_entities"] += num_mentions
    counts["coref_chains"] = len(list(sentence.clusters.keys()))
    counts["mentions"] = sum(len(cluster) for cluster in sentence.clusters.values())
    if len(sentence.quotes) >= 1:
        counts["quotes"] = len(sentence.quotes)
        counts["speaker"] = 1
        for speaker in sentence.quotes:
            speakers.add(speaker[0])
    return counts, entity_counts


def _get_all_counts(stat_list):
    all_counts = Counter()
    for key in KEYS:
        all_counts[key] = sum([stat_counts[key] for stat_counts in stat_list])
    return all_counts


def get_stat_test(cache_file, dir_path, chs_file, ref_file, chs_entity_json, ref_entity_json, langs=("en", "zh")):
    """
    Get the statistics of the BWB test set.
    :param chs_file: the path of a csv file where you want to save the stats of Chinese (source)
    :param ref_file: the path of a csv file where you want to save the stats of English (reference)
    :return: zh_counts, en_counts, zh_stats, en_stats.
    (counts, stats) is in the format of (Counter, Dict[Counter]).
        counts: A counter where the keys are:
            "tokens": the number of tokens,
            "sents": the number of sentences,
            "pronouns": the number of pronouns,
            "omitted_pronouns": the number of omitted pronouns,
            "terms": the number of total terms,
            "distinct_terms": the number of distinct terminologies,
            "total_entities": the number of total entities,
            "distinct_entities": the number of distinct PER entities,
            "distinct_PERs": the number of distinct PER entities,
            "total_PERs": the number of total PER entities,
            ... (same for other 5 entity types: FAC, GPE, LOC, VEH, ORG),
            "coref_chains": the number of coreference chains,
            "mentions": the number of mentions,
            "quotes": the number of sentences that are quoted,
            "speakers": the number of speakers.
        stats: A dict of counters where the keys are `chapter_id`s, e.g. "Book153-3",
               and the values are a list of stat counters, of which each item corresponds to a sentence.
    """
    bwb_reader = BWB()
    zh_stats, en_stats = [], []
    zh_entity_types, en_entity_types = {}, {}
    zh_entity_counts, en_entity_counts = [], []
    zh_speakers, en_speakers = set(), set()
    zh_meta_info, en_meta_info = [], []
    for sentences in bwb_reader.dataset_iterator_from_cache(cache_file, dir_path):
        for sentence in sentences:
            if sentence.lang == "zh":
                zh_counts, zh_entity_count = _local_counter(sentence, zh_speakers, zh_entity_types)
                zh_meta_info.append(f"{sentence.document_id}\t{sentence.sentence_id}\t")
                zh_stats.append(zh_counts)
                zh_entity_counts.append(zh_entity_count)
            elif sentence.lang == "en":
                en_counts, en_entity_count = _local_counter(sentence, en_speakers, en_entity_types)
                en_meta_info.append(f"{sentence.document_id}\t{sentence.sentence_id}\t")
                en_stats.append(en_counts)
                en_entity_counts.append(en_entity_count)
            else:
                raise Exception("Impossible")
    zh_all_counts = _get_all_counts(zh_stats)
    zh_all_counts["speakers"] = len(zh_speakers)
    en_all_counts = _get_all_counts(en_stats)
    en_all_counts["speakers"] = len(en_speakers)
    # to save
    for meta_info, stats, all_counts, path in zip((zh_meta_info, en_meta_info), (zh_stats, en_stats), (zh_all_counts, en_all_counts), (chs_file, ref_file)):
        with codecs.open(path, "w", encoding="utf8") as f:
            line = "doc_id\tsent_id\t"
            line += "\t".join(KEYS)
            f.write(f"{line}\t\n")
            # Total
            line = "Total\t\t"
            for key in KEYS:
                line += f"{all_counts[key]}\t"
            f.write(f"{line}\n")
            for sent_count, meta in zip(stats, meta_info):
                line = meta
                for key in KEYS:
                    line += f"{sent_count[key]}\t"
                f.write(f"{line}\n")
    with open(ref_entity_json, 'w') as f:
        json.dump(en_entity_types, f)
    with open(chs_entity_json, 'w') as f:
        json.dump(zh_entity_types, f)
    return

from typing import DefaultDict, List, Optional, Iterator, Set, Tuple, Dict, Sequence
from collections import defaultdict
import copy
import re
import codecs
import os
import logging, warnings

import pandas as pd
import spacy, pandas, json
import en_core_web_sm, zh_core_web_sm
import sys
import pickle
from collections import Counter, defaultdict
sys.path.insert(0, '.')
from util.align_bpe import align_bpe
from util.sents2doc_mapping import align_sent2doc, map_span, merge_clusters, merge_quotes, get_attr
# Counts = Dict[str, Union[Sequence[Counter], Counter]]
# Weight = Dict[str, Union[Tuple[float], float]]

PUNCTS = "!\"#$%&'()*+,-./:;<=>?@[\]^`{|}~…"
Span = Tuple[int, int]
TypedSpan = Tuple[int, Span]

logger = logging.getLogger(__name__)


class Mention:
    def __init__(self,
                 entity_id=None,
                 entity_type=None,
                 term_type=None,
                 is_pronoun=False,
                 pronoun_type=None,
                 start_index=None,
                 end_index=None):
        self.entity_id = entity_id
        self.is_pronoun = is_pronoun
        self.entity_type = entity_type
        self.term_type = term_type
        self.pronoun_type = pronoun_type
        self.start_index = start_index
        self.end_index = end_index

    def set_clusters_and_pronouns(self, mentions, clusters, pronouns):
        clusters[self.entity_id].append((self.start_index, self.end_index))
        if self.is_pronoun:
            pronouns[self.pronoun_type].append((self.start_index, self.end_index))
        mentions.append(self)


class Quote:
    def __init__(self,
                 speaker_id=None,
                 start_index=-1,
                 end_index=-1):
        self.speaker_id = speaker_id
        self.start_index = start_index
        self.end_index = end_index

    def set_quotes(self, quotes):
        quotes.append((self.speaker_id, [self.start_index, self.end_index]))


class BWBSentence:
    """
    A class representing the annotations available for a single formatted sentence.
    # Parameters
    document_id : `str`. This is a variation on the document filename.
    sentence_id : `int`. The integer ID of the sentence within a document.
    lang : `str`. The language of this sentence.
    line: `str`. The original annotation line.
    words : `List[str]`. This is the tokens as segmented/tokenized in BWB.
    mentions : `List[Mention]`. The list of mentions in this sentence.
    pronouns: `List[str]`. The pronoun tag of each token, chosen from `P`, `O` or `-`.
    entities : `Dict[int, Tuple[str, str]]`. Entity id -> (Entity type, Terminology).
    clusters : `Dict[int, List[Span]]`.
             A dict of coreference clusters, where the keys are entity ids, and the values
             are spans. A span is a tuple of int, i.e. (start_index, end_index).
    surface_forms: `Dict[int, List[str]]`.
            A dict of surface forms, where the keys are entity ids, and the values are lists of words.
    lexical_forms: `Dict[int, List[str]]`.
            A dict of lexical forms, where the keys are entity ids, and the values are lists of words.
    quotes: List[Tuple[int, Span]]. The entity id of the speaker if it is a quote, or `None.
    pos_tags : `List[str]`. This is the Penn-Treebank-style part of speech. Default: `None.`
    """

    def __init__(
            self,
            document_id: str,
            sentence_id: int,
            lang: str,
            line: str,
            words: List[str],
            mentions: List[Mention],
            pronouns: Dict[str, List[Span]],
            entities: Dict[int, Tuple[str, str]],
            clusters: Dict[int, List[Span]],
            quotes: List[Tuple[int, Span]] = None,
            pos_tags: List[str] = None,
            lemmas: List[str] = None,
            tokens: List[str] = None
    ) -> None:
        self.document_id = document_id
        self.sentence_id = sentence_id
        self.lang = lang
        self.line = line
        self.words = words
        self.mentions = mentions
        self.pronouns = pronouns
        self.entities = entities
        self.clusters = clusters
        self.quotes = quotes
        self.pos_tags = pos_tags
        self.lemmas = lemmas
        self.tokens = tokens

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value

class BWB:
    """
    This `DatasetReader` is designed to read the BWB data.
    Once you download and unzip the anootated BWB test set, you will have a folder
    structured as follows:
    ```
    BWB_annotation/
    ── Book0
        ├── 0.chs_re.an.txt
        ├── 0.ref_re.an.txt
        ├── 1.chs_re.an.txt
        ├── 1.ref_re.an.txt
        └──  ...
    ── Book1
        ├── 0.chs_re.an.txt
        └── ...
    ── Book153
        ├── 0.chs_re.an.txt
        └── ...
    ── Book216
        ├── 0.chs_re.an.txt
        └── ...
    ── Book270
        ├── 0.chs_re.an.txt
        └── ...
    ── Book383
        ├── 0.chs_re.an.txt
        └── ...
    ```
    """

    def __init__(self, entity_dict_pth=""):
        self.en_nlp = None
        self.zh_nlp = None
        self.zh_mentions = defaultdict(list)
        self.en_mentions = defaultdict(list)
        self.entity_dict = None
        if os.path.isfile(entity_dict_pth):
            self.entity_dict = pd.read_csv(entity_dict_pth, sep='\t')

    def dataset_iterator_from_cache(self, cache_file: str = "", dir_path: str = "") -> Iterator[BWBSentence]:
        if not os.path.isfile(cache_file):
            assert os.path.isdir(dir_path), "No cached file, please specify the original directory."
            self.to_cache(dir_path, cache_file)
        with codecs.open(cache_file, "rb") as f:
            zh_list, en_list = pickle.load(f)
        for zh_sent, en_sent in zip(zh_list, en_list):
            yield zh_sent, en_sent

    def _add_spacy(self, bwbSentence):
        """
        Add pos_tags, surface_forms and lexical_forms to a BWBSentence.
        The pos_tags and lexical_forms are from spacy.
        """
        if bwbSentence.lang == "en":
            nlp = self.en_nlp
        elif bwbSentence.lang == "zh":
            nlp = self.zh_nlp
        else:
            raise RuntimeError("For now, we only provide support zh or en.")
        doc = nlp(" ".join(bwbSentence.words))
        if bwbSentence.lang == "en":
            lemmas = [w.lemma_ for w in doc]
            tokens = [w.text for w in doc]
            # start_align, end_align = align_bpe([w.text for w in doc], bwbSentence.words)
        else:
            lemmas = bwbSentence.words
            tokens = bwbSentence.words
            end_align = start_align = [i for i in range(len(bwbSentence.words))]
        # for entity_id, spans in bwbSentence.clusters.items():
        #     for span in spans:
        #         surface_forms[entity_id].append(bwbSentence.words[span[0]:span[1]])
        #         lexical_forms[entity_id].append(lemmas[start_align[span[0]]:end_align[span[1]]])

        bwbSentence.pos_tags = [w.tag_ for w in doc]
        bwbSentence.lemmas = lemmas
        bwbSentence.tokens = tokens

        return bwbSentence

    def dataset_iterator(self, dir_path: str) -> Iterator[BWBSentence]:
        """
        An iterator over the entire dataset, yielding all sentences processed.
        :param dir_path: the path to the BWB annotation folder.
        """
        for chs_path, ref_path in self.dataset_path_iterator(dir_path):
            yield from self.sentence_iterator(chs_path, ref_path)

    def dataset_iterator_doc(self, dir_path: str) -> Iterator[BWBSentence]:
        """
        An iterator over the entire dataset, yielding all sentences processed.
        :param dir_path: the path to the BWB annotation folder.
        """
        for chs_path, ref_path in self.dataset_path_iterator(dir_path):
            yield from self.dataset_document_iterator(chs_path, ref_path)

    @staticmethod
    def dataset_path_iterator(dir_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory.
        """
        logger.info("Reading BWB sentences from dataset files at: %s", dir_path)
        book_names = os.listdir(dir_path)
        for book in book_names:
            if not book.startswith('Book'):
                continue
            book_dir = os.path.join(dir_path, book)
            files = os.listdir(book_dir)
            count = 0
            for file in files:
                if file.endswith("chs_re.an.txt"):
                    count += 1
            for i in range(count):
                chs_path = os.path.join(book_dir, f"{i}.chs_re.an.txt")
                ref_path = os.path.join(book_dir, f"{i}.ref_re.an.txt")
                yield chs_path, ref_path

    def dataset_document_iterator(self, chs_path: str, ref_path: str) -> Iterator[List[BWBSentence]]:
        """
        An iterator over BWB formatted files which yields documents, regardless
        of the number of document annotations in a particular file.
        """
        chs_document: List[BWBSentence] = []
        ref_document: List[BWBSentence] = []
        book_id = chs_path.split("/")[-2]
        doc_id = chs_path.split("/")[-1].split(".")[0]
        document_id = f"{book_id}-{doc_id}"
        with codecs.open(chs_path, "r", encoding="utf8") as open_file:
            for sentence_id, line in enumerate(open_file):
                line = line.strip()
                chs_document.append(self._line_to_BWBsentence(line, "zh", document_id, sentence_id))
        with codecs.open(ref_path, "r", encoding="utf8") as open_file:
            for sentence_id, line in enumerate(open_file):
                line = line.strip()
                ref_document.append(self._line_to_BWBsentence(line, "en", document_id, sentence_id))
        if chs_document and ref_document:
            yield chs_document, ref_document

    def sentence_iterator(self, chs_path: str, ref_path: str) -> Iterator[BWBSentence]:
        """
        An iterator over the sentences in an individual CONLL formatted file.
        """
        for chs_document, ref_document in self.dataset_document_iterator(chs_path, ref_path):
            for chs_sentence, ref_sentence in zip(chs_document, ref_document):
                yield chs_sentence, ref_sentence

    def _proceed_to_next_word(self, line, words, begin_pos, lang):
        """
        return the position of the next character.
        """
        k = begin_pos
        # For English, we tokenize the sentence by white space. Punctuations are tokens.
        if lang == "en":
            if line[k] in PUNCTS:
                words.append(line[k:k + 1])
                return k + 1
            elif line[k] == '}':
                return k
            else:
                while k < len(line) and line[k] != '\s' and line[k] != ' ' and line[k] != '}' and line[k] not in PUNCTS:
                    k = k + 1
                if k - begin_pos > 1:
                    words.append(line[begin_pos:k])
                if k == len(line) or line[k] == '}' or line[k] in PUNCTS:
                    return k
                else:
                    return k + 1
        # For Chinese, we do character-level tokenization
        elif lang == "zh":
            words.append(line[k:k + 1])
            k = k + 1
        return k

    def _deal_with_ann_span(self, line, k, mention_stack, quote_stack,
                            words, entities, mentions, clusters, pronouns, quotes,
                            document_id, sentence_id, lang):
        """
        When we encouter <, we check the content recursively:
         1) is of format: <'PER', 'T', 1>{...}
         2) is of format: <'O', 1>{...}
         3) is of format: <Q, 1> .... <\Q>
        Return the cursor to the start token in the content within {...}, right after {.
        """
        # Scenario 0: Exit condition, the cursor k reaches the end of the line
        if k >= len(line):
            return k
        # Scenario 1: encounter '<'.
        # There are three sub-scenarios: <PER, T, 1> or <P, 1>, <Q, 1>, <\Q>
        if line[k] == '<':
            if k + 1 >= len(line):
                raise RuntimeError(f"Line ends with '<': {line}")
            tmp_k = k + 1
            while line[k] != '>':
                k = k + 1
            ann_span = line[tmp_k:k]
            ann_lst = ann_span.split(',')
            k = k + 1
            # now the cursor k is at the position right after <...>
            # Sub-scenario 1: <PER, T, 1> or <P, 1>
            if ann_span[0] != 'Q' and ann_span[0] != '/' and ann_span[0] != '\\':
                # Entity: ann_lst= ['PER', 'T', '1']
                if not isinstance(ann_lst[-1].isnumeric(), int):
                    raise RuntimeError(f"the entity id is not an integer in the annotation span: <{ann_span}>."
                                       f"{document_id}-{sentence_id}-{line}")

                mention = Mention(entity_id=int(ann_lst[-1]))

                if len(ann_lst) == 3:
                    # Sanity check:
                    # if mention.entity_id in entities.keys():
                        # if entities[mention.entity_id] != ann_lst:
                        #     warnings.warn(f"The entity types of two mentions that corefer to the same entity are not the same: "
                        #                   f"{ann_lst} vs {entities[mention.entity_id]} in {line}")
                    if mention.entity_id not in entities.keys():
                        entities[mention.entity_id] = (ann_lst[0], ann_lst[1])
                    mention.entity_type = ann_lst[0]
                    mention.term_type = ann_lst[1]
                # Pronoun: <O, 1>
                elif len(ann_lst) == 2:
                    mention.is_pronoun = True
                    mention.pronoun_type = ann_lst[0]
                mention.start_index = len(words)
                mention_stack.append(mention)
                # The annotated span must be followed by a {...}
                if line[k] != '{':
                    raise RuntimeError(f'the annotated span <{ann_span}> is not followed by a ''. \n'
                                       f'document_id: {document_id}, sentence_id: {sentence_id}')
                k = k + 1
                # now the cursor k is at the position right after <...>{
            # Sub-scenario 2: <Q, 1>
            elif ann_span[0] == 'Q':
                quote = Quote(speaker_id=int(ann_lst[-1]))
                quote.start_index = len(words)
                quote_stack.append(quote)
                # now the cursor k is at the position right after <...>
            # Sub-scenario 3:  <\Q> or </Q>
            elif ann_span[0] == '/' or ann_span[0] == '\\':
                quote = quote_stack.pop()
                quote.end_index = len(words)
                quote.set_quotes(quotes)
                # now the cursor k is at the position right after <...>
            else:
                raise RuntimeError(f"Invalid!")

        # Scenario 2: encounter '}'.
        elif line[k] == '}':
            mention = mention_stack.pop()
            mention.end_index = len(words)
            mention.set_clusters_and_pronouns(mentions, clusters, pronouns)
            k = k + 1
            # now the cursor k is at the position right after }

        # Scenario 3: normal text
        else:
            k = self._proceed_to_next_word(line, words, k, lang)
            # now the cursor k is at the position of the next token

        # Recursively deal with the start token in the content within {...}
        k = self._deal_with_ann_span(line, k, mention_stack, quote_stack,
                                     words, entities, mentions, clusters, pronouns, quotes,
                                     document_id, sentence_id, lang)
        return k

    def _line_to_BWBsentence(self, line: str, lang: str, document_id: str, sentence_id: int) -> BWBSentence:
        """
        Convert the raw discourse annotations in BWB to BWBsentence.
        """
        # The words in the sentence.
        words: List[str] = []
        # The pos tags of the words in the sentence, if available.
        pos_tags: List[str] = []
        # The language of this sentence.
        lang = lang
        # The quotes, if available.
        quotes = []
        # The pronoun tag of each token, chosen from `P`, `O` or `-`.
        pronouns: Dict[str, List[Span]] = {'P': [], 'O': []}
        # Entity type -> Set of entity ids.
        entities: Dict[int, Tuple[str, str]] = {}
        # Cluster id -> List of (start_index, end_index) spans.
        clusters: DefaultDict[str, Set[Optional[int]]] = defaultdict(list)
        # Mentions
        mentions: List[Mention] = []

        # Process the sentence
        # the stacks to deal with nested mentions and nested quotes, the elements are Mention or Quote objects.
        mention_stack, quote_stack = [], []
        k = 0
        k = self._deal_with_ann_span(line, k, mention_stack, quote_stack,
                                     words, entities, mentions, clusters, pronouns, quotes,
                                     document_id, sentence_id, lang)
        if k != len(line):
            raise RuntimeError(f"Error.")
        # Sanity check: whether the stack is empty
        if len(mention_stack) != 0:
            raise RuntimeError(f"the mention_stack is not empty.")
        if len(quote_stack) != 0:
            raise RuntimeError(f"the quote_stack is not empty.")

        bwbSentence = BWBSentence(
            document_id=document_id,
            sentence_id=sentence_id,
            lang=lang,
            line=line,
            words=words,
            pronouns=pronouns,
            entities=entities,
            mentions=mentions,
            clusters=dict(clusters),
            quotes=quotes
        )

        # add pos_tags, surface_forms and lexical_forms to a BWBSentence.
        bwbSentence = self._add_spacy(bwbSentence)

        return bwbSentence

    def to_file(self, chs_file: str, ref_file: str,
                       dir_path: str = "", cache_file: str = "") -> None:
        zh_dict = defaultdict(list)
        en_dict = defaultdict(list)
        for sentences in self.dataset_iterator_from_cache(cache_file, dir_path):
            for sentence, this_dict in zip(sentences, (zh_dict, en_dict)):
                this_dict['document_id'].append(sentence.document_id)
                this_dict['sentence_id'].append(sentence.sentence_id)
                this_dict['lang'].append(sentence.lang)
                this_dict['words'].append(sentence.words)
                this_dict['pronouns'].append(sentence.pronouns)
                this_dict['clusters'].append(sentence.clusters)
                this_dict['quotes'].append(sentence.quotes)
                this_dict['pos_tags'].append(sentence.pos_tags)
                this_dict['lemmas'].append(sentence.lemmas)
        if chs_file.endswith(".csv"):
            zh_df = pandas.DataFrame.from_dict(zh_dict)
            en_df = pandas.DataFrame.from_dict(en_dict)
            zh_df.to_csv(chs_file, index=False, sep='\t')
            en_df.to_csv(ref_file, index=False, sep='\t')
        else:
            with codecs.open(chs_file, "w", encoding="utf8") as chs_f, \
                    codecs.open(ref_file, "w", encoding="utf8") as ref_f:
                json.dump(zh_dict, chs_f)
                json.dump(en_dict, ref_f)

    def to_mention_csv(self, chs_file: str, ref_file: str,
                       dir_path: str = "", cache_file: str = "") -> None:
        zh_mention_dict = defaultdict(list)
        en_mention_dict = defaultdict(list)
        for sentences in self.dataset_iterator_from_cache(cache_file, dir_path):
            for sentence, this_dict in zip(sentences, (zh_mention_dict, en_mention_dict)):
                start_align, end_align = align_bpe(sentence.tokens, sentence.words)
                for mention in sentence.mentions:
                    surface_form = sentence.words[mention.start_index: mention.end_index]
                    lexical_form = sentence.lemmas[start_align[mention.start_index]: end_align[mention.end_index]]
                    this_dict['document_id'].append(sentence.document_id)
                    this_dict['sentence_id'].append(sentence.sentence_id)
                    this_dict['lang'].append(sentence.lang)
                    this_dict['entity_id'].append(mention.entity_id)
                    this_dict['entity_type'].append(mention.entity_type)
                    this_dict['term_type'].append(mention.term_type)
                    this_dict['is_pronoun'].append(mention.is_pronoun)
                    this_dict['pronoun_type'].append(mention.pronoun_type)
                    this_dict['start_index'].append(mention.start_index)
                    this_dict['end_index'].append(mention.end_index)
                    this_dict['surface_form'].append(surface_form)
                    this_dict['lexical_form'].append(lexical_form)
        zh_df = pandas.DataFrame.from_dict(zh_mention_dict)
        en_df = pandas.DataFrame.from_dict(en_mention_dict)
        zh_df.to_csv(chs_file, index=False, sep='\t')
        en_df.to_csv(ref_file, index=False, sep='\t')

    @staticmethod
    def merge2doc(list_of_BWBSentence):
        """
        :param list_of_BWBSentence: List[BWBSentence].
                A BWBSentence object has at least two fields, i.e.bwbSentence.words and bwbSentence.clusters
        :return A BWBSentence concatenated by all the sentences.
        """
        bwbDocument = copy.deepcopy(list_of_BWBSentence[0])
        bwbDocument.sentence_id = -1

        # Deal with feilds with spans
        list_of_tokens = [sent.words for sent in list_of_BWBSentence]
        pos_matrix = align_sent2doc(list_of_tokens)
        bwbDocument.clusters, _ = merge_clusters([sent.clusters for sent in list_of_BWBSentence], pos_matrix)
        bwbDocument.pronouns, _ = merge_clusters([sent.pronouns for sent in list_of_BWBSentence], pos_matrix)
        bwbDocument.quotes = merge_quotes([sent.quotes for sent in list_of_BWBSentence], pos_matrix)

        # Append for other fields:
        for bwbSentence in list_of_BWBSentence[1:]:
            for k, v in dict(bwbSentence).items():
                if isinstance(v, list) or k == 'line':
                    setattr(bwbDocument, k, getattr(bwbDocument, k) + v)
        return bwbDocument

    def to_cache(self, dir_path: str, cache_file: str) -> None:
        zh_list, en_list = [], []
        self.en_nlp = en_core_web_sm.load(disable=['parser', 'ner'])
        self.zh_nlp = zh_core_web_sm.load(disable=['parser', 'ner'])
        if 'doc' in cache_file:
            for documents in self.dataset_iterator_doc(dir_path):
                for document, this_list in zip(documents, (zh_list, en_list)):
                    document = self.merge2doc(document)
                    this_list.append(document)
        else:
            for sentences in self.dataset_iterator(dir_path):
                for sentence, this_list in zip(sentences, (zh_list, en_list)):
                    this_list.append(sentence)
        with codecs.open(cache_file, "wb") as f:
            pickle.dump((zh_list, en_list), f)


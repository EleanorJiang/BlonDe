from typing import DefaultDict, List, Optional, Iterator, Set, Tuple, Dict
from collections import defaultdict
import re
import codecs
import os
import logging
import spacy, pandas

PUNCTS = "!\"#$%&'()*+,-./:;<=>?@[\]^`{|}~…"
Span = Tuple[int, int]
TypedSpan = Tuple[int, Span]

logger = logging.getLogger(__name__)

class BWBSentence:
    """
    A class representing the annotations available for a single formatted sentence.
    # Parameters
    document_id : `str`. This is a variation on the document filename.
    sentence_id : `int`. The integer ID of the sentence within a document.
    lang : `str`. The language of this sentence.
    line: `str`. The original annotation line.
    words : `List[str]`. This is the tokens as segmented/tokenized in BWB.
    pronouns: `List[str]`. The pronoun tag of each token, chosen from `P`, `O` or `-`.
    entities : `Dict[int, Tuple[str, str]]`. Entity id -> (Entity type, Terminology).
    clusters : `Dict[int, List[Span]]`.
             A dict of coreference clusters, where the keys are entity ids, and the values
             are spans. A span is a tuple of int, i.e. (start_index, end_index).
    speakers: `int`. The entity id of the speaker if it is a quote, or `None.
    pos_tags : `List[str]`. This is the Penn-Treebank-style part of speech. Default: `None.`
    """

    def __init__(
            self,
            document_id: str,
            sentence_id: int,
            lang: str,
            line: str,
            words: List[str],
            pronouns: Dict[str, Span],
            entities: Dict[int, Tuple[str, str]],
            clusters: Dict[int, List[Span]],
            quotes: List[Tuple[int, Span]] = None,
            pos_tags: List[str] = None
    ) -> None:
        self.document_id = document_id
        self.sentence_id = sentence_id
        self.lang = lang
        self.line = line
        self.words = words
        self.pronouns = pronouns
        self.clusters = clusters
        self.entities = entities
        self.quotes = quotes
        self.pos_tags = pos_tags


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

    def dataset_iterator(self, dir_path: str) -> Iterator[BWBSentence]:
        """
        An iterator over the entire dataset, yielding all sentences processed.
        :param dir_path: the path to the BWB annotation folder.
        """
        for chs_path, ref_path in self.dataset_path_iterator(dir_path):
            yield from self.sentence_iterator(chs_path, ref_path)

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
                words.append(line[k:k+1])
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
            words.append(line[k:k+1])
            k = k + 1
        return k

    def _line_to_BWBsentence(self, line: str, lang: str, document_id: str, sentence_id: int) -> BWBSentence:
        # The words in the sentence.
        words: List[str] = []
        # The pos tags of the words in the sentence, if available.
        pos_tags: List[str] = []
        # The language of this sentence.
        lang = lang
        # The quotes, if available.
        quotes = []
        # The pronoun tag of each token, chosen from `P`, `O` or `-`.
        pronouns: Dict[str, Span] = {'P':[], 'O':[]}
        # Entity type -> Set of entity ids.
        entities: Dict[int, Tuple[str, str]] = {}
        # Cluster id -> List of (start_index, end_index) spans.
        clusters: DefaultDict[str, Set[Optional[int]]] = defaultdict(list)


        # process the sentence
        entity_id = None
        current_speaker, quote_span = None, [-1, -1]
        is_pronoun, pronoun_type = False, None
        k = 0
        while k < len(line):
            # First, let check < ... >:
            if line[k] == '<':
                if k + 1 >= len(line):
                    raise RuntimeError(f"Line ends with '<': {line}")
                # <Q, 1>
                if line[k+1] == 'Q':
                    tmp_k = k + 1
                    while line[k] != '>':
                        k = k + 1
                    ann_span = line[tmp_k:k]
                    ann_lst = ann_span.split(',')
                    current_speaker = int(ann_lst[-1])
                    quote_span[0] = len(words)
                    k = k + 1
                # <PER, T, 1> or <P, 1>
                elif line[k+1] != '/' and line[k+1] != '\\':
                    tmp_k = k + 1
                    while line[k] != '>':
                        k = k + 1
                    ann_span = line[tmp_k:k]
                    ann_lst = ann_span.split(',')
                    # Entity: ann_lst= ['PER', 'P', '1']
                    if not isinstance(ann_lst[-1].isnumeric(), int):
                        raise RuntimeError(f"the entity id is not an integer in the annotation span: <{ann_span}>.")
                    entity_id = int(ann_lst[-1])
                    if len(ann_lst) == 3:
                        entities[entity_id] = (ann_lst[0], ann_lst[1])
                    # Pronoun: <O, 1>
                    elif len(ann_lst) == 2:
                        is_pronoun = True
                        pronoun_type = ann_lst[0]
                    k = k + 1
                # <\Q>
                elif line[k+1] == '/' or line[k+1] == '\\':
                    quote_span[1] = len(words) + 1
                    quotes.append((current_speaker, quote_span))
                    current_speaker = None
                    quote_span = [-1, -1]
                    while line[k] != '>':
                        k = k + 1
                else:
                    raise RuntimeError(f"Invalid!")
            # Second, let check { ... }:
            elif line[k] == '{':
                if k + 1 >= len(line):
                    raise RuntimeError(f"Line ends with '<': {line}")
                if entity_id is None:
                    raise RuntimeError(f"{{name}} begin without any entity_id being specified: {document_id}-{sentence_id}-{line}")
                start_index = len(words)
                k = k + 1
                while line[k] != '}':
                    k = self._proceed_to_next_word(line, words, k, lang)
                k = k + 1
                end_index = len(words)
                clusters[entity_id].append((start_index, end_index))
                if is_pronoun:
                    pronouns[pronoun_type] = (start_index, end_index)
                entity_id = None
                is_pronoun = False
                pronoun_type = None
            # Third, normal text: What happened?
            else:
                k = self._proceed_to_next_word(line, words, k, lang)


        return BWBSentence(
            document_id=document_id,
            sentence_id=sentence_id,
            lang=lang,
            line=line,
            words=words,
            pronouns=pronouns,
            entities=entities,
            clusters=dict(clusters),
            quotes=quotes,
            pos_tags=pos_tags
        )

    def to_csv(self, dir_path: str, chs_file: str, ref_file: str) -> None:
        with codecs.open(chs_file, "w", encoding="utf8") as chs_f, \
                codecs.open(ref_file, "w", encoding="utf8") as ref_f:
            for sentences in self.dataset_iterator(dir_path):
                for sentence, f in zip(sentences, (chs_f, ref_f)):
                    line = f"{sentence.document_id}\t{sentence.sentence_id}\t{sentence.lang}\t" \
                           f"{sentence.words}\t{sentence.pronouns}\t" \
                           f"{sentence.clusters}\t{sentence.quotes}\t{sentence.pos_tags}"
                    f.write(f"{line}\n")

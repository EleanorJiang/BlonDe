import os
from collections import defaultdict
from typing import List, Dict, Tuple, Sequence
# util
from util.csv_util import txt2list

Document = List[str]  # Chapter
Corpus = List[Document]  # Book
Dataset = List[Corpus]  # A list of books

SYS_ID = {"smt": 0, "ms": 1, "google": 2, "bd": 3, "sent": 4, "doc": 5, "cad": 6, "ctx": 7, "pe": 8}
SYSTEMS = list(SYS_ID.keys())
BOOKS = [0, 1, 153, 215, 216, 383]

def load_corpus(data_dir: str, set: str='test') -> Tuple[Dict[str, Dataset], Dataset, Dataset, Dataset, Sequence[str]]:
    book_names = os.listdir(os.path.join(data_dir, set))  # ['book0', 'book1', 'book153', 'book216', 'book270', 'book383']
    datasets, ref_dataset, an_dataset, ner_dataset = defaultdict(list), [], [], []
    for book in book_names:
        if book.startswith('.'):
            continue
        corpora, ref_corpus, an_corpus, ner_corpus = defaultdict(list), [], [], []
        book_dir = os.path.join(data_dir, set, book)
        # Get the number of chapters in this book.
        files = os.listdir(book_dir)
        count = 0
        for file in files:
            if "an.txt" in file:
                count += 1
        # Get the number of chapters in this book.
        for i in range(count):
            # Load all the system outputs along with reference, annotation
            an_path = os.path.join(book_dir, "{}.an.txt".format(i))
            ner_path = os.path.join(book_dir, "{}.ner_re.txt".format(i))
            ref_path = os.path.join(book_dir, "{}.ref_re.txt".format(i))
            ms_path = os.path.join(book_dir, "{}.mt_re.txt".format(i))
            sent_path = os.path.join(book_dir, "{}.sent_re.txt".format(i))
            doc_path = os.path.join(data_dir, "Naive", book, "{}.chs_re.txt".format(i))
            cad_path = os.path.join(data_dir, "CAD", book, "{}.chs_re.txt".format(i))
            ctx_path = os.path.join(book_dir, "{}.ctx_re.txt".format(i))
            pe_path = os.path.join(book_dir, "{}.pe.txt".format(i))
            # The following system outputs are in separate folders
            smt_path = os.path.join(data_dir, "SMT_Hiero", book, "{}.chs_re.txt.SMT".format(i))
            google_path = os.path.join(data_dir, "Google", book, "{}.chs_re.txt.Google".format(i))
            bd_path = os.path.join(data_dir, "Baidu", book, "{}.chs_re.txt.Baidu".format(i))

            # We have 9 system outputs in total.
            paths = dict(zip(SYSTEMS,
                             [smt_path, ms_path, google_path, bd_path, sent_path, doc_path, cad_path, ctx_path, pe_path]))

            ref_doc, an_doc, ner_doc = txt2list(ref_path), txt2list(an_path), txt2list(ner_path)
            # Sanity Check
            assert len(an_doc) == len(ref_doc), \
                f"{book}-{i}: Oh no! txt2list for an is not working properly!"
            assert len(ner_doc) == len(ref_doc), \
                f"{book}-{i}: Oh no! txt2list for ner is not working properly!"
            # No problem, we proceed:
            ref_corpus.append(ref_doc)
            an_corpus.append(an_doc)
            ner_corpus.append(ner_doc)

            for system_name, path in paths.items():
                document = txt2list(path)
                assert len(document) == len(ref_doc), \
                    f"{book}-{i}: Oh no! txt2list for {system_name} is not working properly!"
                corpora[system_name].append(document)

        for system_name, path in paths.items():
            datasets[system_name].append(corpora[system_name])
        ref_dataset.append(ref_corpus)
        an_dataset.append(an_corpus)
        ner_dataset.append(ner_corpus)

    return datasets, ref_dataset, an_dataset, ner_dataset, book_names

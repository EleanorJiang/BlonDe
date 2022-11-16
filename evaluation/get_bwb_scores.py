import os, sys, argparse, logging
from collections import defaultdict
import pandas as pd
import numpy as np
import json
from typing import List, Tuple
from operator import itemgetter

# util
sys.path.insert(0, '.')
from util.logging_util import init_logging
from util.csv_util import list2txt, txt2list, flat_list
# For BlonDe
from blonde import BLONDE
from BWB.entity import get_entity_counts, count_entity_spans

"""
We test upon the following systems.
"""
SYSTEMS = ["smt", "ms", "google", "bd", "sent", "doc", "pe"]


class Evaluate:
    def __init__(self, data_dir, bwb_annotation_dir, output_dir, tmp_dir, systems, df_name,
                 chosen_books=None):
        self.data_dir = data_dir
        self.bwb_annotation_dir = bwb_annotation_dir
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.systems = systems
        self.df_name = df_name
        self.chosen_books = chosen_books
        self.entity_counts = get_entity_counts(bwb_annotation_dir)

    def load_system_outputs(self):
        """
        load system outputs from 'data_dir' into 80 documents
        :return corpora: Dict[str, Dict[str, List[str]]]: a dict, where the key is the system name,
                and the value is a dict, where the key is the document_id, and the value is a list of sentences.
        """
        corpora = {}
        for system_name in SYSTEMS:
            corpora[system_name] = {}
        for book_id in [0, 1, 153, 216, 270, 383]:
            book_dir = os.path.join(self.data_dir, "test", "book{}".format(book_id))
            files = os.listdir(book_dir)
            count = 0
            for file in files:
                if "an.txt" in file:
                    count += 1
            for i in range(count):
                document_id = f"Book{book_id}-{i}"
                # Load all the system outputs along with reference, annotation
                ms_path = os.path.join(book_dir, "{}.mt_re.txt".format(i))
                sent_path = os.path.join(book_dir, "{}.sent_re.txt".format(i))
                doc_path = os.path.join(book_dir, "{}.ctx_re.txt".format(i))
                pe_path = os.path.join(book_dir, "{}.pe.txt".format(i))
                # The following system outputs are in separate folders
                smt_path = os.path.join(self.data_dir, "SMT_Hiero", f"book{book_id}", "{}.chs_re.txt.SMT".format(i))
                google_path = os.path.join(self.data_dir, "Google", f"book{book_id}", "{}.chs_re.txt.Google".format(i))
                bd_path = os.path.join(self.data_dir, "Baidu", f"book{book_id}", "{}.chs_re.txt.Baidu".format(i))
                paths = dict(zip(SYSTEMS,
                                 [smt_path, ms_path, google_path, bd_path, sent_path, doc_path, pe_path]))
                for system_name, path in paths.items():
                    document = txt2list(path)
                    corpora[system_name][document_id] = document
        return corpora

    def append_meta_info(self, scores, category, system_name, book_id, chapter_id):
        """
        Note that scores["book_id"] is a value in [0, 1, 153, 215, 216, 383].
        We are doing this [0,1,2,3,4,5,6] to [0, 1, 153, 215, 216, 383] mapping
        to match the format of human_df (the result we got from human evaluation).
        """
        scores["category"].append(category)
        scores["system"].append(system_name)
        scores["book_id"].append(book_id)
        scores["chapter_id"].append(chapter_id)

    def _append_scores(self, scores, sentences, entity_counts):
        (en_entity_list, ref_entity_counts, chs_entity_counts) = entity_counts.values()
        sys_entity_counts = []
        for i, sentence in enumerate(sentences):
            sys_count = []
            for entity in en_entity_list:
                sys_count.append(count_entity_spans(sentence, entity, lang="en"))
            sys_entity_counts.append(sys_count)
        sys_entity_counts = np.array(flat_list(sys_entity_counts))
        ref_entity_counts = np.array(flat_list(ref_entity_counts))
        chs_entity_counts = np.array(flat_list(chs_entity_counts))
        assert len(sys_entity_counts) == len(ref_entity_counts) == len(chs_entity_counts)
        def get_scores(sys_entity_counts, denominator_entity_counts):
            clipped_sys_entity_counts = np.minimum(sys_entity_counts, denominator_entity_counts)
            precision = np.sum(clipped_sys_entity_counts) / np.sum(sys_entity_counts)
            recall = np.sum(clipped_sys_entity_counts) / np.sum(denominator_entity_counts)
            f1 = 2 * precision * recall / (precision + recall)
            return precision, recall, f1
        ref_precision, ref_recall, ref_f1 = get_scores(sys_entity_counts, ref_entity_counts)
        chs_precision, chs_recall, chs_f1 = get_scores(sys_entity_counts, chs_entity_counts)
        scores["ref_precision"].append(ref_precision)
        scores["ref_recall"].append(ref_recall)
        scores["ref_f1"].append(ref_f1)
        scores["chs_precision"].append(chs_precision)
        scores["chs_recall"].append(chs_recall)
        scores["chs_f1"].append(chs_f1)

    def get_actual_document_scores(self):
        """
        We treat a book as a corpus, and evaluate metrics by book, since the domains may differ among books.
        return `score_df`: a panda dataframe, where the columns are ['book-chap', 'system', BLEU, BlonDe-r, ...]
        """
        scores = defaultdict(list)
        corpora = self.load_system_outputs()
        for system_name, corpus in corpora.items():
            for document_id, sentences in corpus.items():
                book_id, chapter_id = document_id.split('-')
                for category in self.entity_counts.keys():
                    if document_id in self.entity_counts[category]:
                        self.append_meta_info(scores, category, system_name, book_id[4:], chapter_id)
                        entity_counts = self.entity_counts[category][document_id]
                        self._append_scores(scores, sentences, entity_counts)
        Evaluate.save(scores, self.output_dir, self.df_name)

    @staticmethod
    def save(scores, output_dir, name):
        with open(os.path.join(output_dir, f"{name}.json"), 'w') as json_file:
            json.dump(scores, json_file)
        score_df = pd.DataFrame.from_dict(scores)
        # save to csv, set the format to four decimal places
        score_df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False, float_format='%.4f')


def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')
    parser.add_argument('--data_dir', default='../../DATA/BWB_dataset', help='path to the BWB dataset')
    parser.add_argument('--bwb_annotation_dir', default='../../DATA/BWB_annotation_20220727', help='path to the ourput directory')
    parser.add_argument('--out_dir', default='output_bwb', help='path to the ourput directory')
    parser.add_argument('--df_name', default='bwb_score_df', help='the name of the output csv file.')
    parser.add_argument('--tmp_dir', default='.tmp', help='path to the temporary output of evaluation data')
    parser.add_argument('--systems', default=None, type=str,
                        help='Choose from SYSTEMS, for example: smt, ms, google ...')
    parser.add_argument('--para', default=None, type=str, help='hyper parameters for BlonDe, separated by `,`')
    parser.add_argument('--log_file', default='bwb_evaluate.log', help='specify the log file')
    parser.add_argument('--n_samples', default=10, type=int, help='the number of samples for bootstrap resampling')
    parser.add_argument('--books', default=None, type=str,
                        help='Choose from book0, book1, book153, book216,book270,book383')

    return parser.parse_args()


def main(args):
    systems, choosed_books = SYSTEMS, None
    if args.systems is not None:
        systems = args.systems.split(',')
    if args.books is not None:
        choosed_books = args.books.split(',')
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    if not os.path.isdir(args.tmp_dir):
        os.mkdir(args.tmp_dir)
    if args.log_file is not None:
        log_file = os.path.join(args.out_dir, args.log_file)
        init_logging(log_file)
    evaluate = Evaluate(args.data_dir, args.bwb_annotation_dir, args.out_dir, args.tmp_dir, systems, args.df_name)
    evaluate.get_actual_document_scores()


if __name__ == '__main__':
    args = get_args()
    main(args)

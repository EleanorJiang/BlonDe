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
from util.csv_util import list2txt, txt2list
from evaluation.dataset import load_corpus
# For BlonDe
from BlonDe.BlonDe import BlonDe
# For other metrics
import sacrebleu
from other_metrics.rc import lc_and_rc
from nlgeval import compute_metrics


"""
We report the scores of the following metrics.
"""
METRICS = ["BLEU",
           "dB-r", "dB-p", "dB-F1",  # dBlonDe (recall, precision, F1)
           "sB-r", "sB-p", "sB-F1",  # BlonDe (recall, precision, F1)
           "sBp-r", "dBp-r",  # BlonDe plus (recall only)
           # Linguistics Phenomena: enity, pronoun, verb, ambiguity, ellipsis (recall, precision, F1)
           # "entity-r", "pron-r", "verb-r", "dm-r", "amb-r", "ell-r",
           # "entity-p", "pron-p", "verb-p",
           # "entity-F1", "pron-F1", "verb-F1",
           # Other Metrics:
           "lc", "rc",  # (Wong and Kit, 2012)
           # "NLGEval"
           # "METEOR", "TER", "ROUGE", "CIDEr",  # Sentence-level Metrics (other than BLUE)
           # "SkipThoughts", "Embedding", 'Vector', "GreedyMatching"  # Embedding-based Metrics
           ]

"""
We test upon the following systems.
"""
SYSTEMS = ["smt", "ms", "google", "bd", "sent", "doc", "pe"]
PART_SYSTEMS = ["smt", "ms", "sent", "ctx", "pe"]  # for human evaluation


class Evaluate:
    def __init__(self, data_dir, output_dir, tmp_dir, metrics, systems, show_BlonDe_detail=False, set='test',
                 chosen_books=None):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.tmp_dir = tmp_dir
        self.metrics = metrics
        self.systems = systems
        self.chosen_books = chosen_books
        self.show_BlonDe_detail = show_BlonDe_detail
        self.set = set
        datasets, ref_dataset, an_dataset, ner_dataset, book_names = load_corpus(data_dir, set)
        self.datasets = datasets
        self.ref_dataset = ref_dataset
        self.an_dataset = an_dataset
        self.ner_dataset = ner_dataset
        self.book_names = book_names
        self.do_lc_and_rc = False
        if 'rc' in metrics:
            self.do_lc_and_rc = True

    def append_meta_info(self, scores, system_name, book_id, chapter_id):
        """
        Note that scores["book_id"] is a value in [0, 1, 153, 215, 216, 383].
        We are doing this [0,1,2,3,4,5,6] to [0, 1, 153, 215, 216, 383] mapping
        to match the format of human_df (the result we got from human evaluation).
        """
        scores["system"].append(system_name)
        scores["book_id"].append(self.book_names[book_id].strip('book'))
        scores["chapter_id"].append(chapter_id)

    @staticmethod
    def save(scores, output_dir, name):
        with open(os.path.join(output_dir, f"{name}.json"), 'w') as json_file:
            json.dump(scores, json_file)
        score_df = pd.DataFrame.from_dict(scores)
        score_df.to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)

    def _append_scores(self, scores, corpus, flat_corpus, BlonDe_plus=None, bleu=None, ter=None,
                      nlgeval_txt: Tuple[str, List[str]]=None):
        BlonDe_score = BlonDe_plus.corpus_score(corpus, None)  # we use cached references
        bleu_score = bleu.corpus_score(flat_corpus, None)
        ter_score = ter.corpus_score(flat_corpus, None)
        scores["BLEU"].append(bleu_score.score)
        scores["BlonDe_plus-r"].append(BlonDe_score.recall)
        scores["BlonDe_plus-p"].append(BlonDe_score.precision)
        scores["BlonDe_plus-F1"].append(BlonDe_score.F1)
        scores["BlonDe-r"].append(BlonDe_score.detail['recalls']['sBlonDe'])
        scores["BlonDe-p"].append(BlonDe_score.detail['precisions']['sBlonDe'])
        scores["BlonDe-F1"].append(BlonDe_score.detail['F1s']['sBlonDe'])
        scores["dBlonDe-r"].append(BlonDe_score.detail['recalls']['dBlonDe'])
        scores["dBlonDe-p"].append(BlonDe_score.detail['precisions']['dBlonDe'])
        scores["dBlonDe-F1"].append(BlonDe_score.detail['F1s']['dBlonDe'])
        scores["ter_score"].append(ter_score.score)
        if self.do_lc_and_rc:
            rc, lc = lc_and_rc(flat_corpus)
            scores["lc"].append(lc)
            scores["rc"].append(rc)
        if nlgeval_txt is not None:
            sys_txt, ref_txts = nlgeval_txt[0], nlgeval_txt[1]
            metrics_dict = compute_metrics(hypothesis=sys_txt, references=ref_txts)
            scores["METEOR"].append(metrics_dict['METEOR'])
            scores["ROUGE"].append(metrics_dict['ROUGE_L'])
            scores["CIDEr"].append(metrics_dict['CIDEr'])
            scores["SkipThoughts"].append(metrics_dict['SkipThoughtCS'])
            scores["Embedding"].append(metrics_dict['EmbeddingAverageCosineSimilarity'])
            scores["Vector"].append(metrics_dict['VectorExtremaCosineSimilarity'])
            scores["GreedyMatching"].append(metrics_dict['GreedyMatchingScore'])
        if self.show_BlonDe_detail:
            for measure, score_dict in BlonDe_score.detail.items():
                for key, score in score_dict.items():
                    if key not in ['sBlonDe', 'dBlonDe']:
                        scores[f"{key}-{measure[:1]}"].append(score)
        for key, values in scores.items():
            if type(values[-1]) is str or type(values[-1]) is int:
                logging.info(f"{key}: {values[-1]}\t")
            else:
                if values[-1] < 1:
                    values[-1] *= 100
                logging.info(f"{key}: {values[-1]:.2f}\t")
        logging.info('')

    def _get_corpus_scores(self, scores, corpora, ref_corpus, an_corpus, ner_corpus, book_id, chapter_id= 'actual'):
        
        """
        We treat a book as a corpus, and evaluate metrics by book, since the domains may differ among books.
        return `score_df`: a panda dataframe, where the columns are ['book-chap', 'system', BLEU, BlonDe-r, ...]
        """
        BlonDe_plus = BlonDe(average_method='geometric',
                           references=[ref_corpus],
                           annotation=an_corpus,
                           ner_refined=ner_corpus
                           )
        flat_ref_corpus = [sentence for document in ref_corpus for sentence in document]
        bleu = sacrebleu.BLEU(references=[flat_ref_corpus])
        ter = sacrebleu.TER(references=[flat_ref_corpus])
        ref_txt = os.path.join(self.tmp_dir, f'{book_id}.ref.txt')
        list2txt(flat_ref_corpus, ref_txt)
        # now let's test the 9 system outputs
        for idx, system_name in enumerate(self.systems):
            self.append_meta_info(scores, system_name, book_id, chapter_id)
            corpus = corpora[idx]  # corpora = self.datasets[*][book_id]
            flat_corpus = [sentence for document in corpus for sentence in document]
            assert len(flat_ref_corpus) == len(flat_corpus), "Oh no! flat_corpus is not working properly!"
            nlgeval_txt = None
            if 'NLGEval' in self.metrics:
                sys_txt = os.path.join(self.tmp_dir, f'{book_id}.{system_name}.txt')
                list2txt(flat_corpus, sys_txt)
                # flat_corpus_2 = txt2list(sys_txt)
                # assert len(flat_corpus_2) == len(flat_corpus), "Oh no! list2txt is not working properly!"
                nlgeval_txt = (sys_txt, [ref_txt])
            self._append_scores(scores, corpus, flat_corpus, BlonDe_plus,
                          bleu, ter, nlgeval_txt)
        return scores

    def _do_this_book(self, book_name):
        return (self.chosen_books is None) or (str(book_name) in self.chosen_books)
    
    
    def get_actual_document_scores(self):
        # now let's get the per document scores
        scores = defaultdict(list)
        for book_id, (ref_corpus, an_corpus, ner_corpus) in enumerate(zip(self.ref_dataset, self.an_dataset, self.ner_dataset)):
            book_name = self.book_names[book_id]
            if not book_name:
                continue
            for chapter_id, (ref_doc, an_doc, ner_doc) in enumerate(zip(ref_corpus, an_corpus, ner_corpus)):
                assert len(ref_doc) == len(an_doc),  f"{book_name}-{chapter_id}: Oh no! len(ref_doc) != len(an_doc)! "
                assert len(ref_doc) == len(ner_doc),  f"{book_name}-{chapter_id}: Oh no! len(ref_doc) != len(ner_doc)!"
                corpora = [[self.datasets[sys][book_id][chapter_id]] for sys in self.systems]
                self._get_corpus_scores(scores, corpora, [ref_doc], [an_doc], [ner_doc], book_id, chapter_id)
        Evaluate.save(scores, self.output_dir, "document_scores")

    def get_actual_corpus_scores(self):
        """
        We treat a book as a corpus, and evaluate metrics by book, since the domains may differ among books.
        return `score_df`: a panda dataframe, where the columns are ['book-chap', 'system', BLEU, BlonDe-r, ...]
        """
        scores = defaultdict(list)
        for book_id, (ref_corpus, an_corpus, ner_corpus) in enumerate(zip(self.ref_dataset, self.an_dataset, self.ner_dataset)):
            book_name = self.book_names[book_id]
            if not book_name:
                continue
            corpora = [self.datasets[sys][book_id] for sys in self.systems]
            self._get_corpus_scores(scores, corpora, ref_corpus, an_corpus, ner_corpus, book_id, chapter_id='actual')
        Evaluate.save(scores, self.output_dir, "corpus_scores")

    @staticmethod
    def _bootstrap_resample_idxs(ref_corpus, n_samples=100):
        """Performs bootstrap resampling for a single system to estimate
        a confidence interval around the true mean.
        """

        # Set numpy RNG's seed
        # If  given -> Fix to the given value
        # If given but =='[Nn]one', don't fix the seed i.e. pull entropy from OS
        seed = os.environ.get('SACREBLEU_SEED', '12345')
        _seed = None if seed.lower() == 'none' else int(seed)
        rng = np.random.default_rng(_seed)

        # The indices that'll produce all bootstrap resamples at once
        idxs = rng.choice(len(ref_corpus), size=(n_samples, len(ref_corpus)), replace=True)

        return idxs

    def get_boostrap_scores(self, n_samples):
        scores = defaultdict(list)
        for book_id, (ref_corpus, an_corpus, ner_corpus) in enumerate(zip(self.ref_dataset, self.an_dataset, self.ner_dataset)):
            book_name = self.book_names[book_id]
            if not book_name:
                continue
            idxs = Evaluate._bootstrap_resample_idxs(ref_corpus, n_samples)
            for i, idx in enumerate(idxs):
                new_ref_corpus, new_an_corpus, new_ner_corpus = itemgetter(*idx)(ref_corpus), \
                                                    itemgetter(*idx)(an_corpus), itemgetter(*idx)(ner_corpus)
                new_corpora = [itemgetter(*idx)(self.datasets[sys][book_id]) for sys in self.systems]
                self._get_corpus_scores(scores, new_corpora, new_ref_corpus, new_an_corpus, new_ner_corpus,
                                       book_id, chapter_id=f'boostrap-{i}')
        Evaluate.save(scores, self.output_dir, "boostrap_scores")


def get_args():
    """ Defines generation-specific hyper-parameters. """
    parser = argparse.ArgumentParser('Sequence to Sequence Model')
    parser.add_argument('--seed', default=42, type=int, help='pseudo random number generator seed')
    parser.add_argument('--data_dir', default='/PATH/TO/DATA/BWB_dataset', help='path to data directory')
    parser.add_argument('--out_dir', default='output', help='path to the ourput directory')
    parser.add_argument('--df_name', default='score_df', help='the name of the output csv file.')
    parser.add_argument('--tmp_dir', default='.tmp', help='path to the temporary output of evaluation data')
    parser.add_argument('--systems', default=None, type=str, help='Choose from SYSTEMS, for example: smt, ms, google ...')
    parser.add_argument('--metrics', default=None, type=str, help='Choose from METRICS, for example: BlEU, BlonDe, NLGEval')
    parser.add_argument('--para', default=None, type=str, help='hyper parameters for BlonDe, separated by `,`')
    parser.add_argument('--log_file', default='evaluate.log', help='specify the log file')
    parser.add_argument('--n_samples', default=10, type=int, help='the number of samples for bootstrap resampling')
    parser.add_argument('--detail', default=True, type=bool, help= 'for BlonDe, to print the scores of different linguistics categories')
    parser.add_argument('--books',default=None, type=str, help='Choose from book0, book1, book153, book216,book270,book383')

    return parser.parse_args()


def main(args):
    metrics, systems = METRICS, SYSTEMS
    if args.metrics is not None:
        metrics = args.metrics.split(',')
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
    evaluate = Evaluate(args.data_dir, args.out_dir, args.tmp_dir, metrics, systems, args.detail, 'test', choosed_books)
    # evaluate.get_actual_corpus_scores()
    # evaluate.get_actual_document_scores()
    evaluate.get_boostrap_scores(args.n_samples)


if __name__ == '__main__':
    args = get_args()
    main(args)

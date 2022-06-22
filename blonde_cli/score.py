#!/usr/bin/env python
import argparse, os
from blonde import BLONDE, CATEGORIES
from blonde import __version__ as VERSION


def parse_args():
    arg_parser = argparse.ArgumentParser(
        description="BlonDe: automatic evaluation metric for document-level machine translation.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    arg_parser.add_argument('-r', '--reference', type=str, nargs="+", required=True,
                            help='reference file path(s), each line is a sentence')
    arg_parser.add_argument('-s', '--system', type=str, required=True,
                            help='system file path, each line is a sentence')
    # BLONDE related arguments
    blonde_args = arg_parser.add_argument_group('BLONDE related arguments')
    blonde_args.add_argument('--categories', '-c', type=str, nargs='+',
                             default=('tense', 'pronoun', 'entity', 'dm', 'n-gram'),
                             help="The categories of BLONDE. "
                                  "Default: ('tense', 'pronoun', 'entity', 'dm', 'n-gram')")
    blonde_args.add_argument('--average-method', '-aver', choices=('geometric', 'arithmetic'), default='geometric',
                             help='The average method to use, geometric or arithmetic(Defaults: (geometric, 1)')
    blonde_args.add_argument('--smooth-method', '-sm', choices=BLONDE.SMOOTH_DEFAULTS.keys(), default='exp',
                           help='Smoothing method: exponential decay, floor (increment zero counts), '
                                'add-k (increment num/denom by k for n>1), or none. (Default: %(default)s)')
    blonde_args.add_argument('--smooth-value', '-sv', type=float, default=None,
                           help='The smoothing value. Only valid for floor and add-k. '
                                f"(Defaults: floor: {BLONDE.SMOOTH_DEFAULTS['floor']}, "
                                f"add-k: {BLONDE.SMOOTH_DEFAULTS['add-k']})")
    blonde_args.add_argument('--lowercase', '-lc', type=bool, default=True,
                             help='If True, enables case-insensitivity. (Default: %(default)s)')
    # Weight
    blonde_args.add_argument('--override-weights', '-w', action='store_true', default=False,
                             help='Whether to customize the weights of BLONDE')
    blonde_args.add_argument('--reweight', '-rw', action='store_true', default=False,
                             help='Whether to reweight the weights of BLONDE to 1')
    blonde_args.add_argument('--weight-tense', '-wt', type=float, nargs='+',
                             default=(1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7),
                             help="The weights of TENSE (verb types), should be a tuple of length 7, "
                                  "corresponds to ('VBD', 'VBN', 'VBP', 'VBZ', 'VBG', 'VB', 'MD'). "
                                  "Defaults: (1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7). "
                                  "Only valid when `override_weights` is used")
    blonde_args.add_argument('--weight-pronoun', '-wp', type=float, nargs='+', default=(0.5, 0.5, 0, 0),
                             help="The weights of PRONOUN, should be a tuple of length 4, "
                                  "corresponds to ('masculine', 'feminine', 'neuter', 'epicene'). "
                                  "Defaults: (0.5, 0.5, 0, 0). "
                                  "Only valid when `override_weights` is used")
    blonde_args.add_argument('--weight-entity', '-we', type=float, nargs='+', default=(1, 0),
                             help='The weights of PERSON and NONPERSON entities, Defaults: (1/2, 1/2). '
                                  'Only valid when `override_weights` is used')
    blonde_args.add_argument('--weight-discourse-marker', '-wdm', type=float, nargs='+', default=(0.2, 0.2, 0.2, 0.2, 0.2),
                             help="The weights of DISCOURSE MARKER, should be a tuple of length 5, "
                                   "corresponds to ('comparison', 'cause', 'conjunction', 'asynchronous', 'synchronous'). "
                                  "Defaults: (0.5, 0.5, 0, 0). Only valid when `override_weights` is used")
    # BLONDE PLUS
    blonde_plus_args = arg_parser.add_argument_group('BLONDE PLUS related arguments, annotation required')
    blonde_plus_args.add_argument('--plus', '-p', action='store_true', default=False,
                                  help='Whether to add BLONDE PLUS categories. '
                                       'If so, please provide annotation files that are in the required format.')
    blonde_plus_args.add_argument('--annotation', '-an', type=str, default=None,
                                  help='Annotation file path, each line is the annotation corresponding a sentence. '
                                       'See README for annotation format')
    blonde_plus_args.add_argument('--ner-refined', '-ner', type=str, default=None,
                                  help='Named entity file path, each line is the named entities '
                                       'corresponding a sentence. '
                                       'If provided, the annotated named entities instead of the automated recognized '
                                       'ones are used in BLONDE. '
                                       'See README for named entity annotation format')
    blonde_plus_args.add_argument('--plus-categories', '-pc', type=str, nargs='+', default=('ambiguity', 'ellipsis'),
                                  help="The categories that your annotation files contain, "
                                       "Defaults: ('ambiguity', 'ellipsis'). "
                                       "Only valid when `plus` is used")
    blonde_plus_args.add_argument('--plus-weights', '-pw', type=float, nargs='+', default=(1, 1),
                                  help='The corresponding weights of plus categories, '
                                       'should be in the same length as `plus_categories`. '
                                       'Defaults: (1, 1). Only valid when `plus` is used')

    arg_parser.add_argument('--version', '-V', action='version', version='%(prog)s {}'.format(VERSION))
    args = arg_parser.parse_args()
    return args


def main():
    args = parse_args()

    if os.path.isfile(args.system):
        with open(args.system) as f:
            sys = [line.strip() for line in f]

        refs = []
        for ref_file in args.reference:
            assert os.path.exists(ref_file), f"reference file {ref_file} doesn't exist"
            with open(ref_file) as f:
                curr_ref = [line.strip() for line in f]
                assert len(curr_ref) == len(sys), f"# of sentences in {ref_file} doesn't match the # of candidates"
                refs.append([curr_ref])
        refs = list(zip(*refs))
    elif os.path.isfile(args.reference[0]):
        assert os.path.exists(args.cand), f"system file {args.cand} doesn't exist"

    categories = {}
    for category in args.categories:
        categories[category] = CATEGORIES[category]

    weights = BLONDE.WEIGHTS_DEFAULTS
    if args.override_weights:
        weights = {
            "tense": args.weight_tense,
            "pronoun": args.weight_pronoun,
            "entity": args.weight_entity,
            "dm": args.weight_discourse_marker,
            "n-gram": (0.25, 0.25, 0.25, 0.25)
            }

    plus_categories, plus_weights, annotation = None, None, None
    if args.plus:
        plus_categories = args.plus_categories
        plus_weights = args.plus_weights
        if os.path.isfile(args.annotation):
            with open(args.annotation) as f:
                annotation = [line.strip() for line in f]

    blond = BLONDE(weights=weights,
                 weight_normalize=args.reweight,
                 average_method=args.average_method,
                 categories=categories,
                 plus_categories=plus_categories,
                 plus_weights=plus_weights,
                 lowercase=args.lowercase,
                 smooth_method=args.smooth_method,
                 smooth_value=args.smooth_value,
                 references=refs,
                 annotation=[annotation],
                 ner_refined=args.ner_refined)
    score = blond.corpus_score([sys], annotation=[annotation])
    print(score)

if __name__ == "__main__":
    main()

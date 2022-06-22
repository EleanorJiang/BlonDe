#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__version__ = '0.1.2'


VB_TYPE = ('VBD', 'VBN', 'VBP', 'VBZ', 'VBG', 'VB', 'MD')
# VB_MAP = dict(zip(VB_TYPE, VB_TYPE))

PRONOUN_TYPE = ('masculine', 'feminine', 'neuter', 'epicene')
PRONOUN_MAP = {"masculine": ["he", "his", "him", "He", "His", "Him", "himself", "Himself"],
               "feminine": ["she", "her", "hers", "She", "Her", "Hers", "herself", "Herself"],
               "neuter": ["it", "its", "It", "Its", "itself", "Itself"],
               "epicene": ["they", "their", "them", "They", "Their", "Them", "themselves", "Themselves"]
               }
# PRONOUN_TYPE = list(PRONOUN_MAP.keys())

"""
DM_MAP is based on the PDTB hierarchy
The top hierarchy are: 'comparison', 'contingency', 'expansion', 'temporal'
    - Comparison: combine "concession" and "contrast"
    - Contingency: only consider "cause"
    - Expansion: only consider "conjunction"
    - Temporal: "synchronous" and "asynchronous"
"""
DM_TYPE = ('comparison', 'cause', 'conjunction', 'asynchronous', 'synchronous')
DM_MAP = {
    # Comparison:
    'comparison': ["but", "while", "however", "although", "though", "still", "yet", "whereas",
                   "on the other hand", "in contrast", "by contrast", "by comparison", "conversely"],
    # Contingency:
    'cause': ["if", "because", "so", "since", "thus", "hence", "as a result", "therefore", "thereby",
              "accordingly", "consequently", "in consequence", "for this reason"], #"because of that", "because of this"
    # "condition": ["if", "as long as" , "provided that", "assuming that", "given that"],
    # Expansion:
    'conjunction': ["also", "in addition", "moreover", "additionally", "besides", "else,", "plus"],
    # "instantiation": ["for example", "for instance"],
    # "alternative": ["instead", "or", "unless", "separately" ],
    # "restatement": ["indeed", "in fact", "clearly", "in other words", "specifically"]
    # Temporal:
    'asynchronous': ["when", "after", "then", "before",
                     "until", "later", "once", "afterward", "next"],
    'synchronous': ["meantime", "meanwhile", "simultaneously"]
}
# DM_TYPE = list(DM_MAP.keys())


CATEGORIES = {
    "tense": VB_TYPE,
    "pronoun": PRONOUN_TYPE,
    "entity": ["PERSON", "NON-PERSON"],
    "dm": DM_TYPE,
    "n-gram": [1, 2, 3, 4]
}

WEIGHTS = {
        "tense": (1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7),
        "pronoun": (0.5, 0.5, 0, 0),
        "entity": (1, 0),
        "dm": (0.2, 0.2, 0.2, 0.2, 0.2),
        "n-gram": (0.25, 0.25, 0.25, 0.25)
}
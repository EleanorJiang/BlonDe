from typing import List, Tuple, Dict
from copy import deepcopy
import numpy as np


def lcs(S,T):
    """
    LONGEST COMMON SUBSTRING ALGORITHM
    """
    m = len(S)
    n = len(T)
    counter = [[0]*(n+1) for x in range(m+1)]
    longest = 0
    lcs_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                c = counter[i][j] + 1
                counter[i+1][j+1] = c
                if c > longest:
                    lcs_set = set()
                    longest = c
                    lcs_set.add(S[i-c+1:i+1])
                elif c == longest:
                    lcs_set.add(S[i-c+1:i+1])

    return lcs_set


def longestCommonPrefix(a):
    """
    to find longest common prefix of given array of words.
    :param a: a list of strings
    :return:
    """
    size = len(a)

    # if size is 0, return empty string
    if (size == 0):
        return ""

    if (size == 1):
        return a[0]

    # sort the array of strings
    a.sort()

    # find the minimum length from
    # first and last string
    end = min(len(a[0]), len(a[size - 1]))

    # find the common prefix between
    # the first and last string
    i = 0
    while (i < end and
           a[0][i] == a[size - 1][i]):
        i += 1

    pre = a[0][0: i]
    return pre


def align_bpe(src_tokens: List[str], tgt_tokens: List[str],
              vocab_match: Dict[str, str]=None) -> List[int]:
    """
    The current version is basically detokenizing the first sentence and then matching character position of the second tokenization.
    So it doesn't deal with the first tokenization at all
    Also, the current version cannot deal with UNK mismatch
    :param src_tokens: ["dr", "one", "is"]
    :param tgt_tokens: ["d", "rone is"]
    :param vocab_match: {sos_src: sos_tgt,
                         eos_src: eos_tgt,
                         unk_src: unk_tgt
                         }
                        (Note that " ", "SOS", "EOS" and "UNK" could be hidden anywhere in the string.
                        SOS and EOS represent the start and end of a sequence, respectively.)
    :return start_alignment: same length as tgt_tokens, each token in tgt_tokens is assigned the id of corresponding
                start position in src_tokens.
            end_alignment: same length as tgt_tokens, each token in tgt_tokens is assigned the id of corresponding
                end position in src_tokens.
    For example,
    if src_tokens=["dr", "one", "is", "a"],  tgt_tokens=["d", "rone is", "a"] - [0, 0, 3], [0, 2, 3]
    if src_tokens=["d", "rone is", "a"],  tgt_tokens=["dr", "one", "is", "a"] - [0, 1, 1, 2], [1, 1, 1, 2]
    Another example:
    if the src_tokens are labeled with NEW BIO
    the labels of i-th tokens in tgt_tokens would be labels[sent_alignment[i]]
    where sent_alignment[i] is the id in src_tokens
    Special Case: the tokens in tgt_tokens are subwords of those in src_tokens,
        src_tokens = ["sos", "I'll", "have", "moon-cakes", "and", "unk", ".", "eos"]  (len: 8)
        tgt_tokens = ["<sos>", "I", "'", "ll", "have", "moon", "-", "cakes", "and", "<unk>", ".", "<eos>"] (len: 12)
        vocab_match={"sos": "<sos>", "eos": "<eos>", "unk": "<unk>"}
        start_alignment = end_alignment = [0, 1, 1, 1, 2, 3, 3, 3, 4, 5, 6, 7]
    General Case: when UNK can be different
        src_tokens = ["sos", "dr", "one", "is", "a", "unk", ".", "eos"]  (len: 8)
        tgt_tokens = ["<sos>", "d", "rone is", "a", "self", "-", "<unk>", ".", "<eos>"] (len: 12)
    """
    # remove all the whitespaces in src_tokens and tgt_tokens
    def rm_white(tokens):
        return [tok.replace(" ", "") for tok in tokens]

    def replace_spe(tokens):
        tmp_tokens = deepcopy(tokens)
        for i, tok in enumerate(tokens):
            if tok in vocab_match.keys():
                tmp_tokens[i] = vocab_match[tok]
        return tmp_tokens

    def get_str(tokens):
        toks = rm_white(tokens)
        if vocab_match:
            toks = replace_spe(toks)
        str = "".join(toks)
        lens = [len(tok) for tok in toks]
        return toks, str, lens

    _, str_src, lens_src = get_str(src_tokens)
    toks_tgt = rm_white(tgt_tokens)
    str_tgt = "".join(toks_tgt)
    cum_lens_src = np.cumsum(lens_src)
    assert str_src == str_tgt  # todo: consider UNK later
    start_alignment = []
    end_alignment = []
    ptr_src, idx = 0, 0
    for i, tok_tgt in enumerate(toks_tgt):
        # perform alignment
        tmp_str = str_src[ptr_src:]
        prefix = longestCommonPrefix([tok_tgt, tmp_str])
        next_ptr_src = ptr_src + len(prefix)
        while cum_lens_src[idx] <= ptr_src:
            idx += 1
        start_alignment.append(idx)
        while cum_lens_src[idx] < next_ptr_src:
            idx += 1
        end_alignment.append(idx)
        ptr_src = next_ptr_src
    return start_alignment, end_alignment



if __name__ == '__main__':
    src_tokens = ["sos", "I'll", "have", "moon-cakes", "and", "unk", ".", "eos"]
    tgt_tokens = ["<sos>", "I", "'", "ll", "have", "moon", "-", "cakes", "and", "<unk>", ".", "<eos>"]
    # src_tokens = ["dr", "one", "is", "a"]
    # tgt_tokens = ["d", "rone is", "a"]
    print(align_bpe(src_tokens, tgt_tokens,  vocab_match={"sos": "<sos>", "eos": "<eos>", "unk": "<unk>"}))
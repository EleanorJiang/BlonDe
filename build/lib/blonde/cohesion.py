"""
TF-IDF based Repetition: enhanced RC with TF-IDF for fluency implementation.
"""
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from collections import Counter
import json
import os

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def cohesion_tf(doc, word_freq_file, weight_for_oov=300000, exclu_stop=True, norm=True):
    '''
    'ref_sentences' is a 'list' of strings.
    'sys_doc' is a 'doc' after processing a list of strings: 'doc = process_corpus(sys_sentences)[0]'.
    'word_frequency_file' is a 'json' file containing word frequency table.
    'weight_for_oov' is the unormalized weight for out of vocabulary.
    'exclu_stop' is 'True' when the stop words is excluded.
    'norm' is 'True' when normalization is conducted.
    '''
    lemmatizer = WordNetLemmatizer()
    json_file = open(word_freq_file)
    word_freq = json.load(json_file)
    content_words = Counter()
    weights = {}
    for sent in doc:
        for i, tok in enumerate(sent['sent_tok']):
            tok = tok.lower()
            if tok in [',', '.', '!', '"', '&quot;', '"', '”', '“', '\n']:
                continue
            if exclu_stop and tok in stopwords.words('english'):
                continue
            content_words[tok] += 1
            wordnet_pos = get_wordnet_pos(sent['sent_tag'][i]) or wordnet.NOUN
            lexical_word = lemmatizer.lemmatize(tok, pos=wordnet_pos)
            if lexical_word in word_freq:
                weights[tok] = word_freq[lexical_word]
            else:
                weights[tok] = weight_for_oov
    content_num = sum(content_words.values())
    weights_sum = sum(weights.values())
    if norm:
        scores = [(weight / weights_sum) * (content_words[tok]) for tok, weight in weights.items()]
    else:
        scores = [(weight) * (content_words[tok]) for tok, weight in weights.items()]

    return sum(scores) / content_num


def cohesion(doc, word_freq_file, weight_for_oov=300000, exclu_stop=True, norm=True):
    lemmatizer = WordNetLemmatizer()
    json_file = open(word_freq_file)
    word_freq = json.load(json_file)
    content_words = Counter()
    weights = {}
    for sent in doc:
        for i, tok in enumerate(sent['sent_tok']):
            tok = tok.lower()
            if tok in [',', '.', '!', '"', '&quot;', '"', '”', '“', '\n']:
                continue
            if exclu_stop and tok in stopwords.words('english'):
                continue
            content_words[tok] += 1
            wordnet_pos = get_wordnet_pos(sent['sent_tag'][i]) or wordnet.NOUN
            lexical_word = lemmatizer.lemmatize(tok, pos=wordnet_pos)
            if lexical_word in word_freq:
                weights[tok] = word_freq[lexical_word]
            else:
                weights[tok] = weight_for_oov
    content_num = sum(content_words.values())
    weights_sum = sum(weights.values())
    if norm:
        scores = [(weight / weights_sum) * (content_words[tok] - 1) for tok, weight in weights.items()]
    else:
        scores = [(weight) * (content_words[tok] - 1) for tok, weight in weights.items()]

    return sum(scores) / content_num
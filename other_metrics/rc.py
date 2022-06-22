from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None

def lc_and_rc(hyps):
    """Calcurate Translation Error Rate
    inputwords and refwords are both list object.
    >>> hyp = ['THIS WEEK THE SAUDIS denied information published in the new york times']
    >>> '{0:.3f}'.format(rc(sys_doc]))
    '0.308'
    """
    tok_lists = [word_tokenize(hyp) for hyp in hyps]
    lemmatizer = WordNetLemmatizer()
    repetition = 0
    lexical_devices = 0
    content_num = 0
    lexical_words = Counter()
    content_words = Counter()

    for tok_list in tok_lists:
        for tok in tok_list:
            if tok in [',', '.' ,'!','"', '&quot;', '"', '”','“','\n'] or tok in stopwords.words('english'):
                continue
            content_words[tok] += 1
            lexical_word = lemmatizer.lemmatize(tok)
            raw_synsets = wn.synsets(lexical_word)
            synsets = set()
            for raw_synset in raw_synsets:
                synsets.union(set(raw_synset._lemma_names))
            isset = set(lexical_words.keys()).intersection(synsets)
            if isset:
                for key in isset:
                    lexical_words[key] += 1
            else:
                lexical_words[lexical_word] += 1
    content_num = sum(content_words.values())
    repetition = content_num - len(content_words.keys())
    lexical_devices = content_num - len(lexical_words.keys())

    return repetition/content_num , lexical_devices/content_num
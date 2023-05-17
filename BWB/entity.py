import re
import pandas as pd
import json
from BWB.BWBReader import BWB
from typing import List, Dict, Tuple
"""
Get counts for entity spans for both chinese and reference.
"""
def get_entity_dictionary(chs_mention_file, ref_mention_file, entity_list_file):
    """
    get a entity_dictionary csv, where each line is an unique entity defined by (document_id, entity_id),
    and the columns are (document_id, entity_id, zh_surface_form, en_surface_form, en_lexical_form)
    """

    def to_list(x):
        DELETE_WORDS = ['自己', '女人', '人', '整个人', '妈', '妈妈', '别人', '另外一个人', '那两人', '几人', '姐姐', '弟弟',
                        '一个人', '三个人', '两个人', '十个人', '所有人', '谁', 'who', 'auntie', 'mom', '那老头', '这人',
                        '一批人', '这群人', '这些人', '他', '一人', 'one', 'man', '她', '两人' , '家', 'home', '车', 'van',
                        'person',  'man', '那个男人', '对方', 'everyone', '二人','room',
                        '阿姨', '每个女人', '这个男人', '男人', 'this man']
        REMOVE_WORDS = ['these', 'this', '几个', '一名', 'his ', 'her ']

        lst=list(set(list(x)))
        new_list = []
        for phrase in lst:
            if phrase in DELETE_WORDS or '个人' in phrase:
                continue
            for word in REMOVE_WORDS:
                phrase = phrase.replace(word, "")
            phrase = " ".join([token.strip() for token in phrase.split()])
            new_list.append(phrase)
        return list(set(new_list))


    def get_entity_df(csv_file):
        df = pd.read_csv(csv_file, sep='\t')
        df = df[df["is_pronoun"] == False]
        df['no_space_form'] = df['lexical_form'].apply(lambda x: "".join(x[2:-2].split("', '"))).str.lower()
        df['surface_form'] = df['surface_form'].apply(lambda x: " ".join(x[2:-2].split("', '"))).str.lower()
        df['lexical_form'] = df['lexical_form'].apply(lambda x: " ".join(x[2:-2].split("', '"))).str.lower()
        df = df.groupby(['document_id', 'entity_id', 'lang', 'entity_type', 'term_type']).agg({'surface_form': to_list,
                                                                                               'lexical_form': to_list,
                                                                                               'no_space_form': to_list})
        df.reset_index()
        return df
    chs_df = get_entity_df(chs_mention_file)
    ref_df = get_entity_df(ref_mention_file)
    df = ref_df.merge(chs_df, how='inner', on=['document_id', 'entity_id', 'entity_type', 'term_type'])
    df.rename(columns={'surface_form_x': 'surface_form_en', 'lexical_form_x': 'lexical_form_en',
                       'no_space_form_y': 'surface_form_zh'}, inplace=True)
    df = df.drop(columns=['no_space_form_x', 'lexical_form_y', 'surface_form_y', 'lexical_form_en'])
    df = df[(df['surface_form_zh'].str.len() != 0) & (df['surface_form_en'].str.len() != 0)]
    df.to_csv(entity_list_file, sep='\t')


def resolve_repetition(span_lst):
    """
    if span_lst= ['song yuanxi', 'yuanxi']
    return ['yuanxi']
    """
    sub_span_lst = []
    for span in span_lst:
        tmp_sub_span_lst = sub_span_lst
        sub_span_lst = []
        flg = False
        for prev_span in tmp_sub_span_lst:
            if span in prev_span:
                sub_span_lst.append(span)
                flg = True
            else:
                sub_span_lst.append(prev_span)
        if flg is False:
            sub_span_lst.append(span)
    return sub_span_lst


def count_entity_spans(sent_text: str, span_lst: List[str], lang: str) -> int:
    """
    checkpoints is the list of annotated spans of a certain category, e.g. Ambiguity or Ellipsis
    """
    count = 0
    span_lst = resolve_repetition(span_lst)
    for span in span_lst:
        if lang== 'zh':
            pattern = re.compile(span)
        else:
            pattern = re.compile(r"\b{}\b".format(span), re.IGNORECASE)
        lst = re.findall(pattern, sent_text)
        count += len(lst)
    return count


def get_entity_counts(dir_path):
    """
    Count the frequency of the entities listed in `entity_list_file` in
    the Chinese and English documents in the BWB test, respectively.
    For further entity recall computation in `get_bwb_scores.py`.

    :return entity_counts_list: a list of 80 dicts corresponding to 80 documents.
    The keys of each document dict are the count categories we want to calculate, i.e.
        ["PER", "FAC", "GPE", "LOC", "VEH", "ORG", "N", "T"] 8 in total
    Each value is a dict corresponding to two (recall, precision, f1) score-tuples:
    'en_entity_list': List[List[str]]. A two-dimensional array, the first dimension is the number of entities,
                        the second dimension is how many surface forms each entity has.
    ref_entity_counts：List[List[int]]. One-dimensional array,
                    the first dimension is the number of sentences in the reference document,
                    the second dimension is the number of entities,
                    and the value is how many times the entity appears in the reference sentence.
    chs_entity_counts: List[List[int]]. Same as ref_entity_counts,
                    but the value is how many times the entity appears in the Chinese sentence.
    """
    cache_file = f"{dir_path}/sent.cache"
    entity_list_file = f"{dir_path}/entity_dict.csv"
    df = pd.read_csv(entity_list_file, sep='\t')
    df['surface_form_en'] = df['surface_form_en'].apply(lambda x: x[2:-2].split("', '"))
    df['surface_form_zh'] = df['surface_form_zh'].apply(lambda x: x[2:-2].split("', '"))
    def _get_entity_list_from_df(df, lang="en"):
        tmp_df = df.groupby('document_id')[f'surface_form_{lang}'].apply(list).groupby(level=0).apply(list)
        tmp_df = {k: v[0] for k, v in dict(tmp_df).items()}
        return tmp_df
    en_entity_list_dict, zh_entity_list_dict  = {}, {}
    for key in ["PER", "FAC", "GPE", "LOC", "VEH", "ORG"]:
        sub_df = df[df['entity_type'] == key]
        # en_entity_list_dict: Dict[str, List[List[int]]]. 80 items in total.
        # en_entity_list_dict['PER']['Book0-0'] = [['qiao lian'], ['wechat'],['paparazzi', 'reporter', 'paparazzi']]
        en_entity_list_dict[key] = _get_entity_list_from_df(sub_df, lang="en")
        zh_entity_list_dict[key] = _get_entity_list_from_df(sub_df, lang="zh")
    for key in ["T", "N"]:
        sub_df = df[df['term_type'] == key]
        en_entity_list_dict[key] = _get_entity_list_from_df(sub_df, lang="en")
        zh_entity_list_dict[key] = _get_entity_list_from_df(sub_df, lang="zh")

    entity_counts = {}
    for key in ["PER", "FAC", "GPE", "LOC", "VEH", "ORG", "T", "N"]:
        entity_counts[key] = {}
        document_id = list(en_entity_list_dict[key].keys())[0]
        entity_counts[key][document_id] = {"en_entity_list": en_entity_list_dict[key][document_id],
                                           "ref_entity_counts": [], "chs_entity_counts": []}
        bwb_reader = BWB()
        for (zh_sent, en_sent) in bwb_reader.dataset_iterator_from_cache(cache_file=cache_file, dir_path=dir_path):
            if document_id != en_sent.document_id:
                document_id = en_sent.document_id
                if document_id not in en_entity_list_dict[key]:
                    continue
                entity_counts[key][document_id] = {"en_entity_list": en_entity_list_dict[key][document_id],
                                                   "ref_entity_counts": [], "chs_entity_counts": []}
            if document_id not in en_entity_list_dict[key]:
                continue
            ref_sent_count, chs_sent_count = [], []
            for entity_id in range(len(entity_counts[key][document_id]["en_entity_list"])):
                entity = entity_counts[key][document_id]["en_entity_list"][entity_id]
                zh_entity = zh_entity_list_dict[key][document_id][entity_id]
                ref_sent_count.append(count_entity_spans(" ".join(en_sent.words), entity, lang="en"))
                chs_sent_count.append(count_entity_spans("".join(zh_sent.words), zh_entity, lang="zh"))
            entity_counts[key][document_id]["ref_entity_counts"].append(ref_sent_count)
            entity_counts[key][document_id]["chs_entity_counts"].append(chs_sent_count)
    with open(f"{dir_path}/entity_counts.json", "w") as f:
        json.dump(entity_counts, f, indent=2)
    return entity_counts

# BWB Annotation
The BWB test set is an annotated dataset of 80 documents of Chinese-English parallel 
web novels.

# What do we annotate? 
 
Entity, terminology, coreference, quotation. 

## Entity Annotations
 
The entity annotation layer of BWB covers six of the ACE 2005 categories in text: 
 
* People (`PER`): *Qiao Lian*, *her daughter* 
* Facilities (`FAC`): *the house*, *the kitchen* 
* Geo-political entities (`GPE`): *London*, *the village* 
* Locations (`LOC`): *the forest*, *the river*,  
* Vehicles (`VEH`): *the ship*, *the car* 
* Organizations (`ORG`): *the army*, *the Church*, *the Shen fan clubs(凉粉群)*, *Weibo(微博)*

We assign a unique `entity_id` to each distinct entity.


## Terminology Annotations
Other than the six ACE entity categories outlined above, we also annotate terminology.

The terminology annotation layer of BWB is a binary classification of whether 
a certain span is a terminology (`T`) or not (`N`).

Generally speaking, terminology refers to proper noun phrases (Qiao Lian).

### Examples of the entity and terminology layers
```angular2html
<PER, T, 1>{Qiao Lian}  
<PER, T, 2>{Shen Liangchuan} 
<FAC, N, 3>{the house} 
<GPE, N, 4>{Europe} 
<LOC, N, 5>{the sea} 
<VEH, N, 6>{the ship} 
<ORG, T, 7>{凉粉群}
```

## Coreference Annotations
The coreference annotation layer of BWB covers not only the entity tagging above:
proper noun phrases (Qiao Lian) and common noun phrases (the kitchen), 
but also personal pronouns (he). 
We tag the omitted pronouns with `O` and other pronouns with `P`. An omitted refers to
the pronoun which is omitted in the other language. For example, 

ZH:
```angular2html
<PER, T, 1>{乔恋}攥紧了拳头，<P, 1>{她}垂下了头。
```
EN:
```angular2html
<PER, T, 1>{Qiao Lian} clenched <O, 1>{her} fists and lowered <P, 1>{her}head.
```

In this example, because the first `her` is omitted in the corresponding Chinese translation, it is
marked as `<O, 1>`, where `1` is the `entity_id` of `Qiao Lian`.



The following personal pronouns are considered:
```angular2html
“masculine": ["he", "his", "him", "He", "His", "Him", "himself", "Himself"], 
"feminine": ["she", "her", "hers", "She", "Her", "Hers", "herself", "Herself"], 
"neuter": ["it", "its", "It", "Its", "itself", "Itself"], 
"epicene": ["they", "their", "them", "They", "Their", "Them", "themselves", "Themselves"] 
```


## Quotation Annotations
The quotation layer identifies all instances of direct speech in the text, 
attributed to its speaker.

For example, `‘ Oh dear ! Oh dear ! I shall be late ! ’` is annotated as
`<Q, 2> ‘ Oh dear ! Oh dear ! I shall be late ! ’  <\Q>`, 
where `2` is the `entity_id` of the speaker.

# Annotation Principles:
### Maximal Span
Following OntoNotes, we mark the maximal extent of a span, as in the following: 
 
```angular2html
[The boy who painted the fence and ate lunch] ran away.
``` 

### Rather Lack Than Abuse
When in doubt, do not mark any coreference. 

# The Data Format of BWB
ZH: 
```angular2html
<PER, T, 1>{乔恋}攥紧了拳头，垂下了头。 
其实<P, 2>{他}说得对。 
自己就是一个蠢货，竟然会相信了网络上的爱情。 
<P, 1>{她}勾起了嘴唇，深呼吸一下，正打算将手机放下，<ORG, T, 3>{微信}上却被炸开了锅。 
<P, 1>{她}点进去，发现是<ORG, T, 4>{凉粉群}，所有人都在@<P, 1>{她}。 
<Q, 1>【<PER, P, 1>{乔恋}：怎么了？<\Q> 
<Q, 5>【<PER, 5>{川流不息}：<PER, T, 1>乔恋}，快看<ORG, T, 6>{微博}头条！ <ORG, T, 6>{微博}头条？<\Q>
```

EN:
```angular2html
<PER, T, 1>{Qiao Lian} clenched <O, 1>{her} fists and lowered <O, 1>{her}head. 
Actually, <P,2>{he} was right. 
<O, 1>{She} was indeed an idiot, as only an idiot would believe that they could find true love online. 
<P, 1>{She} curled <P, 1>her} lips and took a deep breath. Just when <P, 1>{she} was about to put down <P, 1>{her} cell phone, a barrage of posts bombarded <P, 1>{her} <ORG, T, 3>{WeChat} account. 
<P,1>{She} logged into <O, 1>{her}account and saw that a large number of fans in the <ORG, T, 4>{<PER, P, 2>{Shen} Liangchuan fan group} had tagged <P, 1>{her}. 
<Q, 1> [<PER, T, 1>{Qiao Lian}: What happened?] <\Q> 
<Q, 5> [<PER, 5>{Chuan Forever}: <PER, T, 1>{Qiao Lian}, look at the headlines on <ORG, T, 6>{Weibo}, quickly!] <\Q>
```

# Processing BWB
We provide a dataset reader `BWB`
in  `BWB.py`, which provide an iterator over the entire dataset,
yielding all sentences processed. The processed sentences are `BWBSentence` objects.
`BWBSentence` is a class representing the annotations available for a single formatted sentence.
The parameters are:

|  parameter    |  Description  |    
|-----------|-------------|
| ``document_id``     | `str`. This is a variation on the document filename.    |
| ``sentence_id``     | `int`. The integer ID of the sentence within a document.    |
| ``lang``     | `str`. The language of this sentence.    |
| ``line`` | `str`. The original annotation line. |
| ``words``     | `List[str]`. This is the tokens as segmented/tokenized in BWB.    |
| ``pronouns``     | `Dict[str, Span]`. Pronoun type (`P` or `O`) -> Span.    |
| ``entities`` | `Dict[int, Tuple[str, str]]`. Entity id -> (Entity type, Terminology). |
| ``clusters`` | `Dict[int, List[Span]]`. A dict of coreference clusters, where the keys are entity ids, and the values are spans. A span is a tuple of int, i.e. (start_index, end_index).|
| ``quotes`` | `List[Tuple[int, Span]]`. A list of (entity_id, quote). Default: `[]`. |
| ``pos_tags`` | `List[str]`. This is the Penn-Treebank-style part of speech. Default: `None.`|
 
**We have processed the annotations to be in the `csv` format:**

- `test.chs.csv` and `test.ref.csv`
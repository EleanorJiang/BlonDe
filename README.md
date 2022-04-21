# BlonD Scores Package

- Link to paper: [here](https://openreview.net/forum?id=Bl-gR45hkZc).

**If you use this package in your research, please cite:**
```
Accepted to NAACL2022
```


## What in this package:
- BlonD: combine dBlonD with sentence-level measurement
  - dBlonD: measure the discourse phonomena with reference
- BlonD_plus: take human annotation (annotated ambiguous/ommited phrases and manually-annotated NER) into consideration
- Cohesion Score: measure document-level fluency without reference

## Usage
See example.py
### Load Package
 ```
        from blond.BlonD import BLOND
   ```
### For a single document:
 ```
        from blond.BlonD import BLOND
        blond = BLOND()
        score = blond.corpus_score([sys_doc], [[ref_doc_1], [ref_doc_2], ...])
   ```
          'sys_doc', 'ref_doc': List[str]
### For a corpus:
 ```
        score = blond.corpus_score(sys_corpus, [ref_corpus_1, ref_corpus_2, ...])
   ```
          'sys_corpus', 'ref_corpus': List[List[str]]

### For multiple systems & statistical testing:
 ```
        blond_plus = BLOND(references=[ref_corpus]) # for faster recomputation
        score = blond.corpus_score(sys_corpus)
   ```

### BlonD+:
 ```
        blond_plus = BLOND(average_method='geometric',
                           references=[ref_corpus],
                           annotation=an_corpus,
                           ner_refined=ner_corpus
                           )
        score = blond_plus.corpus_score(sys_corpus)
   ```

### Adjust parameters:
 ```
        blond_plus = BLOND(weights: Dict[str, Union[Tuple[float], float]]=None,
                          weight_normalize: bool = False,
                          average_method: str = 'geometric',
                          categories: dict = CATEGORIES,
                          plus_categories=None,  # ("ambiguity", "ellipsis")
                          plus_weights=(1, 1),
                          lowercase: bool = False,
                          smooth_method: str = 'exp',
                          smooth_value: Optional[float] = None,
                          effective_order: bool = False,
                          references: Optional[Sequence[Sequence[Sequence[str]]]] = None,
                          annotation: Sequence[Sequence[str]] = None,
                          ner_refined: Sequence[Sequence[str]] = None)
                           )
        score = blond_plus.corpus_score(sys_corpus, [ref_corpus_1, ref_corpus_2, ...])
   ```

### Cohesion Score:
 ```
    cohesion_score = cohesion(sys_doc,word_frequency_file, weight_for_oov=300000, exclu_stop=True, norm=True)
   ```
    'word_frequency_file' is a 'json' file containing word frequency table.
    'weight_for_oov' is the unormalized weight for out of vocabulary.
    'exclu_stop' is 'True' when the stop words is excluded.
    'norm' is 'True' when normalization is conducted.


## Requirements
- spacy: for BlonD

     ```
      pip install spacy
      python -m spacy download en_core_web_sm
       ```
- nltk.stopwords, nltk.wordnet: for cohesion score

  ```
    pip install nltk
  ```
  then run the Python interpreter and type the commands:
  ```
    import nltk
    nltk.download()
  ```
  A new window should open, showing the NLTK Downloader. Choose 'wordnet'.

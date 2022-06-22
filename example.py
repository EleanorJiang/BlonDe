from blonde.BlonDe import BLOND
from sacrebleu.metrics import BLEU



if __name__ == "__main__":
    ref = ["Qiao looked at the photo and recalled twenty years ago.",
           "This bearded man was her newlywed husband,",
           "yet this was the first time they were meeting with each other.",
           "So Qiao's heart jolted as soon as she saw him, and she quickly stood up.",
           "Joe’s heart is squeaky as soon as he saw him, and she quickly stands up."
           ]
    sys_1 = ["Qiao looked at the photo and recalled twenty years ago.",
           "This bearded man is her newlywed husband.",
           "This is the first time they meet with each other.",
             "Joe’s heart is squeaky and he quickly stands up."]
    sys_2 = ["Qiao looked at the photo and recalled the past twenty years ago.",
           "The man with the beard was her newly-wed husband.",
           "However, that was the first time they met.",
           "So as soon as Qiao saw him, her heart became squeaky, and she swiftly stood up."]
    blond = BLOND()
    bleu = BLEU()
    for sys in [sys_1, sys_2]:
        score = blond.corpus_score([sys], [[ref]])
        print(score)
        bleu_score = bleu.corpus_score(sys, [ref])
        print(bleu_score)
import nltk
import pandas as pd
import numpy as np
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

st = StanfordNERTagger('../stanford_ner/classifiers/english.all.3class.distsim.crf.ser.gz',
					   '../stanford_ner/stanford-ner.jar',
					   encoding='utf-8')

def tagText(comment):
    tokenized = word_tokenize(comment)
    classified = st.tag(tokenized)
    return classified

with open("./true_positives.txt") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content] 

samp = content[:5]

for i in samp:
    res = tagText(i)
    print(res)
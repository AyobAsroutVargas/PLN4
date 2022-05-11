import sys
import pandas as pd
from collections import Counter
from pathlib import Path
import numpy as np
import re
import pandas as pd
from math import log
import spacy
import openpyxl

nlp = spacy.load('en_core_web_sm')

spaces_regex = '\s+'
hashtags_regex = '#\S+'
mentions_regex = '@\S+'
unicode_regex = '[^\x00-\x7F]+'
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def preprocess_tweet(text):
  tokens = []
  doc = nlp(text)
  for token in doc:
      word = token.text
      mentions = re.findall(mentions_regex, word)
      hashtags = re.findall(hashtags_regex, word)
      url = re.findall(url_regex, word)
      unicode = re.findall(unicode_regex, word)
      spaces = re.findall(spaces_regex, word)
      if not token.is_stop and not token.is_punct and not word.isdigit() and not len(word) < 1 and not word == '' and not spaces and not mentions and not hashtags and not unicode and not url:
        if(word.islower()):
            tokens.append(word)
            #tokens = np.append(tokens, word)
        else:
            tokens.append(word.lower())
            #tokens = np.append(tokens, word.lower())
  #np_tokens = np.array(tokens)
  return tokens

data = pd.read_excel("COV_train_test.xlsx", header=None, engine="openpyxl")
data.columns = ["Tweet"]

test = preprocess_tweet(data["Tweet"][103])
print(test)
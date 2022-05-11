import spacy
import openpyxl
from pathlib import Path
import numpy as np
import re

nlp = spacy.load('en_core_web_sm')

spaces_regex = '\s+'
hashtags_regex = '#\S+'
mentions_regex = '@\S+'
unicode_regex = '[^\x00-\x7F]+'
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

#read text present in xlsx file. returns an array with all the tweets.
def read_file(file_name):
  file_path = Path(file_name)
  workbook = openpyxl.load_workbook(filename=file_path, read_only=True)
  sheet = workbook.active
  text = []
  for row in sheet.iter_rows(max_col=1):
      for cell in row:
          text.append(cell.value)
  return text

#Given an array containing text returns a sorted numpy array with every word
#present in the texts of the given array. Words will be present once despite there
# being multiple instances of said word.
#Punctuators, english stopwords, urls, hastags, numbers, unicode characters and spaces will be filtered.
#The stopwords used are the ones defined in spacy library.
def get_vocabulary_from_text(text):
  tokens = []
  for row in text:
    doc = nlp(row)
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
        else:
          tokens.append(word.lower())

  np_tokens = np.array(tokens)
  uniqueTokens = np.unique(np_tokens)
  uniqueTokens.sort()
  return uniqueTokens

def write_vocabulary_to_file(filename, uniqueTokens):
  file = open(filename, "w+", encoding="utf-8")
  file.write("Number of tokens: " + str(uniqueTokens.size) + "\n")
  for word in uniqueTokens:
    file.write(word + "\n")
  file.close()

text = read_file('COV_train.xlsx')
uniqueTokens = get_vocabulary_from_text(text)
write_vocabulary_to_file("vocabulario.txt", uniqueTokens=uniqueTokens)
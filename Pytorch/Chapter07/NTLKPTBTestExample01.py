import nltk
from collections import Counter

# Download PTB if not already present
nltk.download('ptb')

from nltk.corpus import treebank as ptb

print("PTB tokens : ", ptb.words())
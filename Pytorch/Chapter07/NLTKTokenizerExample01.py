from nltk.tokenize import TreebankWordTokenizer

print("> TreebankWordTokenizer Example")
tokenizer = TreebankWordTokenizer()
text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(text))

print("> WordPunctTokenizer Example")
from nltk.tokenize import WordPunctTokenizer
text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
print(WordPunctTokenizer().tokenize(text))

import nltk
nltk.download("punkt_tab")
print("> Word Tokenize Example")
from nltk.tokenize import word_tokenize
text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
print(word_tokenize(text))
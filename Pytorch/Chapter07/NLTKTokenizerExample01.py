import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize


def run_example():
    text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."
    print("> Original Text:", text)

    tokenizer = TreebankWordTokenizer()
    tokens = tokenizer.tokenize(text)
    print("> Tokens using TreebankWordTokenizer:", tokens)

    word_punct_tokenizer = WordPunctTokenizer()
    word_punct_tokens = word_punct_tokenizer.tokenize(text)
    print("> Tokens using WordPunctTokenizer:", word_punct_tokens)

    nltk_tokens = word_tokenize(text)
    print("> Tokens using word_tokenize:", nltk_tokens)


if __name__ == "__main__":
    print("> NLTK Data Download")
    nltk.download("punkt_tab")

    print("> NLTK Tokenizer Example.")
    run_example()

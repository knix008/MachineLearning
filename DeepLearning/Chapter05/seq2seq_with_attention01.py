import nltk
import numpy as np
import re
import shutil
import tensorflow as tf
import os
import unicodedata
import zipfile

def download_and_read(url, num_sent_pairs=30000):
    local_file = url.split('/')[-1]
    if not os.path.exists(local_file):
        os.system("wget -O {:s} {:s}".format(local_file, url))
        with zipfile.ZipFile(local_file, "r") as zip_ref:
            zip_ref.extractall(".")
    local_file = os.path.join(".", "fra.txt")
    en_sents, fr_sents_in, fr_sents_out = [], [], []
    with open(local_file, "r") as fin:
        for i, line in enumerate(fin):
            en_sent, fr_sent = line.strip().split('\t')
            en_sent = [w for w in preprocess_sentence(en_sent).split()]
            fr_sent = preprocess_sentence(fr_sent)
            fr_sent_in = [w for w in ("BOS " + fr_sent).split()]
            fr_sent_out = [w for w in (fr_sent + " EOS").split()]
            en_sents.append(en_sent)
            fr_sents_in.append(fr_sent_in)
            fr_sents_out.append(fr_sent_out)
            if i >= num_sent_pairs - 1:
                break
    return en_sents, fr_sents_in, fr_sents_out

# data preparation
download_url = "http://www.manythings.org/anki/fra-eng.zip"
sents_en, sents_fr_in, sents_fr_out = download_and_read(download_url)
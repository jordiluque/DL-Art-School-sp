# 
# Train a tokenizer

import re
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from nemo_text_processing.text_normalization.normalize import Normalizer

text_normalizer = Normalizer(input_case="cased", lang="es")


def text_cleaners(text):
  text = text_normalizer.normalize(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  text = text.replace('"', '')
  text = text.replace('¿', '')
  text = text.replace('¡', '')
  return text


def remove_extraneous_punctuation(word):
    replacement_punctuation = {
        '{': '(', '}': ')',
        '[': '(', ']': ')',
        '`': '\'', '—': '-',
        '—': '-', '`': '\'',
        'ʼ': '\''
    }
    replace = re.compile("|".join([re.escape(k) for k in sorted(replacement_punctuation, key=len, reverse=True)]), flags=re.DOTALL)
    word = replace.sub(lambda x: replacement_punctuation[x.group(0)], word)
    extraneous = re.compile(r'^[@#%_=\$\^&\*\+\\]$')
    word = extraneous.sub('', word)
    return word

with open('./datasets/voxpopuli_LJ/transcriptions.txt', 'r', encoding='utf-8') as at:
    ttsd = at.readlines()
    allowed_characters_re = re.compile(r'^[a-zñáéíóúüï!:;"/, \-\(\)\.\'\?ʼ]+$')
    
    def preprocess_word(word, report=False):
        word = text_cleaners(word)
        word = remove_extraneous_punctuation(word)
        if not bool(allowed_characters_re.match(word)):
            if report and word:
                print(f"REPORTING: '{word}'")
            return ''
        return word

    def batch_iterator(batch_size=1000):
        print("Processing ASR texts.")
        for i in range(0, len(ttsd), batch_size):
            yield [preprocess_word(t, True) for t in ttsd[i:i+batch_size]]

    trainer = BpeTrainer(special_tokens=['[STOP]', '[UNK]', '[SPACE]'], vocab_size=255)
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(batch_iterator(), trainer, length=len(ttsd))
    tokenizer.save('./spanish_language_tokenizer.json')

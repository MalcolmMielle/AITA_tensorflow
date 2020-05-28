import pandas
import re
import numpy as np
from nltk import tokenize
from nltk.corpus import stopwords
import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


class AITA_Generator(tf.keras.utils.Sequence):

    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.data) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.data[idx * self.batch_size: (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx+1) * self.batch_size]

        return batch_x, batch_y


def generating_data(file_path: str = "../data/aita_clean_light.csv"):
    df = pandas.read_csv(file_path)
    # Make a composite text field of the title and body of each post
    df['text'] = df["title"] + df["body"].fillna("")
    # temp = embedding_f(df['text'].to_list()).numpy()
    # df['emb'] = temp.tolist()
    # pandas.options.display.max_colwidth = 200
    # print(df['emb'])
    return df


def format_ponctuation(text: str):
    text = re.sub(r'(\?+)', r'? ', text)
    text = re.sub(r'(!+)', r'! ', text)
    return re.sub(r'(\n+)', r'. ', text)


def split_sentences(text):
    ar = np.array(tokenize.sent_tokenize(text))
    # tf_ar = tf.constant(ar)
    # print(tf_ar)
    return ar


def remove_stopwords(text: str, stopword: set):
    for word in stopword:
        token = ' ' + word + ' '
        text = text.replace(token, ' ')
        text = text.replace(' ', ' ')
    return text


# turn a doc into clean tokens
def clean_doc(text: str) -> str:
    text_lowercase = text.lower()
    # split into tokens by white space
    tokens = text_lowercase.split()
    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def tokenize_encoding(data_pandas_serie, max_length: int, vocab_size: int,
                      oov_tok: str, padding_type, trunc_type) -> list:
    dataset = data_pandas_serie.to_numpy()
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(dataset)
    # word_index = tokenizer.word_index
    # dict(list(word_index.items())[0:10])
    tok = tokenizer.texts_to_sequences(dataset)
    return pad_sequences(tok, maxlen=max_length, padding=padding_type, truncating=trunc_type), tokenizer

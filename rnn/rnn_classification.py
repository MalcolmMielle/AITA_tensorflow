# https://towardsdatascience.com/multi-class-text-classification-with-lstm-using-tensorflow-2-0-d88627c10a35
import tensorflow as tf
import numpy as np
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk
import get_data

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))

print(tf.__version__)

vocab_size = 5000
embedding_dim = 64
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

data = get_data.generating_data()
data['no_newline'] = data['text'].apply(get_data.format_ponctuation)
data['stopwords'] = data['no_newline'].apply(get_data.remove_stopwords, stopword=STOPWORDS)
data['split_text'] = data['stopwords'].apply(get_data.split_sentences)

print(data['stopwords'])

train, validate, test = np.split(data.sample(frac=1), [int(.6 * len(data)), int(.8 * len(data))])
# print(train_sequences)


# define the function#
def find_max_list(list):
    list_len = [len(i) for i in list]
    return max(list_len)


def create_dataset_token(data_pandas, max_length: int) -> list:
    dataset = data_pandas.stopwords.to_numpy()
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(dataset)
    # word_index = tokenizer.word_index
    # dict(list(word_index.items())[0:10])
    tok = tokenizer.texts_to_sequences(dataset)
    return pad_sequences(tok, maxlen=max_length, padding=padding_type, truncating=trunc_type), tokenizer


def smote(features, labels):
    # Set up some sklearn objects that are going to be in the pipeline
    # SMOTE for class balancing via oversampling the minority class
    smt = SMOTE(random_state=12)
    return smt.fit_resample(features, labels)


max_length = 200
train_sequences, tokenizer_train = create_dataset_token(train, max_length=max_length)
train_label = train.is_asshole.to_numpy()

unique, counts = np.unique(train_label, return_counts=True)
dict(zip(unique, counts/len(train_label)))

print(train_sequences[0:2])
print(train_label[0:3])

# Check dataset
reverse_word_index = dict([(value, key) for (key, value) in tokenizer_train.word_index.items()])


def decode_article(text):
    return ' '.join([reverse_word_index.get(i, 'PAD') for i in text])


print(train_sequences[1])
print(decode_article(train_sequences[1]))
print('---')
print(train_sequences[10])

validate_sequences, _ = create_dataset_token(validate, max_length=max_length)
validate_label = validate.is_asshole.to_numpy()

# RIGHT SMOTE ADS PADDING IN THE MIDDLE OF THE SET . NOT GOOD I GUESS
use_smote = True
if use_smote:
    train_sequences, train_label = smote(train_sequences, train_label)
    print(decode_article(train_sequences[-1]))
    validate_sequences, validate_label = smote(validate_sequences, validate_label)

print(len(train_sequences))

unique_smote, counts_smote = np.unique(train_label, return_counts=True)
dict(zip(unique_smote, counts_smote/len(train_label)))
# unique_smote_v, counts_smote_v = np.unique(validate_labels_smote, return_counts=True)
# dict(zip(unique_smote_v, counts_smote_v/len(validate_labels_smote)))


# train_sequences_pad = pad_sequences(train_smote, maxlen=max_length, padding=padding_type, truncating=trunc_type)
# validate_sequences_pad = pad_sequences(validate_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(validate_sequences))
print(validate_sequences.shape)

# maximum = find_max_list(train_sequences)
# print(maximum)
# print(len(train_sequences[0]))
# print(len(train_padded[0]))
#
# print(len(train_sequences[1]))
# print(len(train_padded[1]))
#
# print(len(train_sequences[10]))
# print(len(train_padded[10]))


def model_f():
    model = tf.keras.Sequential([
        # Add an Embedding layer expecting input vocab of size 5000, and output embedding dimension of size 64 we set at the top
        tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # use ReLU in place of tanh function since they are very good alternatives of each other.
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Add a Dense layer with 6 units and softmax activation.
        # When we have multiple outputs, softmax convert outputs layers into a probability distribution.
        tf.keras.layers.Dense(1)
        ])
    model.summary()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = model_f()


num_epochs = 3
history = model.fit(train_sequences, train_label, epochs=num_epochs,
                    validation_data=(validate_sequences, validate_label), verbose=1,
                    validation_steps=30)


import matplotlib.pyplot as plt
%matplotlib inline


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()



plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

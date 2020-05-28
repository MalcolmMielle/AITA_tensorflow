# %% codecell
import pandas
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import tensorflow as tf

# %% [markdown]
# Loading the data

# %% codecell
df = pandas.read_csv("../data/aita_clean.csv")
print(df)
counts = df['verdict'].value_counts()
verdict = counts.index.to_list()
freq = counts.to_list()
freq_df = pandas.DataFrame(list(zip(verdict, freq)), columns=['Verdict', 'Frequency'])
freq_df['Verdict'] = freq_df['Verdict'].str.title()
print("Verdict")
print(freq_df.head())


# Make a composite text field of the title and body of each post
df['text'] = df["title"] + df["body"].fillna("")
print(df)

# %% [markdown]
# Create the feature extraction object

# %% codecell
vocab_size = 1000
embedding_dim = 64
# Set up some sklearn objects that are going to be in the pipeline
# SMOTE for class balancing via oversampling the minority class
smt = SMOTE(random_state=12)
# TF-IDF Vectorizer: https://www.quora.com/How-does-TfidfVectorizer-work-in-laymans-terms?share=1
# Define the importance of words in the corpus depending on frequency AND "uniqueness"
tfidf = TfidfVectorizer(sublinear_tf=True,
                        max_features=vocab_size,
                        min_df=5, norm='l2',
                        encoding='latin-1',
                        ngram_range=(1, 1),
                        stop_words='english')

# %% [markdown]
# Separate the data in train validate and test

# %% codecell
train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

features = tfidf.fit_transform(train.text).toarray()
# Binary classification result
labels = train.is_asshole
X_train, y_train = smt.fit_resample(features, labels)
# TF-IDF on test set
X_validate = tfidf.transform(validate.text).toarray()
y_validate = validate.is_asshole

# %% codecell
print("One examle")
print(X_train[0])

print(features[0:2])

print(len(features))

with np.printoptions(threshold=np.inf):
    print(features[0])

print(features.shape)
print(X_train.shape)

# %% [markdown]
# Build the model


# %% codecell
def model_f():
    model = tf.keras.Sequential([
        # Add an Embedding layer expecting input vocab of size 5000,
        # and output embedding dimension of size 64 we set at the top
        # tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        #    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        # use ReLU in place of tanh function since they are very good alternatives of each other.
        tf.keras.layers.Dense(embedding_dim, activation='relu', input_dim=vocab_size),
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

# %% codecell
num_epochs = 50
history = model.fit(X_train, y_train, epochs=num_epochs,
                    validation_data=(X_validate, y_validate), verbose=1,
                    validation_steps=30)

# %% codecell
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

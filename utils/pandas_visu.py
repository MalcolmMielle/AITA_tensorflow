import pandas
from utils import metric
from utils import get_data
import matplotlib.pyplot as plt


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

# %% codecell
num_classes = metric.get_num_classes(df.is_asshole.to_numpy())
num_word_sample = metric.get_num_words_per_sample(df.text.to_numpy())

# %% codecell
print("Num classes:", num_classes)
print("Median number of word per sample:", num_word_sample)

metric.plot_sample_length_distribution(df.text.to_numpy())
fig = plt.gcf()
fig.savefig('length_distrib.jpg')
plt.show()
metric.plot_class_distribution(df.is_asshole.to_numpy())
fig = plt.gcf()
fig.savefig('class_distrib.jpg')

# %% codecell
number_of_samples = len(df.text)
ratio = number_of_samples / num_word_sample

print(ratio)

# %% [markdown]
#
# From our experiments, we have observed that the ratio of “number of samples” (S)
# to “number of words per sample” (W) correlates with which model performs well.
#
# When the value for this ratio is small (<1500), small multi-layer perceptrons
# that take n-grams as input (which we'll call Option A) perform better or at least
# as well as sequence models. MLPs are simple to define and understand,
# and they take much less compute time than sequence models.
# When the value for this ratio is large (>= 1500), use a sequence model (Option B).
#In the steps that follow, you can skip to the relevant subsections (labeled A or B)
# for the model type you chose based on the samples/words-per-sample ratio.


# %% codecell
df['clean'] = df['text'].apply(get_data.clean_doc)

print(df['clean'])

metric.get_last_layer_units_and_activation(num_classes)

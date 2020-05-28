import pandas
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


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

# Prepare k-splits
cv = 5
kf = KFold(n_splits=cv, shuffle=True, random_state=2)
split_gen = kf.split(df)

# Set up some sklearn objects that are going to be in the pipeline
# SMOTE for class balancing via oversampling the minority class
smt = SMOTE(random_state=12)
# TF-IDF Vectorizer: https://www.quora.com/How-does-TfidfVectorizer-work-in-laymans-terms?share=1
# Define the importance of words in the corpus depending on frequency AND "uniqueness"
tfidf = TfidfVectorizer(sublinear_tf=True,
                        max_features=10000,
                        min_df=5, norm='l2',
                        encoding='latin-1',
                        ngram_range=(1, 1),
                        stop_words='english')

# Multinomial naive bayes classifier
nb = MultinomialNB()

results = []
for i in range(0,cv):
    print("Now processing fold " + str(i+1))
    result = next(split_gen)
    # Get train and test samples
    train = df.iloc[result[0]]
    test = df.iloc[result[1]]
    # TF-IDF on training set
    features = tfidf.fit_transform(train.text).toarray()
    # Binary classification result
    labels = train.is_asshole
    # TF-IDF on test set
    X_test = tfidf.transform(test.text).toarray()
    y_test = test.is_asshole
    # Resample with smote
    X_train, y_train = smt.fit_resample(features, labels)
    # Fit the model
    nb.fit(X_train, y_train)
    yhat_score = nb.score(X_test, y_test)
    results.append(yhat_score)

print(results)

import string
import pandas as pd
import numpy as np
from nltk import pos_tag, WordNetLemmatizer, word_tokenize
from nltk.corpus import wordnet, stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import nltk
nltk.download('wordnet')


df_input = pd.read_csv('sample30.csv')

# Concatenating review title and review text which would be used for sentiment analysis
for_sentiment_analysis=df_input
for_sentiment_analysis['reviews_title_text']= df_input['reviews_title'].fillna('') +" "+ df_input['reviews_text']

# Dropping one row where user_sentiment is null
for_sentiment_analysis=for_sentiment_analysis[for_sentiment_analysis['user_sentiment'].isnull()== False]
for_sentiment_analysis.reset_index(drop=True)


# Function that returns the wordnet object value corresponding to the POS tag

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# Function for cleaning the text

def clean_text(text):
    # lower text
    text = text.lower()

    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]

    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]

    # join all
    text = " ".join(text)
    return (text)


# clean text data
for_sentiment_analysis["reviews_clean"] = for_sentiment_analysis.apply(lambda x: clean_text(x['reviews_title_text']),
                                                                       axis=1)

# Mapping positive sentiment as 1 and negative as 0

for_sentiment_analysis['Sentiment_coded'] = np.where(for_sentiment_analysis.user_sentiment == 'Positive',1,0)

from sklearn.feature_extraction.text import TfidfVectorizer

### Creating a python object of the class CountVectorizer
tfidf_counts = TfidfVectorizer(tokenizer= word_tokenize, # type of tokenization
                               stop_words=stopwords.words('english'), # List of stopwords
                               ngram_range=(1,2)) # number of n-grams

tfidf_data = tfidf_counts.fit_transform(for_sentiment_analysis["reviews_clean"])

# Saving the vectorizer so that it can be used later while deploying the model

import pickle

# Save to file in the current working directory
pkl_filename = "model/Tfidf_vectorizer.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(tfidf_counts, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickled_tfidf_vectorizer = pickle.load(file)


# Splitting the data into train and test

X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_data,
                                                                            for_sentiment_analysis['Sentiment_coded'],
                                                                            test_size = 0.2,
                                                                            random_state = 0)


print("Before OverSampling, counts of label '1': {}".format(sum(y_train_tfidf == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train_tfidf == 0)))

# import SMOTE module from imblearn library
#from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train_tfidf, y_train_tfidf.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res == 0)))

# Training after handling imbalance

LR = LogisticRegression(solver='lbfgs', max_iter=1000)
LR.fit(X_train_res, y_train_res.ravel())
predictions1 = LR.predict(X_test_tfidf)

# Confusion matrix
confusion = confusion_matrix(y_test_tfidf, predictions1)
print(confusion)

# print classification report

print(classification_report(y_test_tfidf, predictions1))
print("Accuracy : ",accuracy_score(y_test_tfidf, predictions1))
print("F1 score: ",f1_score(y_test_tfidf, predictions1))
print("Recall: ",recall_score(y_test_tfidf, predictions1))
print("Precision: ",precision_score(y_test_tfidf, predictions1))


# Saving the model as it will be used later while deploying

pkl_filename = "model/Logistic_Reg_final_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(LR, file)

# Load from file
with open(pkl_filename, 'rb') as file:
    pickled_model = pickle.load(file)





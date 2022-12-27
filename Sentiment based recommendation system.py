#### Capstone Project- Sentiment Based Product Recommendation System

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import pickle


#nltk.download('stopwords')
#ltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('punkt')


#  Building a recommendation system  # - User based recommendation system

# Reading the input data
reviews_dataset=pd.read_csv('sample30.csv')
for_sentiment_analysis=reviews_dataset
for_sentiment_analysis['reviews_title_text']= reviews_dataset['reviews_title'].fillna('') +" "+ reviews_dataset['reviews_text']

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

for_sentiment_analysis['Sentiment_coded'] = np.where(for_sentiment_analysis.user_sentiment == 'Positive',1,0)

for_user_based_reco= for_sentiment_analysis[for_sentiment_analysis['reviews_username'].isnull()== False]
for_user_based_reco.reset_index(drop=True)

# Test and Train split of the dataset.
train, test = train_test_split(for_user_based_reco, test_size=0.30, random_state=31)

# Pivot the train ratings' dataset into matrix format in which columns are products and the rows are user IDs.
df_pivot = train.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).fillna(0)
print(df_pivot.shape)
df_pivot.head(5)

# Copy the train dataset into dummy_train
dummy_train = train.copy()

# The products not rated by user is marked as 1 for prediction. 
dummy_train['reviews_rating'] = dummy_train['reviews_rating'].apply(lambda x: 0 if x>=1 else 1)

# Convert the dummy train dataset into matrix format.
dummy_train = dummy_train.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
).fillna(1)

# **Cosine Similarity**

from sklearn.metrics.pairwise import pairwise_distances

# Creating the User Similarity Matrix using pairwise_distance function.
user_correlation = 1 - pairwise_distances(df_pivot, metric='cosine')
user_correlation[np.isnan(user_correlation)] = 0

# **Prediction user user**

user_correlation[user_correlation<0]=0

# Rating predicted by the user (for products rated as well as not rated) is the weighted sum of correlation with the product rating (as present in the rating dataset).


user_predicted_ratings = np.dot(user_correlation, df_pivot.fillna(0))

user_final_rating = np.multiply(user_predicted_ratings,dummy_train)
user_final_rating.head()


# Finding top 5 recommendation for a user

# Take the user name as input.
user_input = str(input("Enter your user name"))
print(user_input)

d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:5]
print(d)

mapping=for_user_based_reco[['id','name']]
mapping = pd.DataFrame.drop_duplicates(mapping)

# Merging product id with mapping file to get the name of the recommended product
d = pd.merge(d,mapping, left_on='id', right_on='id', how = 'left')
print(d)


# Find out the common users of test and train dataset.
common = test[test.reviews_username.isin(train.reviews_username)]


# convert into the user-product matrix.
common_user_based_matrix = common.pivot_table(index='reviews_username',columns='id',values='reviews_rating')
common_user_based_matrix.head()

# Convert the user_correlation matrix into dataframe.
user_correlation_df = pd.DataFrame(user_correlation)

user_correlation_df['reviews_username'] = df_pivot.index

user_correlation_df.set_index('reviews_username',inplace=True)

list_name = common.reviews_username.tolist()

user_correlation_df.columns = df_pivot.index.tolist()

user_correlation_df_1 =  user_correlation_df[user_correlation_df.index.isin(list_name)]

user_correlation_df_2 = user_correlation_df_1.T[user_correlation_df_1.T.index.isin(list_name)]

user_correlation_df_3 = user_correlation_df_2.T

user_correlation_df_3[user_correlation_df_3<0]=0

common_user_predicted_ratings = np.dot(user_correlation_df_3, common_user_based_matrix.fillna(0))

dummy_test = common.copy()

dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)

dummy_test = dummy_test.pivot_table(index='reviews_username',columns='id',values='reviews_rating').fillna(0)

print(common_user_based_matrix.shape)

common_user_predicted_ratings = np.multiply(common_user_predicted_ratings,dummy_test)

# Calculating the RMSE for only the products rated by user. For RMSE, normalising the rating to (1,5) range

from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_user_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

common_ = common.pivot_table(index='reviews_username',columns='id',values='reviews_rating')

# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)

df_pivot = train.pivot_table(
   index='reviews_username',
    columns='id',
    values='reviews_rating'
).T

from sklearn.metrics.pairwise import pairwise_distances

# Item Similarity Matrix
item_correlation = 1 - pairwise_distances(df_pivot.fillna(0), metric='cosine')
item_correlation[np.isnan(item_correlation)] = 0

# Filtering the correlation only for which the value is greater than 0. (Positively correlated)

item_correlation[item_correlation<0]=0

item_predicted_ratings = np.dot((df_pivot.fillna(0).T),item_correlation)

# Filtering the rating only for products not rated by the user for recommendation

item_final_rating = np.multiply(item_predicted_ratings,dummy_train)

# Finding the top 5 recommendation for the user

# Take the user ID as input
user_input = str(input("Enter your user name"))
print(user_input)

# Recommending the Top 5 products to the user.
d = item_final_rating.loc[user_input].sort_values(ascending=False)[0:5]
print(d)

mapping= for_user_based_reco[['id','name']]
mapping = pd.DataFrame.drop_duplicates(mapping)

d = pd.merge(d,mapping, left_on='id', right_on='id', how = 'left')
print(d)


# **Evaluation item -item**
# 

common = test[test.id.isin(train.id	)]

common_item_based_matrix = common.pivot_table(index='reviews_username', columns='id', values='reviews_rating').T

item_correlation_df = pd.DataFrame(item_correlation)
item_correlation_df['id'] = df_pivot.index
item_correlation_df.set_index('id',inplace=True)
list_name = common.id.tolist()

item_correlation_df.columns = df_pivot.index.tolist()

item_correlation_df_1 =  item_correlation_df[item_correlation_df.index.isin(list_name)]

item_correlation_df_2 = item_correlation_df_1.T[item_correlation_df_1.T.index.isin(list_name)]

item_correlation_df_3 = item_correlation_df_2.T
item_correlation_df_3[item_correlation_df_3<0]=0
common_item_predicted_ratings = np.dot(item_correlation_df_3, common_item_based_matrix.fillna(0))

# Dummy test will be used for evaluation. To evaluate, we will only make prediction on the products rated by the user. So, this is marked as 1. This is just opposite of dummy_train
dummy_test = common.copy()
dummy_test['reviews_rating'] = dummy_test['reviews_rating'].apply(lambda x: 1 if x>=1 else 0)
dummy_test = dummy_test.pivot_table(index='reviews_username', columns='id', values='reviews_rating').T.fillna(0)
common_item_predicted_ratings = np.multiply(common_item_predicted_ratings,dummy_test)

# The products not rated is marked as 0 for evaluation. And make the item- item matrix representaion.

common_ = common.pivot_table(index='reviews_username', columns='id', values='reviews_rating').T

from sklearn.preprocessing import MinMaxScaler
from numpy import *

X  = common_item_predicted_ratings.copy() 
X = X[X>0]

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(X))
y = (scaler.transform(X))

# Finding total non-NaN value
total_non_nan = np.count_nonzero(~np.isnan(y))

rmse = (sum(sum((common_ - y )**2))/total_non_nan)**0.5
print(rmse)


# Finding top 20 recommendations for a selected user using User-based recommendation system

# Take the user ID as input
user_input = str(input("Enter your user name"))
print(user_input)

recommendations = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
mapping= for_user_based_reco[['id','name']]
mapping = pd.DataFrame.drop_duplicates(mapping)
recommendations = pd.merge(recommendations,mapping, left_on='id', right_on='id', how = 'left')

print(recommendations)

user_final_rating.to_pickle("model/user_final_rating.pkl")
pickled_user_final_rating = pd.read_pickle("model/user_final_rating.pkl")

mapping.to_pickle("model/prod_id_name_mapping.pkl")
pickled_mapping = pd.read_pickle("model/prod_id_name_mapping.pkl")

for_sentiment_analysis.to_pickle("model/reviews_data_all_cols.pkl")
pickled_reviews_data = pd.read_pickle("model/reviews_data_all_cols.pkl")


# ### 3. Improving the recommendations using the sentiment analysis model
pkl_filename_model = "model/Logistic_Reg_final_model.pkl"
pkl_filename_vec = "model/Tfidf_vectorizer.pkl"

# Predicting sentiment for the recommended products using the Logistic Regression model developed earlier
# Load vectorizer from file
with open(pkl_filename_vec, 'rb') as file:
    pickled_tfidf_vectorizer = pickle.load(file)

# Load model from file
with open(pkl_filename_model, 'rb') as file:
    pickled_model = pickle.load(file)

improved_recommendations= pd.merge(recommendations,pickled_reviews_data[['id','reviews_clean']], left_on='id', right_on='id', how = 'left')
test_data_for_user = pickled_tfidf_vectorizer.transform(improved_recommendations['reviews_clean'])
sentiment_prediction_for_user= pickled_model.predict(test_data_for_user)
sentiment_prediction_for_user = pd.DataFrame(sentiment_prediction_for_user, columns=['Predicted_Sentiment'])
improved_recommendations= pd.concat([improved_recommendations, sentiment_prediction_for_user], axis=1)

# For each of the 20 recommended products, calculating the percentage of positive sentiments
#   for all the reviews of each product

a=improved_recommendations.groupby('id')
b=pd.DataFrame(a['Predicted_Sentiment'].count()).reset_index()
b.columns = ['id', 'Total_reviews']
c=pd.DataFrame(a['Predicted_Sentiment'].sum()).reset_index()
c.columns = ['id', 'Total_predicted_positive_reviews']
improved_recommendations_final=pd.merge( b, c, left_on='id', right_on='id', how='left')
improved_recommendations_final['Positive_sentiment_rate'] = improved_recommendations_final['Total_predicted_positive_reviews'].div(improved_recommendations_final['Total_reviews']).replace(np.inf, 0)
improved_recommendations_final= improved_recommendations_final.sort_values(by=['Positive_sentiment_rate'], ascending=False )
improved_recommendations_final=pd.merge(improved_recommendations_final, pickled_mapping, left_on='id', right_on='id', how='left')

# Filtering out the top 5 products with the highest percentage of positive review
improved_recommendations_final.head(5)

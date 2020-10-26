import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None

training = pd.read_csv('Groceries_Processed_Training_Data.csv', nrows=10000)
del training['Unnamed: 0']

# read processed Test csv file with only 1 row per product
test = pd.read_csv('Groceries_Processed_Test_Data.csv', nrows=10000)
del test['Unnamed: 0']

print(test.columns)
print(test.head())

print(training.columns)

Y = training['Awesome?']
X = training[['ProductID', 'Reviews', 'Summaries', 'Number of Reviews']]

# feature scaling
scale_factor = X['Number of Reviews'].max()
X['Number of Reviews'] = X['Number of Reviews'] / scale_factor
test['Number of Reviews'] = test['Number of Reviews'] / scale_factor
print(X['Number of Reviews'].head())

print(training.head())
print(training.columns)

from sklearn.feature_extraction.text import CountVectorizer

# create bag of words count vectorizer,
review_body_vectorizer = CountVectorizer(max_features=4000, min_df=3, max_df=0.8, stop_words='english')
review_summary_vectorizer = CountVectorizer(max_features=4000, min_df=3, max_df=0.8, stop_words='english')

# process each product's bodies and summaries
processed_bodies = review_body_vectorizer.fit_transform(X['Reviews'].apply(lambda x: np.str_(x)))
processed_bodies = processed_bodies.toarray()

processed_summaries = review_summary_vectorizer.fit_transform(X['Summaries'].apply(lambda x: np.str_(x)))
processed_summaries = processed_summaries.toarray()

processed_features = [processed_bodies, processed_summaries]
# split X and y into test and test sets
from sklearn.model_selection import train_test_split

X_train, X_cross_validation, y_train, y_cross_validation = train_test_split(X, Y, test_size=0.1)
X_randomForestTrain, X_SVM_train, y_randomForestTrain, y_SVM_train = train_test_split(X_train, y_train, test_size=0.33)

# process the training and cross validation sets into bag of words format

processed_bodies_RF_train = review_body_vectorizer.transform(X_randomForestTrain['Reviews'].apply(lambda x: np.str_(x)))
processed_bodies_RF_train = processed_bodies_RF_train.toarray()

processed_summaries_RF_train = review_summary_vectorizer.transform(
    X_randomForestTrain['Summaries'].apply(lambda x: np.str_(x)))
processed_summaries_RF_train = processed_summaries_RF_train.toarray()

processed_bodies_SVM_train = review_body_vectorizer.transform(X_SVM_train['Reviews'].apply(lambda x: np.str_(x)))
processed_bodies_SVM_train = processed_bodies_SVM_train.toarray()

processed_summaries_SVM_train = review_summary_vectorizer.transform(
    X_SVM_train['Summaries'].apply(lambda x: np.str_(x)))
processed_summaries_SVM_train = processed_summaries_SVM_train.toarray()

processed_summaries_cv = review_summary_vectorizer.transform(
    X_cross_validation['Summaries'].apply(lambda x: np.str_(x)))
processed_bodies_cv = review_body_vectorizer.transform(X_cross_validation['Reviews'].apply(lambda x: np.str_(x)))

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

# create Multinomial NB model based on bag of words for combined summaries of each product
summary_model = GridSearchCV(MultinomialNB(), param_grid={'alpha': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                                                             'fit_prior': [True, False]})

summary_model.fit(processed_summaries_RF_train, y_randomForestTrain)


# create Multinomial NB model based on bag of words for combined reviewTexts of each product
body_model = GridSearchCV(MultinomialNB(), param_grid={'alpha': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                                                             'fit_prior': [True, False]})
body_model.fit(processed_bodies_RF_train, y_randomForestTrain)

# make predictions based on the Multinomial NB to get the sentiment scores
body_scores = body_model.predict(processed_bodies_SVM_train)
summary_scores = summary_model.predict(processed_summaries_SVM_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# print(confusion_matrix(y_cross_validation.tolist(),y_pred))
print(classification_report(np.round(body_scores.tolist()), y_SVM_train))
print(classification_report(np.round(summary_scores.tolist()), y_SVM_train))
# print(accuracy_score(y_cross_validation.tolist(), y_pred))

SVM_processed_training_features = pd.DataFrame({'summScore': summary_scores, 'bodyScore': body_scores})
SVM_processed_training_features['Number of Reviews'] = X_SVM_train['Number of Reviews'].values

# Import svm model
from sklearn import svm

# Create a svm Classifier
final_svm = svm.SVC(kernel='rbf')

# print(SVM_processed_training_features.columns)
print(SVM_processed_training_features.shape)
print(y_SVM_train.shape)
print(y_SVM_train.head())

# Train the model using the second training set
# Features in the set are:
# 1: sentiment score from summaries model
# 2: sentiment score from reviewText model
# 3: Number of reviews
final_svm.fit(SVM_processed_training_features, y_SVM_train.values)

# create cross validation set
cross_validation_features = pd.DataFrame(
    {'summScore': summary_model.predict(processed_summaries_cv),
     'bodyScore': body_model.predict(processed_bodies_cv)
     })
cross_validation_features['Number of Reviews'] = X_cross_validation['Number of Reviews'].values

# Predict awesomeness for cross_validation set, check accuracy
y_pred = final_svm.predict(cross_validation_features[['summScore', 'bodyScore', 'Number of Reviews']].values)
print(classification_report(y_pred, y_cross_validation))

# Run Model on Test Data

# create bag of words features for the test set
processed_test_bodies = review_body_vectorizer.transform(test['Reviews'].apply(lambda x: np.str_(x)))
processed_test_summaries = review_summary_vectorizer.transform(test['Summaries'].apply(lambda x: np.str_(x)))

# run the two random forest models on the bag of words features
# create sentiment scores for combined summaries and review bodies of each product
test['summScore'] = summary_model.predict(processed_test_summaries)
test['bodyScore'] = body_model.predict(processed_test_bodies)

del test['Reviews']
del test['Summaries']

# make predictions on the test set
test_y_predictions = final_svm.predict(test[['summScore', 'bodyScore', 'Number of Reviews']].values)
test['Predicted Awesome?'] = test_y_predictions

final_result = test[['ProductID', 'Predicted Awesome?']]
print(final_result.head())
print(final_result['Predicted Awesome?'].tolist())

final_result.sort_values('ProductID')
print(final_result.head())

# final_result.to_csv('groceries_test_set_predictions.csv')

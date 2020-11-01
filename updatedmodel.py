import pandas as pd
import numpy as np
import sklearn

pd.options.mode.chained_assignment = None

from sklearn.feature_extraction.text import TfidfVectorizer

# returns a sci_kit learn TFIDF vectorizer to convert text to TFIDF bag of words
def get_vectorizer(column, X, ngram_range):
    vectorizer = TfidfVectorizer(max_features=4000, stop_words='english', ngram_range=ngram_range)
    vectorizer.fit(X[column].apply(lambda x: np.str_(x)))
    return vectorizer

# returns bag of words for a Panda column of text entries
def process_TFIDF_bow(vectorizer, unprocessed_column):
    result = vectorizer.transform(unprocessed_column.apply(lambda x: np.str_(x)))
    return result.toarray()

# returns a trained RandomForest model
def get_trained_RandomForest_bodies(training_X, training_Y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=3, bootstrap=False, max_depth=50, max_features='auto', min_samples_leaf=1,
                                   min_samples_split=2, n_estimators=500)
    model.fit(training_X, training_Y)
    return model

def get_trained_RandomForest_summaries(training_X, training_Y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=3, bootstrap=True, max_depth=None, max_features='auto',
                                   min_samples_leaf=2, min_samples_split=10, n_estimators=100)
    model.fit(training_X, training_Y)
    return model

# returns a trained AdaBoost model
def get_trained_AdaBoost_bodies(training_X, training_Y):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(random_state=3, learning_rate=0.1, n_estimators=200)
    model.fit(training_X, training_Y)
    return model

def get_trained_AdaBoost_summaries(training_X, training_Y):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(random_state=3, learning_rate=0.5, n_estimators=50)
    model.fit(training_X, training_Y)
    return model

# returns a trained Multinomial Naive Bayes model
def get_trained_MultinomialNB(training_X, training_Y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(training_X, training_Y)
    return model

# returns a trained Gradient Boosting Classifier Model
def get_trained_GBC(training_X, training_Y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=3)
    model.fit(training_X, training_Y)
    return model

# takes:
# a dictionary of models
#  TFIDF bag of words features for the review summaries
# TFIDF bag of words features for the review bodies,
# the Vader quantile Scores for each product
# returns a panda of all of the features for the final SVM
def get_SVM_features(models, processed_summaries, processed_bodies, vader_scores):
    result = pd.DataFrame()
    for model_name in models.keys():
        # if the model is trained on the review bodies
        if model_name[-6:] == "bodies":
            # make predictions on the body features
            result[model_name] = models[model_name].predict_proba(processed_bodies)[:, 1]
        # else if the model is trained on the summaries
        else:
            # make predictions on the summary features
            result[model_name] = models[model_name].predict_proba(processed_summaries)[:, 1]
    vader_score_col_names = ['compound25', 'compound50', 'compound75', 'pos25', 'pos50', 'pos75', 'neg25', 'neg50', 'neg75', 'neu25', 'neu50', 'neu75']
    for name in vader_score_col_names:
        temp = name + 'Text'
        result[temp] = vader_scores[temp].values
        temp = name + 'Summary'
        result[temp] = vader_scores[temp].values
    return result

# takes a list of models, and makes predictions on the bodies and scores
# def get_SVM_features(models, processed_bodies, vader_scores):
#     result = pd.DataFrame()
#     for model_name in models.keys():
#         # if the model is trained on the review bodies
#         if model_name[-6:] == "bodies":
#             # make predictions on the body features
#             result[model_name] = models[model_name].predict_proba(processed_bodies)[:, 1]
#         # else if the model is trained on the summaries
#         else:
#             continue
#             # make predictions on the summary features
#             # result[model_name] = models[model_name].predict_proba(processed_summaries)[:, 1]
#     vader_score_col_names = ['compound25', 'compound50', 'compound75', 'pos25', 'pos50', 'pos75', 'neg25', 'neg50', 'neg75', 'neu25', 'neu50', 'neu75']
#     for name in vader_score_col_names:
#         result[name] = vader_scores[name].values
#     return result

# returns trained SVM model
def get_trained_SVM(processed_SVM_training_features, y_train):
    from sklearn import svm
    model = svm.SVC(kernel='rbf')
    model.fit(processed_SVM_training_features, y_train)
    return model
#
def addVaderFeatures(panda, unprocessed_text):
    print(unprocessed_text.size)
    print(panda.size)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    panda['compound'] = [analyzer.polarity_scores(x)['compound'] for x in unprocessed_text]
    panda['neg'] = [analyzer.polarity_scores(x)['neg'] for x in unprocessed_text]
    panda['neu'] = [analyzer.polarity_scores(x)['neu'] for x in unprocessed_text]
    panda['pos'] = [analyzer.polarity_scores(x)['pos'] for x in unprocessed_text]

def saveOuterModelFeatures(filename, X, y):
    pass


training = pd.read_csv('Groceries_Processed_Training_Data_Vader.csv', nrows=10000)
print(training.columns)
del training['Unnamed: 0']

Y = training['Awesome?']
X = training[['ProductID', 'Reviews', 'Summaries', 'Number of Reviews', 'compound25Text', 'compound50Text', 'compound75Text', 'pos25Text', 'pos50Text', 'pos75Text', 'neg25Text', 'neg50Text', 'neg75Text', 'neu25Text', 'neu50Text', 'neu75Text', 'compound25Summary', 'compound50Summary', 'compound75Summary', 'pos25Summary', 'pos50Summary', 'pos75Summary', 'neg25Summary', 'neg50Summary', 'neg75Summary', 'neu25Summary', 'neu50Summary', 'neu75Summary']]

# feature scaling
scale_factor = X['Number of Reviews'].max()
X['Number of Reviews'] = X['Number of Reviews'] / scale_factor

# create bag of words TF-IDF vectorizer,
rf_review_body_vectorizer = get_vectorizer('Reviews', X, (1,1))
rf_review_summary_vectorizer = get_vectorizer('Summaries', X, (1,2))

#  split X and y into test and test sets
from sklearn.model_selection import train_test_split
X_train, X_cross_validation, y_train, y_cross_validation = train_test_split(X, Y, test_size=0.1)
X_innerTrain, X_outer_train, y_innerTrain, y_outerTrain = train_test_split(X_train, y_train, test_size=0.33)

# process the training and cross validation sets into bag of words format

# training set for all inner models (RF, NB, GBC, Adaboost, etc.)
processed_bodies_inner_train = process_TFIDF_bow(rf_review_body_vectorizer, X_innerTrain['Reviews'])
processed_summaries_inner_train = process_TFIDF_bow(rf_review_summary_vectorizer, X_innerTrain['Summaries'])

#training set for outer SVM
processed_bodies_outer_train = process_TFIDF_bow(rf_review_body_vectorizer, X_outer_train['Reviews'])
processed_summaries_outer_train = process_TFIDF_bow(rf_review_summary_vectorizer, X_outer_train['Summaries'])

# cross validation set for testing final model
processed_summaries_cv = process_TFIDF_bow(rf_review_summary_vectorizer, X_cross_validation['Summaries'])
processed_bodies_cv = process_TFIDF_bow(rf_review_body_vectorizer, X_cross_validation['Reviews'])

print("done getting features")

models = {}

## RANDOM FOREST

# create RF model based on bag of words for combined summaries of each product
RFC_summaries = get_trained_RandomForest_summaries(processed_summaries_inner_train, y_innerTrain)
models['RFsummaries'] = RFC_summaries

# create RF model based on bag of words for combined reviewTexts of each product
RFC_bodies = get_trained_RandomForest_bodies(processed_bodies_inner_train, y_innerTrain)
models['RFbodies'] = RFC_bodies

# make predictions based on the random forest models to get the sentiment scores
body_scores = RFC_bodies.predict_proba(processed_bodies_outer_train)[:, 1]
summary_scores = RFC_summaries.predict_proba(processed_summaries_outer_train)[:, 1]

from sklearn.metrics import classification_report
print("Random Forest bodies and summaries respective stats")
print(classification_report(np.round(body_scores.tolist()), y_outerTrain))
print(classification_report(np.round(summary_scores.tolist()), y_outerTrain))


## ADAboost Model

ADA_bodies = get_trained_AdaBoost_bodies(processed_bodies_inner_train, y_innerTrain)
models['ADAbodies'] = ADA_bodies
ADA_summaries = get_trained_AdaBoost_summaries(processed_summaries_inner_train, y_innerTrain)
models['ADAsummaries'] = ADA_summaries

# make predictions based on models to get the sentiment scores
body_scores = ADA_bodies.predict_proba(processed_bodies_outer_train)[:, 1]
summary_scores = ADA_summaries.predict_proba(processed_summaries_outer_train)[:, 1]

from sklearn.metrics import classification_report
print("ADAboost bodies/summaries respective stats")
print(classification_report(np.round(body_scores.tolist()), y_outerTrain))
print(classification_report(np.round(summary_scores.tolist()), y_outerTrain))


## Multinomial NaiveBayes Model


NB_summaries = get_trained_MultinomialNB(processed_summaries_inner_train, y_innerTrain)
models['NBsummaries'] = NB_summaries
NB_bodies = get_trained_MultinomialNB(processed_bodies_inner_train, y_innerTrain)
models['NBbodies'] = NB_bodies

NB_body_scores = NB_bodies.predict_proba(processed_bodies_outer_train)[:, 1]
NB_summary_scores = NB_summaries.predict_proba(processed_summaries_outer_train)[:, 1]

print("NaiveBayes bodies/summaries respective stats")
print(classification_report(np.round(NB_body_scores.tolist()), y_outerTrain))
print(classification_report(np.round(NB_summary_scores.tolist()), y_outerTrain))


## Gradient Boosting classifier model

GBC_summaries = get_trained_GBC(processed_summaries_inner_train, y_innerTrain)
models["GB_summaries"] = GBC_summaries
GBC_bodies = get_trained_GBC(processed_bodies_inner_train, y_innerTrain)
models['GB_bodies'] = GBC_bodies

# make predictions based on the two gbc models to get the sentiment scores
body_scores = GBC_bodies.predict_proba(processed_bodies_outer_train)[:, 1]
summary_scores = GBC_summaries.predict_proba(processed_summaries_outer_train)[:, 1]

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("GradientBoostingClassifier bodies/summaries respective stats")
print(classification_report(np.round(body_scores.tolist()), y_outerTrain))
print(classification_report(np.round(summary_scores.tolist()), y_outerTrain))


print("starting SVM")

## Outer SVM

#process training features
SVM_training_features = get_SVM_features(models, processed_summaries_outer_train, processed_bodies_outer_train, X_outer_train)
SVM_training_features["NumberReviews"] = X_outer_train['Number of Reviews'].values
SVM_training_features.to_csv("SVM_training_features_q.csv", index=False)
y_outerTrain.to_csv("SVM_train_Y_q.csv", index=False)
# addVaderFeatures(SVM_training_features, X_outer_train['Summaries'].values)


nan_values = SVM_training_features.isna()
nan_columns = nan_values.any()

columns_with_nan = SVM_training_features.columns[nan_columns].tolist()
print(columns_with_nan)

SVM_training_features.fillna(0)

# get trained model
outer_SVM = get_trained_SVM(SVM_training_features, y_outerTrain)

# process testing features
SVM_testing_features = get_SVM_features(models, processed_summaries_cv, processed_bodies_cv, X_cross_validation)
SVM_testing_features['NumberReviews'] = X_cross_validation['Number of Reviews'].values
SVM_testing_features.to_csv("SVM_testing_features_q.csv", index=False)
y_cross_validation.to_csv("SVM_test_Y_q.csv", index=False)
# addVaderFeatures(SVM_testing_features, X_cross_validation['Summaries'].values)
print(sum(SVM_testing_features.isnull().values.any(axis=1)))
SVM_testing_features.fillna(0)

# make predictions, evaluate performance
y_pred = outer_SVM.predict(SVM_testing_features)
print(classification_report(np.round(y_pred.tolist()), y_cross_validation))

# tests different combinations of the models we have in the svm
combinations = []

combinations.append(['RFbodies', 'ADAbodies'])
combinations.append(['RFbodies', 'GB_bodies', 'NBbodies'])
combinations.append(['RFbodies', 'ADAbodies', 'GB_bodies'])
combinations.append(['ADAbodies', 'GB_bodies'])
combinations.append(['ADAbodies'])
combinations.append(['GB_bodies'])
combinations.append(['GB_bodies', 'NBbodies'])
combinations.append(['RFbodies'])

# combinations.append(['RFsummaries', 'RFbodies', 'ADAsummaries', 'ADAbodies'])
# combinations.append(['RFsummaries', 'RFbodies', 'GB_summaries', 'GB_bodies', 'NBsummaries', 'NBbodies'])
# combinations.append(['RFsummaries', 'RFbodies', 'ADAsummaries', 'ADAbodies', 'GB_summaries', 'GB_bodies', 'NBsummaries'])
# combinations.append(['ADAsummaries', 'ADAbodies', 'GB_summaries', 'GB_bodies'])
# combinations.append(['RFsummaries', 'RFbodies', 'ADAsummaries', 'ADAbodies'])
# combinations.append(['RFsummaries', 'RFbodies', 'GB_summaries', 'GB_bodies'])
# combinations.append(['RFsummaries', 'RFbodies', 'GB_summaries', 'GB_bodies', 'NBsummaries'])
# combinations.append(['RFsummaries', 'RFbodies'])

for comb in combinations:

    done = False
    i = 0

    while not done:
        # if i == 1:
        #     comb.append('NumberReviews')
        if i == 1:
            vader_cols = ['compound25', 'compound50', 'compound75', 'pos25', 'pos50', 'pos75', 'neg25', 'neg50', 'neg75', 'neu25', 'neu50', 'neu75']
            for col in vader_cols:
                comb.append(col + 'Text')
                comb.append(col + 'Summary')
            done = True
        SVM = get_trained_SVM(SVM_training_features[comb], y_outerTrain)
        y_pred = SVM.predict(SVM_testing_features[comb])
        print(comb)
        print(classification_report(np.round(y_pred.tolist()), y_cross_validation))
        i += 1

import numpy as np
import pandas as pd
import sklearn

pd.options.mode.chained_assignment = None
file = open("output.txt", 'w')
from sklearn.feature_extraction.text import TfidfVectorizer
def get_vectorizer(column, X, ngram_range):
    vectorizer = TfidfVectorizer(max_features=4000, stop_words='english', ngram_range=ngram_range)
    vectorizer.fit(X[column].apply(lambda x: np.str_(x)))
    return vectorizer

def process_TFIDF_bow(vectorizer, unprocessed_column):
    result = vectorizer.transform(unprocessed_column.apply(lambda x: np.str_(x)))
    return result.toarray()

def get_trained_RandomForest_bodies(training_X, training_Y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(RandomForestClassifier(random_state=3), param_grid={
        'n_estimators': (100, 200, 500),
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'max_depth': [None, 5, 10, 50, 100],
        'min_samples_split': (2, 5, 10),
        'min_samples_leaf': (1, 2, 5, 10)
    }, n_jobs=-1)
    model.fit(training_X, training_Y)
    file.write("Best parameters for RF bodies model: ")
    file.write(model.best_params_)
    file.write(model.best_score_)
    return model

def get_trained_RandomForest_summaries(training_X, training_Y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(RandomForestClassifier(random_state=3), param_grid={
        'n_estimators': (100, 200, 500),
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'max_depth': [None, 5, 10, 50, 100],
        'min_samples_split': (2, 5, 10),
        'min_samples_leaf': (1, 2, 5, 10)
    }, n_jobs=-1)
    model.fit(training_X, training_Y)
    file.write("Best parameters for RF summaries model: ")
    file.write(model.best_params_)
    file.write(model.best_score_)
    return model

def get_trained_AdaBoost_summaries(training_X, training_Y):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(AdaBoostClassifier(random_state=3), param_grid={
        'n_estimators': (50, 100, 200),
        'learning_rate': (0.1, 0.5, 1)
    })
    model.fit(training_X, training_Y)
    file.write("Best parameters for Adaboost summaries model: ")
    file.write(model.best_params_)
    file.write(model.best_score_)
    return model

def get_trained_AdaBoost_bodies(training_X, training_Y):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(AdaBoostClassifier(random_state=3), param_grid={
        'n_estimators': (50, 100, 200),
        'learning_rate': (0.1, 0.5, 1)
    })
    model.fit(training_X, training_Y)
    file.write("Best parameters for Adaboost bodies model: ")
    file.write(model.best_params_)
    file.write(model.best_score_)
    return model

def get_trained_MultinomialNB(training_X, training_Y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(training_X, training_Y)
    return model

def get_trained_GBC_summaries(training_X, training_Y):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(GradientBoostingClassifier(random_state=3), param_grid={
            'loss': ['deviance', 'exponential'],
            'learning_rate': (0.1, 0.2, 0.5),
            'n_estimators': (100, 200, 500),
            'criterion': ['friedman_mse', 'mse', 'mae'],
            'min_samples_split': (2, 5, 10),
            'max_depth': (3, 5, 10),
            'max_features': ['auto', 'sqrt', 'log2']
        })
    model.fit(training_X, training_Y)
    file.write("Best parameters for GBC summaries model: ")
    file.write(model.best_params_)
    file.write(model.best_score_)
    return model

def get_trained_GBC_bodies(training_X, training_Y):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(GradientBoostingClassifier(random_state=3), param_grid={
            'loss': ['deviance', 'exponential'],
            'learning_rate': (0.1, 0.2, 0.5),
            'n_estimators': (100, 200, 500),
            'criterion': ['friedman_mse', 'mse', 'mae'],
            'min_samples_split': (2, 5, 10),
            'max_depth': (3, 5, 10),
            'max_features': ['auto', 'sqrt', 'log2']
        })
    model.fit(training_X, training_Y)
    file.write("Best parameters for GBC bodies model: ")
    file.write(model.best_params_)
    file.write(model.best_score_)
    return model

def get_SVM_features(models, processed_summaries, processed_bodies):
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
    return result

def get_trained_SVM(processed_SVM_training_features, y_train):
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(svm.SVC(), param_grid={
        'C': (0.1, 1, 2),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
        'shrinking': [True, False]
    })
    model.fit(processed_SVM_training_features, y_train)
    file.write("Best parameters for SVC model: ")
    file.write(model.best_params_)
    file.write(model.best_score_)
    return model

def addVaderFeatures(panda, unprocessed_text):
    file.write(unprocessed_text.size)
    file.write(panda.size)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    panda['compound'] = [analyzer.polarity_scores(x)['compound'] for x in unprocessed_text]
    panda['neg'] = [analyzer.polarity_scores(x)['neg'] for x in unprocessed_text]
    panda['neu'] = [analyzer.polarity_scores(x)['neu'] for x in unprocessed_text]
    panda['pos'] = [analyzer.polarity_scores(x)['pos'] for x in unprocessed_text]



training = pd.read_csv('Groceries_Processed_Training_Data.csv', nrows=5000)
del training['Unnamed: 0']

Y = training['Awesome?']
X = training[['ProductID', 'Reviews', 'Summaries', 'Number of Reviews']]

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

file.write("done getting features")

models = {}

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

file.write(classification_report(np.round(body_scores.tolist()), y_outerTrain))
file.write(classification_report(np.round(summary_scores.tolist()), y_outerTrain))

ADA_bodies = get_trained_AdaBoost_bodies(processed_bodies_inner_train, y_innerTrain)
models['ADAbodies'] = ADA_bodies
ADA_summaries = get_trained_AdaBoost_summaries(processed_summaries_inner_train, y_innerTrain)
models['ADAsummaries'] = ADA_summaries

# make predictions based on the random forest models to get the sentiment scores
body_scores = ADA_bodies.predict_proba(processed_bodies_outer_train)[:, 1]
summary_scores = ADA_summaries.predict_proba(processed_summaries_outer_train)[:, 1]

from sklearn.metrics import classification_report

file.write(classification_report(np.round(body_scores.tolist()), y_outerTrain))
file.write(classification_report(np.round(summary_scores.tolist()), y_outerTrain))

NB_summaries = get_trained_MultinomialNB(processed_summaries_inner_train, y_innerTrain)
models['NBsummaries'] = NB_summaries
NB_bodies = get_trained_MultinomialNB(processed_summaries_inner_train, y_innerTrain)
models['NBbodies'] = NB_bodies

NB_body_scores = NB_bodies.predict_proba(processed_bodies_outer_train)[:, 1]
NB_summary_scores = NB_summaries.predict_proba(processed_summaries_outer_train)[:, 1]

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
file.write(classification_report(np.round(NB_body_scores.tolist()), y_outerTrain))
file.write(classification_report(np.round(NB_summary_scores.tolist()), y_outerTrain))


GBC_summaries = get_trained_GBC_summaries(processed_summaries_inner_train, y_innerTrain)
models["GB_summaries"] = GBC_summaries
GBC_bodies = get_trained_GBC_bodies(processed_bodies_inner_train, y_innerTrain)
models['GB_bodies'] = GBC_bodies

# make predictions based on the random forest models to get the sentiment scores
body_scores = GBC_bodies.predict_proba(processed_bodies_outer_train)[:, 1]
summary_scores = GBC_summaries.predict_proba(processed_summaries_outer_train)[:, 1]

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# file.write(confusion_matrix(y_cross_validation.tolist(),y_pred))
file.write(classification_report(np.round(body_scores.tolist()), y_outerTrain))
file.write(classification_report(np.round(summary_scores.tolist()), y_outerTrain))
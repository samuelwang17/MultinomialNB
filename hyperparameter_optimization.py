
import numpy as np
import pandas as pd
import sklearn

pd.options.mode.chained_assignment = None
from sklearn.feature_extraction.text import TfidfVectorizer
def get_vectorizer(column, X, ngram_range):
    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=ngram_range)
    vectorizer.fit(X[column].apply(lambda x: np.str_(x)))
    return vectorizer

def process_TFIDF_bow(vectorizer, unprocessed_column):
    result = vectorizer.transform(unprocessed_column.apply(lambda x: np.str_(x)))
    return result.toarray()

# hyperparameter optimization for RandomForest bodies
def get_RandomForest_bodies_parameters(training_X, training_Y):
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
    print("Best parameters for RF bodies model: ")
    print(str(model.best_params_) + "\n")


# hyperparameter optimization for RandomForest summaries
def get_RandomForest_summaries_parameters(training_X, training_Y):
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
    print("Best parameters for RF summaries model: ")
    print(str(model.best_params_) + "\n")

# hyperparameter optimization for AdaBoost summaries
def get_AdaBoost_summaries_parameters(training_X, training_Y):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(AdaBoostClassifier(random_state=3), param_grid={
        'n_estimators': (50, 100, 200),
        'learning_rate': (0.1, 0.5, 1)
    })
    model.fit(training_X, training_Y)
    print("Best parameters for Adaboost summaries model: ")
    print(str(model.best_params_) + "\n")

# hyperparameter optimization for AdaBoost bodies
def get_AdaBoost_bodies_parameters(training_X, training_Y):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(AdaBoostClassifier(random_state=3), param_grid={
        'n_estimators': (50, 100, 200),
        'learning_rate': (0.1, 0.5, 1)
    })
    model.fit(training_X, training_Y)
    print("Best parameters for Adaboost bodies model: ")
    print(str(model.best_params_) + "\n")

# hyperparameter optimization for GBC summaries
def get_GBC_summaries_parameters(training_X, training_Y):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(GradientBoostingClassifier(random_state=3), param_grid={
            'learning_rate': (0.1, 0.2, 0.5),
            'n_estimators': (100, 200, 500),
            'min_samples_split': (2, 5, 10)
        })
    model.fit(training_X, training_Y)
    print("\n" + "Best parameters for GBC summaries model: ")
    print("\n" + str(model.best_params_) + "\n")

# hyperparameter optimization for GBC bodies
def get_GBC_bodies_parameters(training_X, training_Y):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(GradientBoostingClassifier(random_state=3), param_grid={
        'learning_rate': (0.1, 0.2, 0.5),
        'n_estimators': (100, 200, 500),
        'min_samples_split': (2, 5, 10)
    })
    model.fit(training_X, training_Y)
    print("Best parameters for GBC bodies model: ")
    print(str(model.best_params_))


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

# hyperparameter optimization for SVM
def get_SVM_parameters(processed_SVM_training_features, y_train):
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(svm.SVC(), param_grid={
        'C': (0.1, 1, 2),
        'kernel': ['poly', 'rbf'],
        'shrinking': [True, False]
    })
    model.fit(processed_SVM_training_features, y_train)
    print("Best parameters for SVC model: ")
    print(str(model.best_params_))


def addVaderFeatures(panda, unprocessed_text):
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

print("done getting features")

get_RandomForest_summaries_parameters(processed_summaries_inner_train, y_innerTrain)
get_RandomForest_bodies_parameters(processed_bodies_inner_train, y_innerTrain)

get_AdaBoost_bodies_parameters(processed_bodies_inner_train, y_innerTrain)
get_AdaBoost_summaries_parameters(processed_summaries_inner_train, y_innerTrain)

get_GBC_summaries_parameters(processed_summaries_inner_train, y_innerTrain)
get_GBC_bodies_parameters(processed_bodies_inner_train, y_innerTrain)

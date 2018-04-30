import pandas
from sklearn import cross_validation as cv
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def return_nonstring_col(data_cols):
    cols_to_keep = []
    train_cols = []
    for col in data_cols:
        if col != 'URL' and col != 'host' and col != 'path':
            cols_to_keep.append(col)
            if col != 'malicious' and col != 'result':
                train_cols.append(col)
    return [cols_to_keep, train_cols]


def svm_classifier(train, query,
                   train_cols):
    clf = svm.SVC()

    clf.fit(train[train_cols], train['malicious'])
    scores = cv.cross_val_score(clf, train[train_cols], train['malicious'], cv=30)
    print('Estimated score SVM: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

    query['result'] = clf.predict(query[train_cols])

    #print(query[['URL', 'result']])
    query[['URL', 'result']].to_csv("csvfiles/svm.csv")


# Called from gui
def forest_classifier_gui(train, query,
                          train_cols):
    rf = RandomForestClassifier(n_estimators=150)

    print(rf.fit(train[train_cols], train['malicious']))

    query['result'] = rf.predict(query[train_cols])

    print(query[['URL', 'result']].head(2))
    return query['result']


def svm_classifier_gui(train, query,
                       train_cols):
    clf = svm.SVC()

    train[train_cols] = preprocessing.scale(train[train_cols])
    query[train_cols] = preprocessing.scale(query[train_cols])

    print(clf.fit(train[train_cols], train['malicious']))

    query['result'] = clf.predict(query[train_cols])

    print(query[['URL', 'result']].head(2))
    return query['result']


def forest_classifier(train, query,
                      train_cols):
    rf = RandomForestClassifier(n_estimators=150)

    rf.fit(train[train_cols], train['malicious'])
    scores = cv.cross_val_score(rf, train[train_cols], train['malicious'], cv=30)
    print('Estimated score RandomForestClassifier: %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

    query['result'] = rf.predict(query[train_cols])
    query[['URL', 'result']].to_csv("csvfiles/rf.csv")


def logistic_regression(train, query,
                        train_cols):
    logis = LogisticRegression()

    logis.fit(train[train_cols], train['malicious'])
    scores = cv.cross_val_score(logis, train[train_cols], train['malicious'], cv=30)
    print('Estimated score logisticregression : %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

    query['result'] = logis.predict(query[train_cols])
    #print(query[['URL', 'result']])
    query[['URL', 'result']].to_csv("csvfiles/logis.csv")


def DecisionTree_Classifier(train, query,
                            train_cols):
    deci = DecisionTreeClassifier(random_state=100, max_depth=3, min_samples_leaf=5)
    deci.fit(train[train_cols], train['malicious'])
    scores = cv.cross_val_score(deci, train[train_cols], train['malicious'], cv=30)
    print('Estimated score decisiontreeclassifier : %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

    query['result'] = deci.predict(query[train_cols])
    #print(query[['URL', 'result']])
    query[['URL', 'result']].to_csv("csvfiles/deci.csv")


def KNeighbors_Classifier(train, query,
                          train_cols):
    Kneigh = KNeighborsClassifier()
    Kneigh.fit(train[train_cols], train['malicious'])
    scores = cv.cross_val_score(Kneigh, train[train_cols], train['malicious'], cv=30)
    print('Estimated score KNeighborsClassifier : %0.5f (+/- %0.5f)' % (scores.mean(), scores.std() / 2))

    query['result'] = Kneigh.predict(query[train_cols])
    query[['URL', 'result']].to_csv("csvfiles/Kneigh.csv")


def train(db, test_db):
    query_csv = pandas.read_csv(test_db)
    cols_to_keep, train_cols = return_nonstring_col(query_csv.columns)
    query=query_csv[cols_to_keep]

    train_csv = pandas.read_csv(db)
    cols_to_keep, train_cols = return_nonstring_col(train_csv.columns)
    train = train_csv[cols_to_keep]

    svm_classifier(train_csv, query_csv, train_cols)
    forest_classifier(train_csv, query_csv, train_cols)
    logistic_regression(train_csv, query_csv, train_cols)
    DecisionTree_Classifier(train_csv, query_csv, train_cols)
    KNeighbors_Classifier(train_csv, query_csv, train_cols)


def gui_caller(db, test_db):
    query_csv = pandas.read_csv(test_db)
    cols_to_keep, train_cols = return_nonstring_col(query_csv.columns)
    query=query_csv[cols_to_keep]

    train_csv = pandas.read_csv(db)
    cols_to_keep, train_cols = return_nonstring_col(train_csv.columns)
    train = train_csv[cols_to_keep]
    return forest_classifier_gui(train_csv, query_csv, train_cols)


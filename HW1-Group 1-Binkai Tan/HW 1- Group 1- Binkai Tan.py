import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate, GridSearchCV
import xgboost as xgb
from sklearn import *
from sklearn.utils import shuffle
from sklearn.metrics import auc
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, make_scorer
import matplotlib.pyplot as plt
import seaborn as sns

# load data & transfer the format
def loadData():
    # 0 - not fake
    # 1 - fake
    data = pd.read_csv('data/onion_reuters_embeddings.csv', header=None)
    X = []
    y = []

    # shuffle the data
    data = shuffle(data)

    # Then split the data into X and y(the last column is y)
    for i in range(0, len(data)):
        temp = data.iloc[i, 0].split(' ')
        X.append([float(i) for i in temp[0:-1]])
        y.append(int(float(temp[-1])))
    # turn to DataFrame
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X.columns = ['x' + str(i) for i in range(1, 51)]
    y.columns = ['label']
    # here should use X.copy()
    data = X.copy()
    data['label'] = y['label']
    # split X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    data.head(10)
    f, (ax1) = plt.subplots(1, 1, figsize=(24, 20))

    # Entire DataFrame
    corr = data.corr()
    sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size': 20}, ax=ax1)
    ax1.set_title("Correlation Matrix", fontsize=14)

    plt.show()

    f, axes = plt.subplots(ncols=4, figsize=(20, 4))

    # Negative Correlations with our Class (The lower our feature value the more likely it will be a fraud transaction)
    sns.boxplot(x="label", y="x9", data=data, ax=axes[0])
    axes[0].set_title('x9 vs label Negative Correlation')

    sns.boxplot(x="label", y="x12", data=data, ax=axes[1])
    axes[1].set_title('x12 vs label Negative Correlation')

    sns.boxplot(x="label", y="x31", data=data, ax=axes[2])
    axes[2].set_title('x31 vs label Negative Correlation')

    sns.boxplot(x="label", y="x39", data=data, ax=axes[3])
    axes[3].set_title('x39 vs label Negative Correlation')

    plt.show()

    return X_train, X_test, y_train, y_test

def XGBoost(X_train, y_train, X_test, y_test):
    # xgb_model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
    #           colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=1,
    #           max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
    #           n_estimators=100, n_jobs=1, nthread=None,
    #           objective='binary:logistic', random_state=0, reg_alpha=0,
    #           reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
    #           subsample=1, verbosity=1)

    # use XGBClassifier
    xgb_model = xgb.XGBClassifier()

    # this is the hyperparameter grid
    # try each combination
    # find the best hyper parameter conmination
    hyper_param_grid = [
        {'max_depth': [1, 3, 10], 'learning_rate': [0.01, 0.05, 0.1, 1]},
        {'verbosity': [1, 2, 3], 'reg_alpha': [0, 0.5, 1], 'reg_lambda': [0, 0.5, 1]},
    ]

    # use grid search and cross validation to search the hyper parameter grid, set the fold is 5, judging the model with accuracy
    xgb_model_grid = GridSearchCV(xgb_model, hyper_param_grid, cv=5, scoring='accuracy')

    # fit the model with X_train and y_train
    xgb_model_grid.fit(X_train, y_train.values.ravel())

    # print the model with hyperparameters
    print(xgb_model_grid.best_estimator_)

    pred = xgb_model_grid.predict(X_test)  # Perform classification on an array of test vectors X.
    print(pred)

    # confusion matrix, precision, recall, f1
    cm = metrics.confusion_matrix(y_test, pred)
    print("confusion matrix: ")
    print(cm)
    precision = metrics.precision_score(y_test, pred)
    recall = metrics.recall_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred)
    print("precision:   %0.3f" % precision)
    print("recall:   %0.3f" % recall)
    print("f1:   %0.3f" % f1)

    # plot the confusion matrix
    classes = ['1-Fake', '0-Not fake']

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)  # Display matrix by pixel
    plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    iters = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (cm.size, 2))
    for i, j in iters:
        plt.text(j, i, format(cm[i, j]))  # Display the corresponding number

    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()

    # plot the precision-recall curve for XGBoostwith AUC
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, pred)
    # calculate precision-recall AUC
    auc_xgb = auc(recall, precision)
    # plot no skill
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.')
    plt.title("Precison-Recall Curve for XGBoost with AUC score: {:.3f}".format(auc_xgb))
    # show the plot
    plt.show()


def randomRofrest(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train.values.ravel())  # Fit XGB classifier according to X, y

    pred = rf_model.predict(X_test)  # Perform classification on an array of test vectors X.
    print(pred)
    # confusion matrix, precision, recall, f1
    cm = metrics.confusion_matrix(y_test, pred)
    print(cm)
    precision = metrics.precision_score(y_test, pred)
    recall = metrics.recall_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred)
    print("precision:   %0.3f" % precision)
    print("recall:   %0.3f" % recall)
    print("f1:   %0.3f" % f1)

    classes = ['1-Fake', '0-Not fake']


    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵
    plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    iters = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (cm.size, 2))
    for i, j in iters:
        plt.text(j, i, format(cm[i, j]))  # 显示对应的数字

    plt.ylabel('Real label')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()

    # plot the precision-recall curve for Randomforest with AUC
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, pred)
    # calculate precision-recall AUC
    auc_xgb = auc(recall, precision)
    # plot no skill
    plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.')
    plt.title("Precison-Recall Curve for Randomforest with AUC score: {:.3f}".format(auc_xgb))
    # show the plot
    plt.show()





if __name__ == '__main__':
    X_train, X_test, y_train, y_test = loadData()
    XGBoost(X_train, y_train, X_test, y_test)
    randomRofrest(X_train, y_train, X_test, y_test)




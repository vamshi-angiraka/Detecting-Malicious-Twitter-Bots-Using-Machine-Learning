import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn import metrics

dataset = pd.read_csv('kaggle_train.csv')

words = ['bot','cannabis','tweet me','mishear','follow me','updates','every','gorilla','forget']
train = dataset[
  ['screen_name','status','name','followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified']]

def getFrequency(bow):
    count = 0
    for i in range(len(words)):
        if words[i] in bow:
            count = count + bow.get(words[i])
    return count        

def method2():
    details = train.values
    for i in range(len(details)):
        screen = details[i,0]
        status = details[i,1]
        name = details[i,2]
        followers = int(details[i,3])
        friends = int(details[i,4])
        listed = int(details[i,5])
        favourite = int(details[i,6])
        status_count = int(details[i,7])
        verified = details[i,8]
        if not verified:
            bow = defaultdict(int)
            data = str(screen)+" "+str(name)+" "+str(status)
            data = data.lower().strip("\n").strip()
            data = re.findall(r'\w+', data)
            for j in range(len(data)):
                bow[data[j]] += 1
            frequency = getFrequency(bow)
            if frequency > 0 and listed < 16000 and followers < 200:
                print(screen)
    train_attr = dataset[
        ['followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified']]
    train_label = dataset[['bot']]

    X = train_attr
    Y = train_label.as_matrix()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    logreg = LogisticRegression().fit(X_train, y_train)
    actual = y_test
    pred = logreg.predict(X_test)
    accuracy = accuracy_score(actual, pred) * 100
    precision = precision_score(actual, pred) * 100
    recall = recall_score(actual, pred) * 100
    f1 = f1_score(actual, pred)
    auc = roc_auc_score(actual, pred)
    print ('Accuracy is {:.4f}%\n\Precision is {:.4f}%\n\Recall is {:.4f}%\n\F1 Score is {:.4f}\n\Area Under Curve is {:.4f}'.
           format(accuracy, precision, recall, f1, auc))

    fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',
    label='AUC = %0.2f'% auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def method3():
    urls = []
    details = dataset.values
    for i in range(len(details)):
        tweets = details[i,14]
        if 'http' in str(tweets):
            urls.append(1)
        else:
            urls.append(0)

    train_attr = dataset[
        ['followers_count', 'friends_count', 'listedcount', 'favourites_count', 'statuses_count', 'verified']]
    train_attr["URLS"] = urls
    train_label = dataset[['bot']]
    X = train_attr
    Y = np.asarray(train_label)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    logreg = LogisticRegression().fit(X_train, y_train)
    actual = y_test
    pred = logreg.predict(X_test)
    accuracy = accuracy_score(actual, pred) * 100
    precision = precision_score(actual, pred) * 100
    recall = recall_score(actual, pred) * 100
    f1 = f1_score(actual, pred)
    auc = roc_auc_score(actual, pred)
    print ('Accuracy is {:.4f}%\n\Precision is {:.4f}%\n\Recall is {:.4f}%\n\F1 Score is {:.4f}\n\Area Under Curve is {:.4f}'.
           format(accuracy, precision, recall, f1, auc))

    fpr, tpr, thresholds = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b',
    label='AUC = %0.2f'% auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

method3()












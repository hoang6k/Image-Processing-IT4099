import pickle
import numpy as np
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

if __name__ == "__main__":
    np.random.seed(0)
    # X_pos = np.load('feature_pos_train.npy')
    # X_neg = np.load('feature_neg_train.npy')
    # len_pos = X_pos.shape[0]
    # len_neg = X_neg.shape[0]
    # X = np.concatenate((X_pos, X_neg))
    # del X_pos, X_neg
    #
    # y_pos = np.ones(len_pos)
    # y_neg = np.zeros(len_neg)
    # y = np.concatenate((y_pos, y_neg))
    # del y_pos, y_neg
    #
    # X, y = shuffle(X, y)

    # clf = MLPClassifier(hidden_layer_sizes=(100, 100,), verbose=True)
    # clf.fit(X, y)
    # pickle.dump(clf, open('model_100_100.sav', 'wb'))
    clf = pickle.load(open('model_400_400.sav', 'rb'))

    # del X, y
    X_pos = np.load('feature_pos_test.npy')
    X_neg = np.load('feature_neg_test.npy')
    len_pos = X_pos.shape[0]
    len_neg = X_neg.shape[0]
    X = np.concatenate((X_pos, X_neg))
    del X_pos, X_neg

    y_pos = np.ones(len_pos)
    y_neg = np.zeros(len_neg)
    y = np.concatenate((y_pos, y_neg))
    del y_pos, y_neg

    try:
        y_pred = clf.predict(X)
    except ValueError:
        pass
    print(classification_report(y, y_pred))

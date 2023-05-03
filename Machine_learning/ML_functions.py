import pandas as pd
from tensorflow import keras
import numpy as np
import xgboost as xgb
from sklearn.svm import SVC
from sklearn import svm
import sklearn.naive_bayes as nb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample


def svm_knn_function(x_train, y_train, x_test, y_test, params):
    '''
    Name: svm_knn_function
    Description: This function implements the method of SVM-KNN
    Inputs:
        x_train: train inputs
        y_train: train targets
        x_test: test inputs
        y_test: test targets
        params: list(k, C)
            k: int, Number of neighbors
            C: float, 'c' regularization parameter
    Outputs: true positives, false negatives, false positives, true negatives
    '''
    def svm_knn_aux(x_train, x_test, y_train, k, c):
        modelo_knn = KNeighborsClassifier(n_neighbors=1, p=1)
        modelo_knn.fit(x_train, y_train)
        neighbors = modelo_knn.kneighbors(
            X=x_test.values.reshape(1, -1), n_neighbors=k, return_distance=True
        )

        # If all the patterns in the subset are of the same class
        if sum(y_train.iloc[neighbors[1][0]]) == 0 or sum(
            y_train.iloc[neighbors[1][0]]
        ) == len(y_train.iloc[neighbors[1][0]]):
            ypred = modelo_knn.predict(x_test.values.reshape(1, -1))

        else:
            modelo_svc = SVC(kernel="linear", C=c)
            modelo_svc.fit(x_train.iloc[neighbors[1][0]], y_train.iloc[neighbors[1][0]])
            ypred = modelo_svc.predict(x_test.values.reshape(1, -1))

        predicted = np.array(ypred)

        return predicted[0]

    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for idx_x_test in range(len(x_test)):
        pred = svm_knn_aux(
            x_train, x_test.iloc[idx_x_test], y_train, params[0], params[1]
        )

        if (np.array(y_test)[idx_x_test]) == 0:
            if pred == 0:
                tp += 1
            else:
                fn += 1
        else:
            if pred == 0:
                fp += 1
            else:
                tn += 1
    return tp, fn, fp, tn


def bagnet_function(X_train, y_train, X_test, y_test, params):
    '''
    Name: bagnet_function
    Description: This function implements the method of BAGNET
    Inputs:
        x_train: train inputs
        y_train: train targets
        x_test: test inputs
        y_test: test targets
        params: neurons
            neurons: list(int), could be [n] or [n,m] where n is de number of neurons of 1st layer
                    and m de number of the 2nd layer
    Outputs: true positives, false negatives, false positives, true negatives
    '''
    def bagnet_aux(x_train, x_test, y_train, y_test, neurons):
        def pred(clf, x, y):
            y_pred = clf.predict(x)
            y_pred_aux = []
            for i in y_pred:
                y_pred_aux.append(int(round(i[0])))

            y = np.array(y)
            y_pred = np.array(y_pred_aux)
            cm = confusion_matrix(y, y_pred)
            acc = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

            return acc

        inp = x_train.shape[1]
        neurons_layer_1 = neurons[0]
        if len(neurons) == 2:
            neurons_layer_2 = neurons[1]

        clf = keras.models.Sequential()
        clf.add(
            keras.layers.Dense(neurons_layer_1, input_dim=inp, activation="sigmoid")
        )
        if len(neurons) == 2:
            clf.add(keras.layers.Dense(neurons_layer_2, activation="sigmoid"))
        clf.add(keras.layers.Dense(1, activation="sigmoid"))
        optimizer = keras.optimizers.Adagrad(lr=0.05)
        clf.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))

        clf.fit(
            x=x_train, y=y_train, epochs=100, verbose=0, class_weight=class_weight_dict
        )

        acc_train = pred(clf, x_train, y_train)

        acc_test = pred(clf, x_test, y_test)

        b_acc = 0.368 * acc_train + 0.632 * acc_test

        return clf, b_acc

    weights = []
    n_times = 30

    test = X_test.iloc[:, 1 : X_test.shape[1]]
    targets = np.zeros(y_test.shape[0])

    for i in range(n_times):
        X_test_i = []
        y_test_i = []
        X_train_i, y_train_i = resample(X_train, y_train, random_state=42 + i)

        for j in range(len(X_train)):
            if X_train.iloc[j, 0] not in X_train_i["index"]:
                X_test_i.append(X_train.iloc[j, :])
                y_test_i.append(y_train.iloc[j])

        X_test_i = pd.DataFrame(X_test_i)
        y_test_i = pd.DataFrame(y_test_i)

        X_train_i = X_train_i.iloc[:, 1 : X_train_i.shape[1]]
        X_test_i = X_test_i.iloc[:, 1 : X_test_i.shape[1]]

        clf, b_acc = bagnet_aux(X_train_i, X_test_i, y_train_i, y_test_i, params)

        y_pred = clf.predict(test)
        y_pred_aux = []
        for i in y_pred:
            y_pred_aux.append(i[0] * b_acc)

        y_pred += np.array(y_pred_aux)
        weights.append(b_acc)

        keras.backend.clear_session()

    y_pred = []
    for i in targets:
        y_pred.append(int(round(i / sum(weights))))

    cm = confusion_matrix(y_test, y_pred)
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]

    return tp, fn, fp, tn


def randomForest_function(x_train, y_train, x_test, y_test, params):
    '''
    Name: randomForest_function
    Description: This function implements the method of Random Forest
    Inputs:
        x_train: train inputs
        y_train: train targets
        x_test: test inputs
        y_test: test targets
        params: list(n_estimators)
            n_estimators: int, The number of trees in the forest
    Outputs: true positives, false negatives, false positives, true negatives
    '''
    clf = RandomForestClassifier(n_jobs=2, random_state=0, n_estimators=params[0])
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    tp, fn, fp, tn = confusion_matrix(y_test, pred)

    return tp, fn, fp, tn


def ann_function(x_train, y_train, x_test, y_test, params):
    '''
        Name: ann_function
        Description: This function implements the method of Artificial Neural Network
        Inputs:
            x_train: train inputs
            y_train: train targets
            x_test: test inputs
            y_test: test targets
            params: list(int), could be [n] or [n,m] where n is de number of neurons of 1st layer
                    and m de number of the 2nd layer
        Outputs: true positives, false negatives, false positives, true negatives
        '''
    iterations = 50
    inp = x_train.shape[1]
    list_tp = []
    list_fn = []
    list_fp = []
    list_tn = []
    neurons_layer_1 = params[0]
    if len(params) == 2:
        neurons_layer_2 = params[1]

    for k in range(iterations):
        modelo = keras.models.Sequential()
        modelo.add(
            keras.layers.Dense(neurons_layer_1, input_dim=inp, activation="sigmoid")
        )
        if len(params) == 2:
            modelo.add(keras.layers.Dense(neurons_layer_2, activation="sigmoid"))
        modelo.add(keras.layers.Dense(1, activation="sigmoid"))
        optimizer = keras.optimizers.Adagrad(lr=0.05)
        modelo.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train), y=y_train
        )
        class_weight_dict = dict(enumerate(class_weights))

        modelo.fit(
            x=x_train,
            y=y_train,
            batch_size=5,
            epochs=10,
            verbose=0,
            class_weight=class_weight_dict,
        )

        pred = modelo.predict(x_test)
        aux = []
        for i in pred:
            aux.append(int(round(i[0])))
        pred = aux
        y_test = np.array(y_test)
        predicted = np.array(pred)

        tp, fn, fp, tn = confusion_matrix(y_test, predicted)
        list_tp.append(tp)
        list_fn.append(fn)
        list_fp.append(fp)
        list_tn.append(tn)
        del modelo
        keras.backend.clear_session()

    return np.mean(list_tp), np.mean(list_fn), np.mean(list_fp), np.mean(list_tn)


def svm_function(x_train, y_train, x_test, y_test, params):
    '''
    Name: svm_function
    Description: This function implements the method of Support Vector Machine
    Inputs:
        x_train: train inputs
        y_train: train targets
        x_test: test inputs
        y_test: test targets
        params: list(kernel, C, param_aux)
            kernel: {'linear', 'rbf', 'poly'} , Specifies the kernel type to be used in the algorithm
            C: float, 'c' regularization parameter
            param_aux:
                if (kernel = 'rbf'): param_aux = gamma. gamma: {‘scale’, ‘auto’} or float, Kernel coefficient
                if (kernel = 'poly'): param_aux = degree. degree: int, Degree of the polynomial kernel function
    Outputs: true positives, false negatives, false positives, true negatives
    '''
    if params[0] == "linear":
        svc = svm.SVC(kernel="linear", C=params[1], gamma="auto").fit(x_train, y_train)
    if params[0] == "rbf":
        svc = svm.SVC(kernel="rbf", gamma=params[2], C=params[1]).fit(x_train, y_train)
    if params[0] == "poly":
        svc = svm.SVC(kernel="poly", degree=params[2], C=params[1], gamma="auto").fit(
            x_train, y_train
        )

    pred = svc.predict(x_test)

    tp, fn, fp, tn = confusion_matrix(y_test, pred)

    return tp, fn, fp, tn


def knn_function(x_train, y_train, x_test, y_test, params):
    '''
    Name: knn_function
    Description: This function implements the method of k-Nearest Neighbors
    Inputs:
        x_train: train inputs
        y_train: train targets
        x_test: test inputs
        y_test: test targets
        params: list(k)
            k: int, Number of neighbors
    Outputs: true positives, false negatives, false positives, true negatives
    '''
    knn = KNeighborsClassifier(
        n_neighbors=params[0], algorithm="auto", p=1, weights="uniform"
    )
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)

    tp, fn, fp, tn = confusion_matrix(y_test, pred)

    return tp, fn, fp, tn


def xgboost_function(x_train, y_train, x_test, y_test, params):
    '''
    Name: xgboost_function
    Description: This function implements the method of XGBoost
    Inputs:
        x_train: train inputs
        y_train: train targets
        x_test: test inputs
        y_test: test targets
        params: list(booster, max_depth, n_estimators, learning_rate)
            booster: {'gbtree', 'dart', 'gblinear'} Specify which booster to use
            max_depth: int, Maximum tree depth for base learners
            n_estimators: int, Number of boosting rounds
            learning_rate: float, Boosting learning rate
    Outputs: true positives, false negatives, false positives, true negatives
    '''
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    param = {"nthread": 4, "objective": "binary:logistic", "verbosity": 0}
    param["booster"] = params[0]
    param["max_depth"] = params[1]
    param["n_estimators"] = params[2]
    param["learning_rate"] = params[3]

    num_round = 10
    bst = xgb.train(param, dtrain, num_round)
    ypred = bst.predict(dtest)

    pred = [round(value) for value in ypred]

    tp, fn, fp, tn = confusion_matrix(y_test, pred)

    return tp, fn, fp, tn


def naive_function(x_train, y_train, x_test, y_test, params):
    '''
    Name: naive_function
    Description: This function implements the method of Naive Bayes
    Inputs:
        x_train: train inputs
        y_train: train targets
        x_test: test inputs
        y_test: test targets
        params: list(type)
            type: {'gaussian', 'multinominal', 'complement', 'bernoulli'}, Algorithm type
    Outputs: true positives, false negatives, false positives, true negatives
    '''
    if params[0] == "gaussian":
        gnb = nb.GaussianNB()
    if params[0] == "multinominal":
        gnb = nb.MultinomialNB()
    if params[0] == "complement":
        gnb = nb.ComplementNB()
    if params[0] == "bernoulli":
        gnb = nb.BernoulliNB()

    pred = gnb.fit(x_train, y_train).predict(x_test)

    tp, fn, fp, tn = confusion_matrix(y_test, pred)

    return tp, fn, fp, tn


def normalizationData_function(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df


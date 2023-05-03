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
from sklearn.model_selection import StratifiedKFold


def _svm_knn_function(x_train, y_train, x_test, y_test, params):
    '''
    Name: _svm_knn_function
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


def _bagnet_function(X_train, y_train, X_test, y_test, params):
    '''
    Name: _bagnet_function
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


def _randomForest_function(x_train, y_train, x_test, y_test, params):
    '''
    Name: _randomForest_function
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


def _ann_function(x_train, y_train, x_test, y_test, params):
    '''
        Name: _ann_function
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


def _svm_function(x_train, y_train, x_test, y_test, params):
    '''
    Name: _svm_function
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


def _knn_function(x_train, y_train, x_test, y_test, params):
    knn = KNeighborsClassifier(
        n_neighbors=params[0], algorithm="auto", p=1, weights="uniform"
    )
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)

    tp, fn, fp, tn = confusion_matrix(y_test, pred)

    return tp, fn, fp, tn


def _xgboost_function(x_train, y_train, x_test, y_test, params):
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    param = {"nthread": 4, "objective": "binary:logistic", "verbosity": 0}
    param["booster"] = params[0]

    if params[0] == "gbtree":
        param = {"max_depth": params[1], "eta": params[2]}

    if params[0] == "dart":
        param["sample_type"] = params[1]
        param["normalize_type"] = params[2]

    if params[0] == "gblinear":
        param["updater"] = params[1]

    num_round = 10
    bst = xgb.train(param, dtrain, num_round)
    ypred = bst.predict(dtest)

    pred = [round(value) for value in ypred]

    tp, fn, fp, tn = confusion_matrix(y_test, pred)

    return tp, fn, fp, tn


def _naive_function(x_train, y_train, x_test, y_test, params):
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


def _cm2dic(dic_confMat, confusion_matrix, algorithm):
    if algorithm not in dic_confMat:
        dic_confMat[algorithm] = []
        dic_confMat[algorithm].append(confusion_matrix)
    else:
        dic_confMat[algorithm].append(confusion_matrix)

    return dic_confMat


def normalizationData_function(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df


def greed_search(X, y, k=10):
    np.random.seed(1)
    dic_confMat = {}

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)

    for train_index, test_index in kf.split(X, y.iloc[:, 1]):
        X_train, X_test = X.iloc[train_index, 0:149], X.iloc[test_index, 0:149]
        y_train, y_test = y.iloc[train_index, 1], y.iloc[test_index, 1]

        X_train_con_idx = X_train
        X_test_con_idx = X_test

        X_train = X_train.iloc[:, 1:149]
        X_test = X_test.iloc[:, 1:149]

        ###########################################################################################################

        confusion_matrix = _randomForest_function(
            X_train, y_train, X_test, y_test, [1000]
        )
        dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "rf")
        confusion_matrix = _ann_function(X_train, y_train, X_test, y_test, [32])
        dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "ann")
        confusion_matrix = _knn_function(X_train, y_train, X_test, y_test, [3])
        dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "knn")
        confusion_matrix = _svm_function(
            X_train, y_train, X_test, y_test, ["rbf", 1, 0.2]
        )
        dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "svm")
        confusion_matrix = _xgboost_function(
            X_train, y_train, X_test, y_test, ["gblinear"]
        )
        dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "xgb")
        confusion_matrix = _naive_function(
            X_train, y_train, X_test, y_test, ["gaussian"]
        )
        dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "nb")

        list_c = [0.1, 0.5, 1, 10]
        for c in list_c:
            for k in range(3, 11, 2):
                confusion_matrix = _svm_knn_function(
                    X_train, y_train, X_test, y_test, [k, c]
                )
                dic_confMat = _cm2dic(
                    dic_confMat, confusion_matrix, "svm_knn_" + str(k) + "_" + str(c)
                )

            for k in range(10, 50, 5):
                confusion_matrix = _svm_knn_function(
                    X_train, y_train, X_test, y_test, [k, c]
                )
                dic_confMat = _cm2dic(
                    dic_confMat, confusion_matrix, "svm_knn_" + str(k) + "_" + str(c)
                )

        list_n1 = [
            [8],
            [10],
            [16],
            [24],
            [32],
            [16, 8],
            [24, 16],
            [32, 16],
            [32, 24],
            [64, 32],
            [128, 32],
            [128, 64],
            [192, 128],
            [256, 192],
        ]
        for n1 in list_n1:
            confusion_matrix = _bagnet_function(
                X_train_con_idx, y_train, X_test_con_idx, y_test, n1
            )
            dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "bagnet_" + str(n1))

    output = np.zeros(shape=(len(dic_confMat), 41))
    fila = 0
    for clave in dic_confMat:
        output[fila, 0] = "".join(str(ord(c)) for c in clave)
        idx_fold = 0
        for fold in dic_confMat[clave]:
            output[fila, 1 + idx_fold] = fold[0]
            output[fila, 2 + idx_fold] = fold[1]
            output[fila, 3 + idx_fold] = fold[2]
            output[fila, 4 + idx_fold] = fold[3]
            idx_fold = idx_fold + 4

        fila = fila + 1

    return output

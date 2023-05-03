from sklearn.model_selection import StratifiedKFold
import numpy as np
import ML_functions as mlFunc


def _cm2dic(dic_confMat, confusion_matrix, algorithm):
    if algorithm not in dic_confMat:
        dic_confMat[algorithm] = []
        dic_confMat[algorithm].append(confusion_matrix)
    else:
        dic_confMat[algorithm].append(confusion_matrix)

    return dic_confMat


def greed_search(X, y, k=10):
    '''
    Name: greed_search
    Description: This function implements main code for ML models comparation
    Inputs:
        X: inputs
        y: targets
        k: int(default = 10), number of folds of StratifiedKFold
    Outputs: test results
    '''
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

        confusion_matrix = mlFunc.randomForest_function(
            X_train, y_train, X_test, y_test, [1000]
        )
        dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "rf")
        confusion_matrix = mlFunc.ann_function(X_train, y_train, X_test, y_test, [32])
        dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "ann")
        confusion_matrix = mlFunc.knn_function(X_train, y_train, X_test, y_test, [3])
        dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "knn")
        confusion_matrix = mlFunc.svm_function(
            X_train, y_train, X_test, y_test, ["rbf", 1, 0.2]
        )
        dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "svm")
        confusion_matrix = mlFunc.xgboost_function(
            X_train, y_train, X_test, y_test, ["gblinear"]
        )
        dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "xgb")
        confusion_matrix = mlFunc.naive_function(
            X_train, y_train, X_test, y_test, ["gaussian"]
        )
        dic_confMat = _cm2dic(dic_confMat, confusion_matrix, "nb")

        list_c = [0.1, 0.5, 1, 10]
        for c in list_c:
            for k in range(3, 11, 2):
                confusion_matrix = mlFunc.svm_knn_function(
                    X_train, y_train, X_test, y_test, [k, c]
                )
                dic_confMat = _cm2dic(
                    dic_confMat, confusion_matrix, "svm_knn_" + str(k) + "_" + str(c)
                )

            for k in range(10, 50, 5):
                confusion_matrix = mlFunc.svm_knn_function(
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
            confusion_matrix = mlFunc.bagnet_function(
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
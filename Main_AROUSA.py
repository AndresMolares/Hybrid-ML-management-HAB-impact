import pandas as pd
import numpy as np
from Machine_learning.ML_comparation import greed_search
from Machine_learning.ML_functions import normalizationData_function


archivo = "RESULTS_AROUSA_DSP.csv"

zona = "./inputs_AROUSA.csv"
df_original = pd.read_csv(zona, sep=";")


df_aux = pd.concat(
    [
        df_original.iloc[:, 0:95],
        df_original.iloc[:, 95 : 95 + 40],
        df_original.iloc[:, 95 + 40 + 20 + 10 : 95 + 40 + 20 + 10 + 51],
        df_original["DSP Viernes"],
        df_original["DSP"],
    ],
    axis=1,
)

df = normalizationData_function(df_aux)

aux = pd.concat([df.iloc[:, 0 : df.shape[1] - 1], df.iloc[:, -1]], axis=1)
aux.dropna(inplace=True)


X = aux.iloc[:, 0 : df.shape[1] - 1].reset_index()
y = aux.iloc[:, -1].reset_index()

###########################################################################################################

output = greed_search(X, y, 10)

np.savetxt(archivo, output, delimiter=";")

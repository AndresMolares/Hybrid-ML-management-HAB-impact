import pandas as pd
import numpy as np
from Machine_learning.ML_comparation import greed_search
from Machine_learning.ML_functions import normalizationData_function


archivo = "RESULTS_MUROS_DSP.csv"

zona = "./inputs_MUROS.csv"
df_original = pd.read_csv(zona, sep=";")


df_aux = pd.concat(
    [
        df_original.iloc[:, 0:54],
        df_original.iloc[:, 54 : 54 + 28],
        df_original.iloc[:, 54 + 28 + 14 + 7 : 54 + 28 + 14 + 7 + 36],
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

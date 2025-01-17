import pandas as pd
import numpy as np
import mlflow
from sklearn.datasets import make_regression
import json
from tabulate import tabulate

import mlflow
import os
import pickle


def main():
    print("Carregando o modelo...")
    # Caminho para o modelo local
    model_pkl_path = "../model/model.pkl"

    # Carregar o modelo
    with open(model_pkl_path, "rb") as f:
        loaded_model = pickle.load(f)

    print()
    print("Carregando o banco de dados de teste final...")
    df = pd.read_csv("../data/Abandono_teste.csv", sep=";")
    df_test_final = df.copy()
    df_test_final.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True) 

    print()
    print("Prevendo o resultado...")
    df_test_final['predictedValues'] = loaded_model.predict(df_test_final)

    df_test_final['Proba_0'] = loaded_model.predict_proba(df_test_final)[:,0]
    df_test_final['Proba_1'] = loaded_model.predict_proba(df_test_final)[:,1]

    print()
    print("Criando o arquivo final com as predições...")
    Abandono_clientes_final = pd.DataFrame({'RowNumber':df['RowNumber'],
                'predictedValues': df_test_final['predictedValues']})

    Abandono_clientes_final.to_csv('../data/Abandono_clientes_final.csv', index=False)

    print("Fim")

if __name__ == '__main__':
    main()
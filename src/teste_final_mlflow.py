import pandas as pd
import numpy as np
import mlflow
from sklearn.datasets import make_regression
import json
from tabulate import tabulate

def main():
    print("Carregando o modelo...")
    mlflow.set_tracking_uri("http://127.0.0.1:8081/")
    loaded_model =  mlflow.sklearn.load_model("models:/Churn-Abandono/production")
    
    print()
    print("Carregando as features...")
    model_info = mlflow.models.get_model_info("models:/Churn-Abandono/production")
    features = [i['name'] for i in json.loads(model_info.signature_dict['inputs'])]

    print()
    print("Carregando o banco de dados de teste final...")
    df = pd.read_csv("../data/Abandono_teste.csv", sep=";")
    df_test_final = df.copy()
    df_test_final.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True) 

    print()
    print("Prevendo o resultado...")
    df_test_final['predictedValues'] = loaded_model.predict(df_test_final[features])

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
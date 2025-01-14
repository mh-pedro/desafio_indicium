# Manipulação de dados
#%%
import pandas as pd
import numpy as np 

# Visualização
import seaborn as sns 
import matplotlib.pyplot as plt 
from IPython.display import HTML 
import plotly.graph_objects as go 
from pprint import pprint

# Algumas configurações
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Configuração para o notebook e plotagem de imagens
def jupyter_settings():
    %matplotlib inline
    plt.style.use('bmh')
    plt.rcParams['figure.figsize'] = [25, 12]
    plt.rcParams['font.size'] = 24
    display(HTML('<style>.container { width:100% !important; }</style>'))
    sns.set()

jupyter_settings()


# Machine Learning

# Processamento dos dados
from sklearn.model_selection import train_test_split , StratifiedKFold # Divisão dos dados em treino e teste
#from scipy import interp
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler # Codifica variáveis categóricas em valores numéricos.

# LabelEncoder, OneHotEncoder ->Codifica variáveis categóricas em valores numéricos.
# O primeiro identifica em número inteiro único e não altera a dimensionalidade, mas pode introduzir ordem artificial
# O segunda cada categoria é uma coluna binária e aumenta a dimensionalidade, mas não há ordem artificial
# Normaliza os dados numéricos para facilitar o treinamento dos modelos.
# RobustScaler ->
# MinMaxScaler -> 
from sklearn.compose import ColumnTransformer # Permite aplicar transformações diferentes para diferentes colunas
from sklearn.pipeline import Pipeline # Permite combinar múltiplas etapas do processo de machine learning


# Modelos de Machine Learning:
from sklearn.tree import DecisionTreeClassifier # Importa o classificador baseado em árvores de decisão.
from sklearn.ensemble import RandomForestClassifier # Importa o classificador baseado em florestas aleatórias.
from sklearn.naive_bayes import GaussianNB #  Importa o classificador Naive Bayes Gaussiano.
from sklearn.neighbors import KNeighborsClassifier # Importa o classificador K-Nearest Neighbors (KNN).
from sklearn.svm import SVC # Importa o classificador de Máquinas de Vetores de Suporte (SVM).
from sklearn.neural_network import MLPClassifier # Importa o classificador baseado em redes neurais artificiais (Perceptron Multicamadas).
from sklearn.ensemble import AdaBoostClassifier # Importa o classificador baseado em AdaBoost.
from sklearn.ensemble import GradientBoostingClassifier # Importa o classificador baseado em Gradient Boosting.
from sklearn.ensemble import ExtraTreesClassifier # Importa o classificador baseado em Árvores Extras (Extra Trees).
from xgboost import XGBClassifier #  Importa o classificador baseado no XGBoost, um algoritmo de Gradient Boosting otimizado.
from sklearn.linear_model import LogisticRegression # Importa o modelo de Regressão Logística.

# Métricas de desempenho
# Calcula a taxa de recuperação (recall).
# Calcula a curva ROC para análise de desempenho.
# Gera a matriz de confusão.
# Calcula a precisão do modelo
# Calcula a métrica F1, que é a média harmônica entre precisão e recall.
# Calcula a acurácia do modelo.
# Gera um relatório detalhado com precisão, recall e F1-score para cada classe.
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, auc, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

# Importanto o mlflow
import mlflow 
from mlflow.models.signature import infer_signature


# Importando os dados

df_raw = pd.read_csv('../data/Abandono_clientes.csv', sep=',')
# Vamos fazer uma copia do dataset original

train_transform = df_raw.copy()
train_transform.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True) 

# Separando as variáveis dependentes e independentes
y = train_transform['Exited']
X = train_transform.drop('Exited', axis=1) 

# Split dos dados em treino e teste
X_train, X_rest1, y_train, y_rest1 = train_test_split(X, y, train_size=7000, random_state=2)

# Dos dados que sobraram, separando em dados de teste
X_test, X_rest2, y_test, y_rest2 = train_test_split(X_rest1, y_rest1, train_size=1000, random_state=42)

# Dos dados que sobraram, separando em calibração e new
X_calib, X_new, y_calib, y_new = train_test_split(X_rest2, y_rest2, train_size=1000, random_state=42)

# Variáveis numéricas categóricas
categorical_numeric_features = ['HasCrCard', 'IsActiveMember', 'NumOfProducts']

# Variáveis numéricas contínuas
numeric_features = [col for col in X_train.select_dtypes(include=np.number).columns if col not in categorical_numeric_features]

# Variáveis para o RobustScaler
robust_scaler_features = ['Age', 'CreditScore', 'NumOfProducts']

# Variáveis para StandardScaler
standard_scaler_features = [col for col in numeric_features if col not in robust_scaler_features]


# Variáveis categóricas
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist() + categorical_numeric_features

# Transformadores
numeric_transformer_standard = StandardScaler()
numeric_transformer_robust = RobustScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# # Criação do pipeline para conversão

preprocesso = ColumnTransformer(transformers=[
    ('std_num', numeric_transformer_standard, standard_scaler_features),
    ('robust', numeric_transformer_robust, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])


# Função para reportar as métricas
def report_metrics(y_true, y_proba,base, cohort=0.5):
    y_pred = (y_proba[:,1] > cohort).astype(int)

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba[:,1])
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
   
    res = {
        f'{base} accuracy': acc,
        f'{base} Curva Roc': auc,
        f'{base} precision': precision,
        f'{base} recall': recall,
        f'{base} f1': f1
            }
    return res

# Conectando ao mlflow
mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
mlflow.set_experiment(experiment_id="829861969097252139")

#%%
mlflow.autolog(disable=True)

with mlflow.start_run(run_name="RandomForest_Hyperparameter_Search"):
    # definindo um pipeline
    pipeline = Pipeline(steps=[('preprocesso', preprocesso),
                            ('clf', RandomForestClassifier(random_state=42))])

    # Critério de separação 
    criterio = ['gini', 'entropy']
    # Queremos medir a qualidade da separação em cada nó
    # gini -> Medição da impureza
    # entropy -> mede a incerteza da separação

    # Núm de arvores 
    n_estimators = [200, 400, 600, 800, 1000, 1100, 1200, 1400, 1600, 1800, 2000]
    # O uso de mais árvores ajuda a estabilidade

    # Núm de variáveis em cada separação
    max_features = ['auto', 'sqrt']
    # auto usa todas as variáveis por padrão em classificadores
    # sqrt usa a raiz quadrada do num total de variáveis
    # O papel e reduzir o num de variáveis, diminuindo o overfitting

    # Núm max de nível da árvore
    max_depth = [10, 20, 30, 40, 60, 80, 90, 100, None]
    # Controla a profundidade máxima das árvores. Obs. Árvores mais profundas capturam mais detalhes, mas podem causar overfitting

    # Mín de amostras para dividir um nó
    min_samples_split = [2, 5, 10]
    # valores baixos permite divisões mais frequentes o que pode causar overfitting
    # já valores autos limitam as divisões, tornando o modelo mais simples

    # Min de amostra por fola
    min_samples_leaf = [1, 2, 4]
    # 1 Permite folhas individuais (mais propenso a overfitting).
    #Valores mais altos: Tornam o modelo mais simples e robusto.

    # Mét de seleção de amostras
    bootstrap = [True, False]
    # True: Usa amostragem com substituição (metodologia padrão de Random Forest).
    # False: Usa todo o dataset sem substituição.

    # Peso das classes 
    class_weight = ['balanced', 'balanced_subsample', None]
    # balanced: Calcula pesos automaticamente com base na proporção inversa das classes.
    # balanced_subsample: Calcula pesos para cada árvore com base em sua amostra bootstrap.
    # None: Não ajusta pesos.

    grid_params_rf = [{'clf__criterion': criterio,
                    'clf__n_estimators': n_estimators,
                    'clf__max_features': max_features,
                    'clf__max_depth': max_depth,
                    'clf__min_samples_split': min_samples_split,
                    'clf__min_samples_leaf': min_samples_leaf,
                    'clf__bootstrap': bootstrap,
                    'clf__class_weight': class_weight
                    }]

    pprint(grid_params_rf)


    # definido os parâmetros para o RandomizedSearchCV
    RandomSearch = RandomizedSearchCV(estimator=pipeline, 
                                        param_distributions=grid_params_rf, 
                                        n_iter = 100, 
                                        cv = 3, 
                                        verbose=2, 
                                        random_state=42, 
                                        n_jobs = -1)
    
    # treinando o modelo randômico
    RandomSearch.fit(X_train, y_train)
 
    # Obtendo o melhor modelo, parâmetros e métricas
    best_model = RandomSearch.best_estimator_
    best_params = RandomSearch.best_params_
    train_score = RandomSearch.score(X_train, y_train)
    test_score = RandomSearch.score(X_test, y_test)

    # Exemplo de entrada (X_train) e saída (previsão do modelo no X_train)  
    input_example = X_train.head(1)  # Um exemplo do conjunto de dados de entrada
    predictions_example = best_model.predict(X_train.head(1))  # Previsão com o modelo treinado

    # Gerando a assinatura com base no exemplo de entrada e saída
    signature = infer_signature(X_train, predictions_example)
    
    # aplicando o modelo em diferentes bases de dados
    y_train_proba = best_model.predict_proba(X_train)
    y_test_proba = best_model.predict_proba(X_test)

    report = {}
    report.update(report_metrics(y_train, y_train_proba, 'train'))
    report.update(report_metrics(y_test, y_test_proba, 'test'))
    

    # Log dos parâmetros e métricas no MLflow
    mlflow.log_metrics(report)
    mlflow.sklearn.log_model(sk_model = best_model, artifact_path ="random_forest_model_Hyperp",signature = signature, input_example=input_example)  # Salva o modelo
    mlflow.log_params(best_params)  # Log dos melhores parâmetros
 


#%%
with mlflow.start_run(run_name="RandomForest_com_todos_os_dados"):
    mlflow.autolog()
    # Modelo treinado com os melhores hiperparâmetros e todo os dados
    modelo_final = best_model.fit(X, y)

   # Exemplo de entrada (X_train) e saída (previsão do modelo no X_train)  
    input_example = X.head(1)  # Um exemplo do conjunto de dados de entrada
    predictions_example = best_model.predict(X.head(1))  # Previsão com o modelo treinado

    # Gerando a assinatura com base no exemplo de entrada e saída
    signature = infer_signature(X, predictions_example)
    
    # aplicando o modelo em diferentes bases de dados
    y_train_proba = best_model.predict_proba(X)
    report = {}
    report.update(report_metrics(y, y_train_proba, 'train')) 

    # Log dos parâmetros e métricas no MLflow
    mlflow.log_metrics(report)
    mlflow.sklearn.log_model(sk_model = best_model, artifact_path ="random_forest_model_total",signature = signature, input_example=input_example)  # Salva o modelo

# %%

# Desafio Data Science

## Resumo

Olá candidato, o objetivo deste desafio é testar os seus conhecimentos sobre construção de modelos preditivos. Queremos testar os seus conhecimentos dos conceitos estatísticos de modelos preditivos, criatividade na resolução de problemas e aplicação de modelos de machine learning em produção. É importante deixar claro que não existe resposta certa e que nos interessa é sua capacidade de descrever e justificar os passos utilizados na resolução do problema.

## Descrição do Problema
 
O seu objetivo é prever o *churn* (abandono de clientes) de um banco de dados fictício de uma instituição financeira. Para isso são fornecidos dois *datasets*: um *dataset* chamado **Abandono_clientes** composto por 10000 linhas e 13 colunas de informação (*features*), sendo uma coluna “Exited” composta por dados binários: 1 se o cliente abandonou o banco, 0 se não. O segundo *dataset* possui 1000 linhas e 12 colunas e não possui a coluna “Exited”. O seu objetivo é construir um pipeline de Machine Learning que permita prever essa coluna a partir dos dados enviados.** 

## Atividades

1. Descreva graficamente os dados disponíveis, apresentando as principais estatísticas descritivas. Comente o porquê da escolha dessas estatísticas.

2. Explique como você faria a previsão do **Churn** a partir dos dados. Quais variáveis e/ou as suas transformações você utilizou e por quê? Qual tipo de problema estamos a resolvendo (regressão, classificação)? Qual modelo melhor se aproxima dos dados e quais os seus prós e contras? Qual medida de desempenho do modelo foi escolhida e por quê?

3. Construa um pipeline de Machine Learning que realize a previsão de churn a partir de um dataset CSV. Esse pipeline deve ser reproduzível e permitir realizar uma previsão a partir de qualquer arquivo CSV com a mesma estrutura de dados. O modelo utilizado pelo pipeline deve ser treinado com o dataset **Abandono_clientes.csv**. Você deve nos enviar o repositório de código com o pipeline.

4. Envie o resultado do modelo num arquivo com apenas duas colunas (rowNumber, predictedValues) gerado ao rodar o pipeline de 3 no dataset **abandono_teste.csv** em anexo.


## Introdução: O que é um Churn e para que serve?

É muito comum empresas enfrentarem problemas com perdas de clientes e/ou receitas, para quantificar o número de clientes perdidos podemos usar uma métrica denominada **churn rate**.

No mundo atual, temos inúmeros modelos de negócios por assinaturas (Netflix, Amazon, Spotify, etc.), poder identificar se o cliente pode ou não cancelar a assinatura tornou-se um problema de negócios. O Churn entendido como um índice de cancelamentos de clientes que cancelam em determinado período. Para podermos calcular o Churn, o que precisamos fazer é somar o número de clientes que cancelou o seu produto/serviço no período analisado. 

Desta forma, se uma empresa deseja fazer uma expansão da base de clientes, é preciso que o número de novos clientes exceda o seu churn rate, melhor dizendo, a adesão de novos clientes deve ser maior do que a taxa de cancelamentos.

### Qual deve ser a taxa ideal de Churn?

Entendido, o que é um churn e qual o seu papel, uma questão que surge é: **Qual deve ser a taxa ideal de Churn?**

Podemos dizer que a melhor taxa de Churn seria 0%, pois isso significa que não temos clientes realizando cancelamento, contudo, no mundo real isto não é possível. Para entendermos qual deve ser o valor ideal, precisamos entender o seguimento de mercado que estamos envolvidos.

Não existe um valor exato para a taxa de Churn, pois podemos ter diversos fatores que influenciam no Churn, no entanto, de forma geral, pode-se dizer que *5% é uma taxa aceitável** para empresas que trabalham com recorrência, de acordo com [David Pakman](https://pakman.com/churn-is-the-single-metric-that-determines-the-success-of-your-subscription-service-6e82d9d9ea01), cocriador do Music Group da Apple.

### O que levam os clientes a cancelar?

Entender os fatores que levam os clientes a cancelar e os fatores que levam os clientes a aderir ao negócio pode ser uma tarefa difícil, pois muitos fatores podem ser combinados para chegar a uma conclusão.

No entanto, alguns fatores que levam os clientes a cancelar podem ser identificados:

- O cliente está sem fluxo de caixa e não pode mais arcar com a mensalidade do seu produto/serviço;
- Ele não consegue ver valor no produto/serviço;
- O cliente não teve a suas expectativas atendidas;
- O produto ou serviço não acompanha as evoluções de mercado e perde em - qualidade e ferramentas;
- O produto é bom, mas o serviço não – e vice-versa;
- O Seu cliente optou pelo produto da concorrência;
- O Seu cliente foi adquirido por outra empresa e o comprador usa outro serviço;
- Crise financeira no mercado;
- Interrupção da operação durante um período de crise;
- O Seu cliente faliu.

Portanto, para um negócio ser saudável é importante conhecer os indicativos de uso do produto, qualidade do serviço, preço, competição, etc.

## Planejamento do Desafio

Vamos aplicar o método CRISP-DS que é uma variável do método CRISP-DM (Cross-individual System Process for Data Mining) adaptando para o cenário de ciência de dados.

O método CRIS-DS consiste pode ser visualizado na figura abaixo:

<p align="center" width="100%">
    <img width="80%" src="./imagens/CRISP_DS.png">
</p>


O método CRISP-DS consiste em:

1. **Identificar o problema de negócio:** 
    - Definir os objetivos e requisitos do negócio;
    - Identificar os problemas que precisam ser resolvidos e como a análise de dados pode ajudar;
    - Formular perguntas de negócios e traduzi-lás em problemas de dados.


2. **Compreensão dos Dados:**
    - Explorar e entender as características principais dos dados.
    - Identificar padrões, tendências e outliers que podem influenciar o processo de modelagem.
    - Garantir que os dados disponíveis estão alinhados com o problema de negócio definido na fase anterior.
    - Envolve sumarizar e visualizar os dados usando estatísticas básicas (média, mediana, desvio padrão) e gráficos (histogramas, boxplots, scatterplots, etc.).
    - Ajuda a entender como as variáveis estão distribuídas, detectar relações entre elas e identificar inconsistências ou dados faltantes.

3. **Preparação dos Dados para modelagem:**
    - Limpar os dados e preencher os valores ausentes;
    - Transformar os dados para melhorar a qualidade dos dados;
    - Normalizar os dados para melhorar a qualidade dos dados;
    - Organizar os dados para melhorar a qualidade dos dados;
    - Dividir os dados em conjuntos de treino e teste;
    - Avaliar a qualidade dos dados;

4. **Modelagem:**
    - Selecionar técnicas de aprendizado de máquina;
    - Construir, treinar e ajustar modelos preditivos ou descritivos;
    - Testas diferentes configurações para encontrar a melhor abordagem;

5. **Avaliação:** 
    - Avaliar o desempenho do modelo usando métricas apropriadas;
    - Verificar se o modelo atende aos objetivos de negócio definidos na Fase 1;
    - Ajustar o modelo, se necessário, ou redefinir o problema de dados.

6. **Aplicação:**
    - Implementar o modelo num ambiente de produção;
    - Apresentar os resultados;


## Projeto 

### Parte 1 - Entendimento de negócio

Como mencionado no início do documento, estamos a tratar de um problema de **Churn**, ou seja, o objetivo principal em problemas como este é determinar se um cliente irá **sair** ou **manter-se**. Isso envolve prever uma variável categórica binária, o que nos levar a tratar como um problema de **classificação binária**. Portanto, queremos criar um modelo preditivo capaz de prever se um cliente vai abandonar o banco ou continuar com ele.

### Parte 2 - Compreensão dos Dados:

Os dados foram fornecidos pelo idealizado do desafio e simula uma empresa de banco fictícia. Esses dados apresentam as seguintes características descritas na tabela:

| Variável           | Significado                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| RowNumber          | Número da linha no conjunto de dados, usado apenas como identificador.     |
| CustomerId         | Identificador único do cliente.                                            |
| Surname            | Sobrenome do cliente.                                                     |
| CreditScore        | Pontuação de crédito do cliente.                                           |
| Geography          | País ou região de origem do cliente.                                       |
| Gender             | Gênero do cliente (masculino ou feminino).                                 |
| Age                | Idade do cliente.                                                         |
| Tenure             | Número de anos que o cliente tem como cliente do banco.                   |
| Balance            | Saldo atual na conta bancária do cliente.                                  |
| NumOfProducts      | Número de produtos bancários que o cliente utiliza.                       |
| HasCrCard          | Indica se o cliente possui um cartão de crédito (1 = sim, 0 = não).        |
| IsActiveMember     | Indica se o cliente é um membro ativo do banco (1 = sim, 0 = não).         |
| EstimatedSalary    | Salário estimado do cliente.                                               |
| Exited             | Indica se o cliente saiu do banco (1 = sim, 0 = não).                      |

A variável target (alvo) será a variável **Exited**. A nossa tarefa consiste em prever se um cliente vai abandonar o banco ou continuar com ele.

Durante a compreensão dos dados identificamos que não há dados ausentes, a tipologia dos dados estava condizente com o tipo de dados, em seguida realizamos algumas análises descritivas que resultaram nas seguintes observações:

 #### Analise descritiva e resultados

 Iniciamos separamos as variáveis numéricas e categóricas. 

 Para as variáveis categóricas, quando olhamos para a quantidade e a $\%$ do total, temos as seguintes distribuições:

<div style="text-align: center;">
  <img src="./imagens/clientes_por_variavel_valor_total.png" alt="figura 2" />
</div>

Podemos ver que em valores totais, o número de clientes que cancelam é sempre menor comparado com os clientes que não cancelam, contudo para termos uma melhor visão, devemos olhar também para a taxa em termos de proporções.

Seguimos em diante e olhamos para os valores proporcionais, como podemos ver abaixo:

<div style="text-align: center;">
  <img src="./imagens/clientes_por_variavel_valor_proporcional.png" alt="figura 3" />
</div>

Se comparamos os gráficos por valores totais e por proporção, temos:

- No caso de clientes por Gênero, a taxa de cancelamento por valores totais é de $11.4\%$ enquanto em valores proporcionais chega a $25\%$. Obs.: É necessário tomarmos cuidado quando realizarmos analises para não reforçamos preconceitos e não vir ser um viés na análise preditiva. Note que a variável categórica **Gênero** nos indica que proporcionalmente, mulheres tem a maior taxa de cancelamento, este resultado não deve ser usado com o intuito de impedir o acesso delas ao banco, mas sim de entender o motivo da evasão.

- Para clientes que possuem cartão de crédito, em valores totais é possível ver que os clientes que não cancelam é bem maior comparado aos que cancelam, contudo quando olhamos proporcionalmente, a taxa de cancelamento para clientes que possuem ou não possuem cartão é quase o mesmo valor. Este resultado pode nos indicar que esta variável categórica não terá tanto papel na identificação dos usuários durante a predição. 

- Por fim, clientes que não são membros possuem cancelam mais os planos comparados com os que são membros.

Seguimos adiante com a análise dos dados numéricos, notamos que para a maioria dos dados, a média e a mediana são similares, com exceção do atributo Balance, que apresenta uma diferença mais significativa.

Com o intuito de obter uma melhor visão dos dados numéricos, plotamos as distribuições de cada variável numérica e avaliamos os comportamentos.

<div style="text-align: center;">
  <img src="./imagens/densidade_das_variaveis.png" alt="figura 4" />
</div>

Nesta etapa olhamos para distribuição das variáveis numéricas, algumas conclusões podem ser obtidas:

- Os gráficos de **Tenure** e **EstimatedSalary** apresentam um topo largo, indicando que os valores dessas variáveis estão mais **uniformemente distribuídos** dentro de um certo intervalo. Esta informação indica que não temos um único valor ou faixa de valores dominante. Quando olhamos com o objetivo de prever o **Churn**, tanto o **Tenure** e **EstimatedSalary** não parecem ser um fator decisivo para o cliente sair ou permanecer.

- Para os gráficos **Age** e **NumOfProducts** mostram picos estreitos, indicando que há faixas específicas de valores com maior concentração de clientes (com idades específicas ou números de produtos). Estas duas variáveis podem ser uteis na análise preditiva. 

- Para o gráfico **CreditScore** temos uma distribuição simétrica, indicando que os dados estão próximos de uma distribuição normal. Indicando que a maioria dos clientes tem pontuações de crédito numa faixa considerada comum (média e boa). A simetria também indica que não temos uma forte concentração em extremos, como clientes com pontuações muito baixa ou muito alta. Esta variável é importante na análise preditiva, pois clientes com baixo score tem mais dificuldades de acessar serviços ou produtos financeiros, tornando-os mais propensos a abandonar o banco e clientes com altas pontuações podem ser alvos de concorrentes oferecendo melhores condições. Obs.: Por ser um grupo com distribuição simétrica, não será necessário um tratamento tão intenso de outliers.

- No gráfico **Balance** temos dois picos principais:
    - Um pico em **0**, provavelmente clientes sem saldo, com contas inativas ou que utilizam o banco apenas para movimentações pontuais;
    - Um segundo pico numa faixa de saldo mais alta (próxima de 100.000 a 150.000), indicando que estes clientes provavelmente possuem uma conta de poupança ou investimento.
Temos então um comportamento bimodal, sugerindo que o banco atende dois grupos principais, essa segmentação pode indicar que o banco precisa de estratégias para atrair e engajar os dois grupos:
    - **Clientes com saldo baixo:** Ofertas de incentivos para movimentação e engajamento.
    - **Clientes com saldo alto:** Ofertas de produtos premium ou investimentos.
Durante a análise preditiva, provavelmente deveremos tratar esses dois segmentos como classes separadas, assim um não mascara o outro. 

Continuando, os outliers podem influenciar nas análises e no tratamento do modelo preditivo, portanto, vamos buscar identificar os outliers e seus comportamento, podemos ver os gráficos abaixo:

<div style="text-align: center;">
  <img src="./imagens/outlier_sim.png" alt="figura 5" />
  <img src="./imagens/outlier_nao.png" alt="figura 6" />
</div>

Podemos observar que para boa parte dos variáveis, não temos muitos outliers, contudo a variável **Age** é a variável que apresenta mais Outliers. Por hora, não iremos tratar os outliers, iremos treinar o modelo com esses outliers e veremos qual o comportamento do modelo para esses outliers, podemos voltar aqui se necessário e avaliar como o modelo fica sem os outliers.

Outro ponto é que notamos é a média e a media são bastante semelhantes como já havíamos comentado, apenas a variável **Balance** que possui um ligeira diferenciação.

Por fim, analisamos também as correlações entre todas as variáveis, o resultado pode ser visto na figura abaixo:

<div style="text-align: center;">
  <img src="./imagens/correlacao.png" alt="figura 7" />
</div>

Podemos notar que os pares de variáveis com maior correlação são:

- **(Balance, Geography)**
- **(NumOfProducts, Balance)**
- **(Exited, Age)**

É importante ressaltar que aqui não diferenciamos homens de mulheres, talvez cabe separa esses dois e ver se existe uma correlação maior se considerarmos apenas mulheres ou homens. O mesmo vale para variável **Geography**.

## Parte 3 - Preparação dos Dados para modelagem:

Após ganhado entendimento dos dados, vamos iniciar nossa fase de preparação dos dados para modelagem. É agora que vamos remover os dados que não são necessários, limpeza, normalização e organização dos dados.

Temos dois bancos de dados, o banco de treino **Abandono_clientes.csv** e **Abandono_teste.csv**. Durante toda essa fase iremos trabalhar apenas com o primeiro banco.

Iniciamos realizando a separação dos dados da seguinte forma:

- 7000 amostras de dados ("`X_train`", "`y_train`") para treinar o modelo;
- 1000 amostras de dados ("`X_test`", "`y_test`") para avaliar o desempenho do modelo em dados "não vistos" no treinamento;
- 500 amostras de dados ("`X_calib`"`y_calib`") para calibração em dados "não vistos" no treinamento;
- 1000 amostras restantes ("`X_new`"`y_new`") para etapa de previsão conforme e para sua avaliação.

O intuito de separamos o banco em quatro amostras é termos um conjunto de treino, um de teste, e dois outros conjuntos que podem ser usados para calibragem. Inicialmente essa será nossa distribuição, a depender dos resultados, podemos voltar e alterar estes valores

### Parte 3.1 - Baseline

O intuito de criarmos uma baseline é para termos um ponto de partida ao avaliar o desempenho do nosso modelo. Iremos utilizar uma métrica que faz uso de Classificador de taxa aleatória (estimativa ponderada), para ver mais acesse o [artigo](https://towardsdatascience.com/calculating-a-baseline-accuracy-for-a-classification-model-a4b342ceb88f). Desta forma teremos a ZeroR baseline que a taxa mais frequente e a Random Rate (Baseline Ponderada) que é construída a seguir.

- Proporção da Classe Minoritária (Exited = 1): $0.2074$
- Proporção da Classe Majoritária (Exited = 0): $0.7926$

- Odds de Adivinhar Corretamente a Classe Minoritária: $0.20742857142857143^2 = 0.0430$
- Odds de Adivinhar Corretamente a Classe Majoritária: $0.7925714285714286^2 = 0.6282$

- Baseline Ponderado: $0.6712$ ou $67.12\%$

Note que nossa Baseline Ponderada é de $67,12\%$ e a ZeroR Baseline é de $79,26\%$ estes valores serão os valores de avaliação inicial.

### Parte 3.2 - Transformação nos dados

Vamos fazer a separação das variáveis numéricas e categóricas. Decidi adicionar `HasCrCard`, `IsActiveMember` como variáveis categóricas, mesmo sendo números. No futuro posso voltar aqui e rever esta decisão. 

As transformações que vamos fazer são:

- **StandardScaler** para padronizar os valores, inicialmente não irei tratar os outliers, desejo ver como os modelos atual considerando esses valores, também não irei o min-max scaler para os valores que não possuem outliers.
- **OneHotEncoder** para transformar as variáveis categóricas do tipo `Gender` e `Geography` e valores binários. 
- **RobustScaler** para tratar os outliers.

Concluímos aqui a nossa fase de preparação dos dados para modelagem, vamos iniciar a fase 4.

## Parte 4 - Modelagem

Iremos utilizar alguns dos mais comuns modelos de aprendizado de máquina, são eles:

- **LogisticRegression:** Um classificador de regressão logística, geralmente utilizado para problemas de classificação binaria.

- **DECISION TREE:** Baseado em divisões recursivas de dados em nós.
- **KNeighbors:** Classifica pontos com base nos vizinhos mais próximos.

- **SCV - Support Vector Classifier :** 
- **Random Forest:** Combina várias árvores de decisão (ensemble) para reduzir overfitting

- **XGB:** Algoritmo otimizado de gradient boosting.

- **AdaBoost:** Modelo ensemble que ajusta o peso de classificadores fracos.
- **ExtraTrees:** Variante do Random Forest com maior aleatoriedade na seleção de divisões.
- **GradientBoosting:** Modelo ensemble que constrói árvores sequencialmente para corrigir erros.

Iremos identificar quais aqueles que estão acima da nossa baseline, posteriormente, iremos selecionar os 3 melhores comparando a acurácia e a F1-Score entre eles.

Realizamos esta fase inicialmente usando apenas duas transformações: StandardScaler e OneHotEncoder. Posteriormente, usando o RobustScaler.  Adicionar o RobustScaler para tratar os outliers não mudou significativamente o desempenho dos modelos. 

Os resultados iniciais podem são exibidos a seguir:

<div style="text-align: center;">
  <img src="./imagens/modelos_resultados_inciais.png" alt="figura 7" />
</div>

Os três melhores modelos foram o `GradientBoostingClassifier` seguido de `XGBClassifier` e `RandomForestClassifier`. Ambos tiveram a melhor acurácia, mas também os melhores valores de F1-Score na classe minoritária.

## Parte 5 Avaliação

Nesta fase iremos buscar ajustar os modelos anteriores buscando quais são os melhores hiperparâmetros para cada um deles, assim como também ir e avaliar os seus resultados. Infelizmente não houveram melhoras significativas entre as versões iniciais, o motivo é devido os dados ja estarem bem tratados, contendo poucos ou nenhum outliers. 

Contudo, quando avaliamos os modelos com todos os dados, notamos que o Random Forest consegue melhorar muito sua capacidade. Este modelo consegue tirar proveito do aumento da base de dados, já os outros dois modelos não tiveram um aumento significativo, os resultados podem ser comparados a seguir:

<div style="text-align: center;">
  <img src="./imagens/modelos_resultados_finais.png" alt="figura 9" />
</div>

É importante notar também que ao usar todo o banco de dados, conseguimos melhorar as previsões da Classe $1$, o que é um ótimo resultado. 
Iniciamos com uma baseline de $67.12\%$, obtendo $87.63\%$ ao usar os melhores hiperparâmetros e agora com toda a base de dados chegamos em $93\%$ de acurácia e $91\%$ de F1-Score na classe minoritária.

## Parte 6 - Aplicação 

### Parte 6.1 - MlFlow
Nesta ultima fase foi criar uma versão em script com o melhor modelo, salvo na pasta `src` como o nome `train_RF.py`. Utilizamos dos recursos do MlFlow para criarmos um cenário mais próximo de uma empresa. 

Foi criado environment chamado `mlflow_env` que irá atuar como o nosso servidor. Dentro do ambiente do Mlflow, criamos um modelo chamado **Churn-Abandono** e nosso primeiro experimento **churn_pedro_indicium**

<div style="text-align: center;">
  <img src="./imagens/mlflow_create_model.png" alt="figura 9" />
   <img src="./imagens/mlflow_create_exp.png" alt="figura 9" />
</div>

Note que temos duas runs, uma para os hiperparâmetros e outra run na qual o modelo treinou com todos os dados. 

Em seguida podemos registar o modelo que escolhemos, no caso o `RandomForest_com_todos_os_dados`:
<div style="text-align: center;">
  <img src="./imagens/mlflow_registro.png" alt="figura 9" />
</div>

Em seguida, colocamos o modelo para produção:
<div style="text-align: center;">
  <img src="./imagens/mlflow_production.png" alt="figura 9" />
</div>

### Parte 6.2 - Teste final


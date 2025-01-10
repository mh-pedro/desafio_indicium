# Desafio Data Science

## Resumo

Olá candidato, o objetivo deste desafio é testar os seus conhecimentos sobre construção de modelos preditivos. Queremos testar seus conhecimentos dos conceitos estatísticos de modelos preditivos, criatividade na resolução de problemas e aplicação de modelos de machine learning em produção.  É importante deixar claro que não existe resposta certa e que o que nos interessa é sua capacidade de descrever e justificar os passos utilizados na resolução do problema.

## Descrição do Problema
 
Seu objetivo é prever o *churn* (abandono de clientes) de um banco de dados fictício de uma instituição financeira. Para isso são fornecidos dois *datasets*: um *dataset* chamado **Abandono_clientes** composto por 10000 linhas e 13 colunas de informação (*features*), sendo uma coluna “Exited” composta por dados binários: 1 se o cliente abandonou o banco, 0 se não.  O segundo *dataset* possui 1000 linhas e 12 colunas e não possui a coluna “Exited”. **Seu objetivo é construir um pipeline de Machine Learning que permita prever essa coluna a partir dos dados enviados.** 

## Atividades

1. Descreva graficamente os dados disponíveis, apresentando as principais estatísticas descritivas. Comente o por quê da escolha dessas estatísticas.

2. Explique como você faria a previsão do **Churn** a partir dos dados. Quais variáveis e/ou suas transformações você utilizou e por quê? Qual tipo de problema estamos resolvendo (regressão, classificação)? Qual modelo melhor se aproxima dos dados e quais seus prós e contras? Qual medida de performance do modelo foi escolhida e por quê?

3. Construa um pipeline de machine learning que realize a previsão de churn a partir de um dataset CSV. Esse pipeline deve ser reproduzível e permitir realizar uma previsão a partir de qualquer arquivo CSV com a mesma estrutura de dados. O modelo utilizado pelo pipeline deve ser treinado com o dataset **Abandono_clientes.csv**. Você deve nos enviar o repositório de código com o pipeline.

4. Envie o resultado final do modelo em um arquivo com apenas duas colunas (rowNumber, predictedValues) gerado ao rodar o pipeline de 3 no dataset **abandono_teste.csv** em anexo.


## Introdução: O que é um Churn e para que serve?

É muito comum empresas enfrentarem problemas com perdas de clientes e/ou receitas, para quantificar o número de clientes perdidos podemos usar uma métrica denominada **churn rate**.

No mundo atual, temos um grande número de modelos de negócios por assinaturas (Netflix, Amazon, Spotify, etc.), poder identificar se o cliente pode ou não cancelar a assinatura tornou-se um problema de negócios.O Churn é entendido como um índice de cancelamentos de clientes que cancelam em determinado período de tempo. Para podermos calcular o Churn, o que precisamos fazer é somar o número de clientes que cancelou seu produto/serviço no período analisado. 

Desta forma, se uma empresa deseja fazer uma expansão da base de clientes, é preciso que o número de novos clientes exceda o seu churn rate, melhor dizendo, a adesão de novos clientes deve ser maior do que a taxa de cancelamentos.

### Qual deve ser a taxa ideal de Churn?

Entendido, o que é um churn e qual seu papel, uma questão que surge é: **Qual deve ser a taxa ideal de Churn?**

Podemos dizer que a melhor taxa de Churn seria 0%, pois isso significa que não temos clientes realizando cancelamento, contudo, no mundo real isto não é possível. Para entendermos qual deve ser o valor ideal, precisamo entender o seguimento de mercado que estamos envolvidos.

Não existe um valor exato para a taxa de Churn, pois podemos ter diversos fatores que influenciam no Churn, no entanto, de forma geral, pode-se dizer que *5% é uma taxa aceitável** para empresas que trabalham com recorrência, de acordo com [David Pakman](https://pakman.com/churn-is-the-single-metric-that-determines-the-success-of-your-subscription-service-6e82d9d9ea01), cocriador do Music Group da Apple.

### O que levam os clientes a cancelar?

Entender os fatores que levam os clientes a cancelar e os fatores que levam os clientes a aderir ao negócio pode ser uma tarefa difícil, pois muitos fatores podem ser combinados para chegar a uma conclusão.

No entanto, alguns fatores que levam os clientes a cancelar podem ser identificados:

- O cliente está sem fluxo de caixa e não pode mais arcar com a mensalidade do seu produto/serviço;
- Ele não consegue ver valor no produto/serviço;
- O cliente não teve suas expectativas atendidas;
- O produto ou serviço não acompanha as evoluções de mercado e perde em - qualidade e ferramentas;
- O produto é bom, mas o serviço não – e vice versa;
- Seu cliente optou pelo produto da concorrência;
- Seu cliente foi adquirido por outra empresa e o comprador usa outro serviço;
- Crise financeira no mercado;
- Interrupção da operação durante um período de crise;
- Seu cliente faliu.

Portanto, para um negócio ser saudável é importante conhecer os indicativos de uso do produto, qualidade do serviço, preço, competição e etc.

## Base de dados

Os dados que vamos utilizar neste projeto foram fornecidos pelo idealizador do desafio e simula uma empresa de banco fictícia. Esses dados apresentam as seguintes características descritas na tabela:

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

A variável target (alvo) será a variável **Exited**. Nossa tarefa consiste em prever se um cliente vai abandonar o banco ou continuar com ele.

## Planejamento do Desafio

Vamos aplicar o método CRIS-DS que é uma variável do método CRISP-DM (Cross-Induvidual System Process for Data Mining) adaptando para o cenário de ciência de dados.

O método CRIS-DS consiste pode ser visualizado na figura abaixo:
![figura 1](images/CRIS-DS.png)

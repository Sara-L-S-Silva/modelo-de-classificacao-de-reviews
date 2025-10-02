# modelo-de-classificacao-de-reviews
[Projeto pessoa] IA capaz de ler reviews de um csv do Kaggle e dizer quais são positivas, quais são negativas e o motivo.

Como testar ambos os algoritmos?

1. Clonar o repositório.
2. Rodar "pip install requirements.txt" no terminal da IDE.
3. Rodar "python3 modelo.py" 

📌 Projeto de Classificação de Textos

Este projeto tem como objetivo desenvolver um modelo de classificação automática de textos de review, explorando diferentes algoritmos de aprendizado de máquina e técnicas de pré-processamento, nesse caso, Naive Bayes e Regressão Logística. O foco foi avaliar o desempenho de modelos clássicos em NLP (Processamento de Linguagem Natural) e apontar se a review foi realmente positiva ou negativa.

🚀 Raciocínio e Escolhas Técnicas
1. Representação dos dados (TF-IDF)

Para transformar os textos em representações numéricas, utilizei o TF-IDF Vectorizer, que avalia a relevância de cada termo em um documento em relação ao corpus. Essa escolha foi feita porque o TF-IDF reduz a influência de palavras muito comuns e destaca termos mais discriminativos, sendo mais eficiente que a simples contagem de palavras.

2. Modelos de classificação testados

Naive Bayes (MultinomialNB): escolhido como baseline, pois é um algoritmo rápido, simples e tradicionalmente eficaz em classificação de textos. Apesar de suas limitações (assume independência entre features), fornece uma boa referência inicial de desempenho e contrapõe bem o algoritmo de regressão logística, focada em definir pesos e saber relações entre palavras.

Regressão Logística: selecionada por ser um modelo mais robusto e interpretável, que aprende pesos para cada feature sem assumir independência entre elas, diferentemente do Naive Bayes. Em bases de dados maiores, tende a superar o Naive Bayes em precisão e recall. Nos resultados com 1000 iterações, tive como reaultado que a regressão logística funcionou melhor.

3. Divisão dos dados

Utilizei o train_test_split do scikit-learn para separar os dados em treino e teste, garantindo a avaliação justa dos modelos.

4. Métricas de avaliação

Adotei o F1-score como principal métrica de análise, por ser mais adequada em casos de classes desbalanceadas, já que combina precisão e recall, bem como a support para saber a volumetria dos dados. Além disso, usei o classification_report para detalhar o desempenho por classe.

5. Tratamento do desbalanceamento de classes

O dataset apresentou forte desbalanceamento, o que resultava em baixo desempenho para a classe minoritária. Para corrigir isso, utilizei o parâmetro class_weight='balanced' na Regressão Logística. Essa técnica ajusta o peso das classes na função de custo, permitindo que o modelo dê mais atenção às classes menos representadas.

Essa decisão foi crucial para melhorar o F1-score dos exemplos negativos, que inicialmente estava insatisfatório (abaixo de 70%).

🛠️ Tecnologias Utilizadas

Python como linguagem principal.
Pandas para manipulação de dados.
scikit-learn para vetorização de texto, implementação dos modelos e métricas de avaliação.

📊 Conclusões

O Naive Bayes cumpriu bem seu papel como baseline, mas apresentou limitações na classe minoritária.

A Regressão Logística com balanceamento por peso trouxe ganhos significativos em recall e F1-score, mostrando-se a melhor abordagem para este problema e o uso do TF-IDF foi fundamental para destacar palavras relevantes e melhorar a qualidade da representação textual.

Principais reclamações incluem termos que referenciam a logística de entrega dos produtos e se o que queriam chegou.

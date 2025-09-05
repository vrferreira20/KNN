import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

# Carregando os dados
mtcars = pd.read_csv('mt_cars.csv')

# Variáveis independentes
x = mtcars[['mpg','hp']].values
# Variável dependente
y = mtcars['cyl'].values

# Criando o modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
modelo = knn.fit(x, y)

# Realizando a previsão 
y_prev = modelo.predict(x) 
#print(y_prev)

# Novo ponto para previsão
novos_dados = np.array([[19.3, 105]])
previsao = modelo.predict(novos_dados)
print(f"Previsão para o novo ponto: {previsao}")

accuracy = accuracy_score(y, y_prev)
precision = precision_score(y, y_prev, average='weighted')
recall = recall_score(y, y_prev, average='weighted')
f1 = f1_score(y, y_prev, average='weighted')
cm = confusion_matrix(y, y_prev)
#print(f'Acurácia: {accuracy}, \nPrecisão: {precision}, \nRevocação: {recall}, \nF1-Score: {f1}, \nMatriz de Confusão:\n{cm}')

# Encontrando vizinhos mais próximos
distancias, indices = modelo.kneighbors(novos_dados)
vizinhos = mtcars.loc[indices[0], ["cyl", "mpg", "hp"]]
print("Vizinhos mais próximos:\n", vizinhos)

# --- Plotando ---
plt.figure(figsize=(8,6))

# Pontos do dataset coloridos por cilindros
scatter = plt.scatter(x[:,0], x[:,1], c=y, cmap="viridis", s=80, edgecolors="k")

# Novo ponto em X vermelho
plt.scatter(novos_dados[0,0], novos_dados[0,1], c="red", marker="X", s=200, label="Novo ponto")

# Vizinhos mais próximos (círculos vermelhos)
plt.scatter(x[indices[0],0], x[indices[0],1], facecolors="none", edgecolors="red", s=300, linewidths=2, label="Vizinhos próximos")

# Adicionando labels para os vizinhos (mostrando cilindros)
for i, idx in enumerate(indices[0]):
    plt.text(x[idx,0]+0.3, x[idx,1]+0.3, f"{y[idx]} cyl", color="red", fontsize=12, fontweight="bold")

# Ajustes do gráfico
plt.xlabel("mpg")
plt.ylabel("hp")
plt.title("KNN - Vizinhos mais próximos")
plt.legend()
plt.colorbar(scatter, label="Número de cilindros (cyl)")
plt.show()

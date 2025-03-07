import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_excel('segnali1.xlsx')
df = df.set_index(df.columns[0])

# Trova le differenze superiori al 20%
riferimento = df.iloc[:, 0]  # Prima colonna come riferimento
outliers = []

for col in df.columns[1:]:
    diff = np.abs(df[col] - riferimento)
    soglia = riferimento * 0.2  # 20% della colonna di riferimento

    # Trova i punti che superano la soglia
    mask = diff > soglia
    for i in df.index[mask]:
        outliers.append((i, riferimento[i], df[col][i]))

# Plot dei punti che superano la soglia
plt.figure(figsize=(8, 6))
for i, ref, val in outliers:
    plt.scatter(i, ref, color='blue', label="Riferimento" if i == 0 else "")
    plt.scatter(i, val, color='red', label="Valore fuori soglia" if i == 0 else "")
    plt.plot([i, i], [ref, val], 'k--')

plt.xlabel("Indice")
plt.ylabel("Valori")
plt.title("Valori che superano il 20% di differenza dal riferimento")
plt.legend()
plt.grid(True)
plt.show()

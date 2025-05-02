import numpy as np
import matplotlib.pyplot as plt


def plot_dataframe(df): #segnale di baseline rosso, altri segnali grigi e fini

    fig, ax = plt.subplots(figsize=(20, 11))
    ax.plot(df.index, df.iloc[:, 0], color='red', linewidth=2, label=df.columns[0])
    for colonna in df.columns[1:]:
        ax.plot(df.index, df[colonna], color='gray', linewidth=0.5, label=colonna)

    plt.xlabel('ms')
    plt.ylabel('microvolt')
    plt.title('Grafico con colonne sovrapposte')
    #plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature_tobs(df): #insieme di grafici con ordinata il valore del target e come ascissa il valore della feature
    n_features = df.shape[0] - 1

    # Calcola il numero di righe e colonne per i subplot
    ncols = int(np.ceil(np.sqrt(n_features))) #ceil per numeri interi
    nrows = int(np.ceil(n_features / ncols))
    fig, ax = plt.subplots(nrows, ncols)


    ax = ax.flatten()

    for a in range(n_features):
        ax[a].scatter(df.iloc[a + 1], df.iloc[0])
        ax[a].axhline(0.5, color='r', linestyle='--')

    for i in range(n_features, nrows * ncols):
        fig.delaxes(ax[i])
    plt.show()

def plot_feat(df, warning):
    nomi_segnali = df.columns.tolist()
    prima_riga = df.iloc[0]  # Ottieni i valori della prima riga

    for index, valori_feature in df.iterrows():
        plt.figure(figsize=(10, 6))  # Crea una nuova figura per ogni feature

        # Usa un indice numerico per l'asse x
        x_values = range(len(nomi_segnali))

        for i, (nome_segnale, valore) in enumerate(zip(nomi_segnali, valori_feature.values)):
            colore = 'red' if prima_riga.iloc[i] > 0.5 else 'blue'  # Imposta il colore in base alla prima riga
            plt.plot(x_values[i], valore, marker='o', linestyle='-', color=colore)

        if warning is not None:
            for idx in warning:
                plt.axvline(x = idx, color='red', linestyle= '--', linewidth= 2)
        # Imposta le etichette dell'asse x
        plt.xticks(x_values, nomi_segnali, rotation=45)  # Ruota le etichette per una migliore leggibilità
        plt.title(f'Feature: {index}')
        plt.xlabel('Numero del Segnale')
        plt.ylabel('Valore della Feature')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
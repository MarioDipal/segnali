import pandas as pd
import matplotlib.pyplot as plt

def plot_dataframe(df):
    """
    Crea un grafico in cui la prima colonna del DataFrame è colorata di rosso e ben
    visibile, e le altre sono in tratto fine e grigio.

    Args:
        df (pd.DataFrame): Il DataFrame da visualizzare.
    """

    fig, ax = plt.subplots(figsize=(20, 11))

    # Colora la prima colonna di rosso e rendila ben visibile
    ax.plot(df.index, df.iloc[:, 0], color='red', linewidth=2, label=df.columns[0])

    # Rendi le altre colonne in tratto fine e grigio
    for colonna in df.columns[1:]:
        ax.plot(df.index, df[colonna], color='gray', linewidth=0.5, label=colonna)

    plt.xlabel('ms')
    plt.ylabel('microvolt')
    plt.title('Grafico con colonne sovrapposte')
    #plt.legend()
    plt.grid(True)
    plt.show()

###########################################


def plot_feature(df):
    n_features = df.shape[0] - 1  # Numero di feature (righe tranne il target)

    fig, ax = plt.subplots(3, 3)
    ax = ax.flatten()

    a = 0
    while a < n_features:
        ax[a].scatter(df.iloc[a+1], df.iloc[0])
        ax[a].axhline(0.5, color='r', linestyle='--')
        a = a+1

    plt.show()

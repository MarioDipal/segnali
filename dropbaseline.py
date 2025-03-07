import pandas as pd
###########################################

def dbs(df):
    baseline = df.iloc[:, 1]
    booldrop = [] #riga di target

    for col in df.columns:
        if col != 0 and col != 1:
            drop = abs(df[col] - baseline)/baseline
            booldrop.append(1 if (drop > 0.5).any() else 0)

    return booldrop
###########################################


def featured_df(df):
    """
    Confronta i punti più bassi dei segnali e crea un DataFrame di risultati.

    Args:
        df (pd.DataFrame): DataFrame con il segnale di riferimento nella prima colonna.

    Returns:
        pd.DataFrame: DataFrame con una singola riga di 1 e 0.
    """

    riferimento = df.iloc[:, 0]  # Segnale di riferimento (prima colonna)
    valore_minimo_riferimento = riferimento.min()  # Valore minimo del riferimento

    risultati = []
    for col in df.columns[1:]:  # Itera su tutte le colonne tranne la prima
        valore_minimo_segnale = df[col].min()  # Valore minimo del segnale corrente
        differenza = abs(valore_minimo_segnale - valore_minimo_riferimento)
        percentuale_differenza = differenza / abs(valore_minimo_riferimento)

        if percentuale_differenza > 0.5:
            risultati.append(1)
        else:
            risultati.append(0)

    # Crea un DataFrame con una singola riga di risultati
    nuovo_df = pd.DataFrame([risultati], columns=df.columns[1:])
    return nuovo_df

###############################
def add_list(df, lista_da_aggiungere, nome_riga):
    df.loc[nome_riga] = lista_da_aggiungere


###############################
def featured_df_continuo(df):

    riferimento = df.iloc[:, 0]  # Segnale di riferimento (prima colonna)
    valore_minimo_riferimento = riferimento.min()  # Valore minimo del riferimento

    risultati = []
    for col in df.columns[1:]:  # Itera su tutte le colonne tranne la prima
        valore_minimo_segnale = df[col].min()  # Valore minimo del segnale corrente
        differenza = abs(valore_minimo_segnale - valore_minimo_riferimento)
        percentuale_differenza = differenza / abs(valore_minimo_riferimento)

        risultati.append(percentuale_differenza)

    # Crea un DataFrame con una singola riga di risultati
    nuovo_df = pd.DataFrame([risultati], columns=df.columns[1:])
    return nuovo_df
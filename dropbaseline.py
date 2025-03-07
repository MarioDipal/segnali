import pandas as pd
###########################################
def featured_df(df): #crea un nuovo df per le feature ed il target

    riferimento = df.iloc[:, 0]  # Segnale di riferimento
    valore_minimo_riferimento = riferimento.min()  # Valore minimo del riferimento

    risultati = []
    for col in df.columns[1:]:  # Itera su tutte le colonne tranne la prima
        valore_minimo_segnale = df[col].min()  # Valore minimo del segnale corrente
        differenza = abs(valore_minimo_segnale - valore_minimo_riferimento)
        percentuale_differenza = differenza / abs(valore_minimo_riferimento)

        #risultati.append(percentuale_differenza) #variante con risultati continui

        if percentuale_differenza > 0.5:
            risultati.append(1)
        else:
            risultati.append(0)

    nuovo_df = pd.DataFrame([risultati], columns=df.columns[1:])
    return nuovo_df

###############################
def add_list(df, lista_da_aggiungere, nome_riga):
    df.loc[nome_riga] = lista_da_aggiungere



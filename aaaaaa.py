import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import periodogram
from statsmodels.tsa.ar_model import AutoReg

def plot_dataframe(df):  # segnale di baseline rosso, altri segnali grigi e fini

    fig, ax = plt.subplots(figsize=(20, 11))

    ax.plot(df.index, df.iloc[:, 0], color='red', linewidth=2, label=df.columns[0])

    for colonna in df.columns[1:]:
        ax.plot(df.index, df[colonna], color='gray', linewidth=0.5, label=colonna)

    plt.xlabel('ms')
    plt.ylabel('microvolt')
    plt.title('Grafico con colonne sovrapposte')
    # plt.legend()
    plt.grid(True)
    plt.show()

def plot_feature(df):  # insieme di grafici con ordinata il valore del target e come ascissa il valore della feature
    n_features = df.shape[0] - 1

    fig, ax = plt.subplots(n_features, 1)
    a = 0

    while a < n_features:
        ax[a].scatter(df.iloc[a + 1], df.iloc[0])
        a = a + 1

    plt.show()

def featured_df(df):  # crea un nuovo df per le feature ed il target

    riferimento = df.iloc[:, 0]  # Segnale di riferimento
    valore_minimo_riferimento = riferimento.min()  # Valore minimo del riferimento

    risultati = []
    for col in df.columns[1:]:  # Itera su tutte le colonne tranne la prima
        valore_minimo_segnale = df[col].min()  # Valore minimo del trial corrente
        differenza = abs(valore_minimo_segnale - valore_minimo_riferimento)
        percentuale_differenza = differenza / abs(valore_minimo_riferimento)

        if percentuale_differenza > 0.5:
            risultati.append(1)
        else:
            risultati.append(0)

    nuovo_df = pd.DataFrame([risultati], columns=df.columns[1:])
    return nuovo_df

def add_list(df, lista_da_aggiungere, nome_riga):  # aggiunge le feature al df
    df.loc[nome_riga] = lista_da_aggiungere

def will_ampl(df, threshold):
    reference = df.iloc[0].values
    counts = (np.abs(df - reference) > threshold).sum(axis=0)
    return counts.values

def max_fact_len(df):
    mfl_values = []
    for col in df.columns:
        signal = df[col].values
        last_value = signal[-1]
        sum_squared_diff = np.sum((signal - last_value) ** 2)
        mfl = np.log10(np.sqrt(sum_squared_diff))
        mfl_values.append(mfl)

    return mfl_values

def peak_frequency(df, fs):

    peak_freqs = []

    for col in df.columns:
        signal = df[col].values
        freqs, power = periodogram(signal, fs=fs)
        peak_freq = freqs[np.argmax(power)]
        peak_freqs.append(peak_freq)

    return peak_freqs

def waveform_lenght(df):
    waveform_lengths = df.diff().abs().sum()
    return waveform_lengths.tolist()

def slope_sign_change(df):
    def count_slope_sign_changes(column):
        return np.sum((np.sign(np.diff(column))[:-1] != np.sign(np.diff(column))[1:]))

    slope_sign_changes = df.apply(lambda col: count_slope_sign_changes(col.dropna()))
    return slope_sign_changes.values

def mean_abs_value(df):
    mav_values = []
    for col in df.columns:
        signal = df[col].values
        mav = ((abs(signal)).sum()) / len(df)
        mav_values.append(mav)
    return mav_values

def root_mean_square(df):
    rms_values = []
    for col in df.columns:
        signal = df[col].values
        rms = (((signal ** 2).sum()) / len(df)) ** 0.5
        rms_values.append(rms)
    return rms_values

def ar_coeff(df, order = 4):
    mean_coeffs = []
    var_coeffs = []
    for col in df.columns:
        signal = df[col].values
        model = AutoReg(signal, lags=order).fit()
        ar_coeff = model.params
        mean_coeffs.append(np.mean(ar_coeff))
        var_coeffs.append((np.var(ar_coeff)))
    return mean_coeffs, var_coeffs

df = pd.read_excel('segnali excel/2022p65cb_PCCortBulb Trace Data.xlsx') #scelta segnale

df = df.set_index(df.columns[0])
plot_dataframe(df) #mostra il dataframe

featdf = pd.DataFrame(featured_df(df)) #crea il df delle feature
featdf = featdf.rename(index={0: 'Target'})
featdf.insert(0, "baseline", 0)

add_list(featdf,will_ampl(df, 112), 'Willison')
add_list(featdf, max_fact_len(df), 'Max Fract Len')
add_list(featdf, waveform_lenght(df), 'Waveform Len')
add_list(featdf, slope_sign_change(df), 'Slope Change')
add_list(featdf, peak_frequency(df, 200), 'Freq Picco')
add_list(featdf, mean_abs_value(df), 'Mean Abs Value')
add_list(featdf, root_mean_square(df), 'Root Mean Square')
mean_coeffs, var_coeffs = ar_coeff(df)
add_list(featdf, mean_coeffs, 'Media AR')
add_list(featdf, var_coeffs, 'Varianza AR')

plot_feature(featdf)



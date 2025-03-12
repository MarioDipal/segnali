import numpy as np
from scipy.signal import periodogram, welch
from statsmodels.tsa.ar_model import AutoReg
###########################################

def will_ampl(df):
    threshold = np.mean(df.iloc[0])
    reference = df.iloc[0].values
    counts = (np.abs(df - reference) > threshold).sum(axis=0)
    return counts.values

###########################################
def max_fact_len(df):

    mfl_values = []
    for col in df.columns:
        signal = df[col].values
        last_value = signal[-1]
        sum_squared_diff = np.sum((signal - last_value) ** 2)
        mfl = np.log10(np.sqrt(sum_squared_diff))
        mfl_values.append(mfl)

    return mfl_values

###########################################
def peak_frequency(df, fs): #esegue la trasformazione in frequenza

    peak_freqs = []

    for col in df.columns:
        signal = df[col].values
        freqs, power = periodogram(signal, fs=fs)
        peak_freq = freqs[np.argmax(power)]
        peak_freqs.append(peak_freq)

    return peak_freqs

###########################################
def waveform_lenght(df):
    waveform_lengths = df.diff().abs().sum()
    return waveform_lengths.tolist()

###########################################
def slope_sign_change(df):
    slope_sign_changes = []
    for col_name in df.columns:
        column = df[col_name].dropna()
        diffs = np.diff(column)
        signs = np.sign(diffs)
        sign_changes = np.sum(signs[:-1] != signs[1:])
        slope_sign_changes.append(sign_changes)
    return slope_sign_changes
##########################################
def mean_abs_value(df):
    mav_values = []
    for col in df.columns:
        signal = df[col].values
        mav = ((abs(signal)).sum()) / len(df)
        mav_values.append(mav)
    return mav_values

###########################################
def root_mean_square(df):
    rms_values = []
    for col in df.columns:
        signal = df[col].values
        rms = (((signal ** 2).sum()) / len(df)) ** 0.5
        rms_values.append(rms)
    return rms_values


def roots_mean_square(df):
    rms_values = {col: np.sqrt(np.mean(df[col].values ** 2)) for col in df.columns}
    return rms_values

###########################################
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
###########################################

def mean_med_freq(df, fs):
    freq_media = []
    freq_mediana = []
    for col in df.columns:
        signal = df[col].values
        freq, psd = welch(signal, fs = fs) #calcolo densità spettrale di potenza
        cdf = np.cumsum(psd) / np.sum(psd) #calcola la somma comulativa degli elementi, ogni elemento è la somma di tutti i precendit
        freq_media.append(np.sum(freq * psd)/np.sum(psd))
        freq_mediana.append(freq[np.where(cdf >=0.5)[0][0]])

    return  freq_media, freq_mediana
###########################################

def av_ampl_cha(df):
    aac = []
    for col in df.columns:
        signal = df[col].values
        aac.append(np.mean(np.abs(np.diff(signal))))
    return aac
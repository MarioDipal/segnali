import numpy as np
import pywt
from scipy.signal import periodogram, welch
from statsmodels.tsa.ar_model import AutoReg
###########################################

def will_ampl(df):
    threshold = None
    results = []
    for col in df.columns:
        signal = df[col].values
        diff = np.diff(signal, n=1)  # calcola la differenza tra campioni successivi.
        if threshold is None:
            threshold = np.mean(np.abs(diff))  # calcola la soglia dinamicamente.
        counts = np.sum(np.abs(diff) > threshold)
        results.append(counts)
    return (results/results[0])
###########################################
def max_fact_len(df):

    mfl_values = []
    for col in df.columns:
        signal = df[col].values
        last_value = signal[-1]
        sum_squared_diff = np.sum((signal - last_value) ** 2)
        mfl = np.log10(np.sqrt(sum_squared_diff))
        mfl_values.append(mfl)

    return (mfl_values/mfl_values[0])

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
    wave_len = []
    for col in df.columns:
        signal = df[col].values
        diff = np.diff(signal)  # calcola le differenze tra campioni successivi
        w_l = np.abs(diff).sum()  #somma i valori assoluti delle differenze
        wave_len.append(w_l)
    return (wave_len/wave_len[0])
###########################################
def slope_sign_change(df):
    threshold = 0.5 #soglia per limitare il rumore
    slope_sign_changes = []
    for col_name in df.columns:
        column = df[col_name].dropna().values
        sign_changes = 0
        for i in range(1, len(column) - 1):
            diff1 = column[i] - column[i - 1]
            diff2 = column[i + 1] - column[i]
            if diff1 * diff2 < 0 and abs(column[i + 1] - column[i - 1]) >= threshold: #verifica cambio di segno e superamento della soglia
                sign_changes += 1

        slope_sign_changes.append(sign_changes)
    slope_sign_changes = [val / slope_sign_changes[0] for val in slope_sign_changes] #normalizzaszione rispetto alla baseline
    return slope_sign_changes
##########################################
def mean_abs_value(df):
    mav_values = []
    for col in df.columns:
        signal = df[col].values
        mav = ((abs(signal)).sum()) / len(df)
        mav_values.append(mav)
    return (mav_values/mav_values[0])

###########################################
def root_mean_square(df):
    rms_values = []
    for col in df.columns:
        signal = df[col].values
        rms = (((signal ** 2).sum()) / len(df)) ** 0.5
        rms_values.append(rms)
    return (rms_values / rms_values[0])


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
    return (mean_coeffs/mean_coeffs[0]), (var_coeffs / var_coeffs[0])
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

    return  (freq_media/freq_media[0]), (freq_mediana/freq_mediana[0])
###########################################

def av_ampl_cha(df):
    aac = []
    for col in df.columns:
        signal = df[col].values
        aac.append(np.mean(np.abs(np.diff(signal))))
    return (aac / aac[0])
###########################################

def wavelet_correlations(df, fs):
    freq_range = (20, 150) #range frequenza normali emg
    baseline = df.iloc[:, 0].values
    media = [0]
    for col in df.columns[1:]:
        signal = df[col].values
        scales = np.arange(1, 128)
        coef_base, freqs = pywt.cwt(baseline, scales, 'cmor1.0-0.5', 1 / fs) #trasformata wavelet di Morlet
        coef_signal, _ = pywt.cwt(signal, scales, 'cmor1.0-0.5', 1 / fs)
        mode_scales = (freqs >= freq_range[0]) & (freqs <= freq_range[1]) #trova le scale che corrispondono alla banda di media frequenza
        corr_values = []
        for scale_idx in np.where(mode_scales)[0]: #calcola la correlazione nella banda di media frequenza
            corr = np.corrcoef(np.abs(coef_base[scale_idx]), np.abs(coef_signal[scale_idx]))[0, 1]
            corr_values.append(corr)
        media.append(np.mean(corr_values))
    return media

def MinR(df, n_segments):
    results = [0]
    baseline = df.iloc[:, 0].values
    def extract_features(signal, n_segments):
        segment_length = len(signal) // n_segments
        features = []
        for i in range(n_segments):
            segment = signal[i * segment_length:(i + 1) * segment_length]
            mean = np.mean(segment)
            std_dev = np.std(segment)
            features.append((mean, std_dev))
        return np.array(features)
    def calculate_distance(features1, features2):
        dist = 0
        for (mean1, std1), (mean2, std2) in zip(features1, features2):
            dist += (mean1 - mean2) ** 2 + (np.sqrt(std1) - np.sqrt(std2)) ** 2
        return np.sqrt(dist)

    baseline_features = extract_features(baseline, n_segments)

    for col in df.columns[1:]:
        signal = df[col].values
        signal_features = extract_features(signal, n_segments)
        dist = calculate_distance(baseline_features, signal_features)
        results.append(dist)

    return results
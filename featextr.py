import numpy as np
import pywt
from scipy.signal import periodogram, welch, hilbert, stft
from statsmodels.tsa.ar_model import AutoReg
from sympy import divisor_sigma
from scipy.interpolate import CubicSpline
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

    return (peak_freqs / peak_freqs[0])

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
    threshold = 0.3 #soglia per limitare il rumore
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
    if slope_sign_changes [0] != 0:
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
    media = [1]
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
###########################################

def MinR(df, n_segments):
    results = [1]
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
        dist = 1
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
###########################################

def hilbert_tras(df):
    fs = 200 #frequenza di campionamento
    rms_amp_envelopes = []
    plv_inst_phases = [] #phase locking value quanto due segnali rimangono in fase tra loro nel tempo
    rms_inst_freqs = []

    first_signal = df.iloc[:, 0]  # Primo segnale nel DataFrame
    first_analytic_signal = hilbert(first_signal)
    first_phase = np.unwrap(np.angle(first_analytic_signal))  # Fase istantanea del primo segnale

    for col in df.columns:
        signal = df[col]
        analytic_signal = hilbert(signal)
        amplitude_envelope = (np.abs(analytic_signal)) # Modulo del segnale analitico
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))  # Fase istantanea
        instantaneous_frequency = (np.diff(np.unwrap(instantaneous_phase)) * fs / (2 * np.pi) ) # Frequenza istantanea
        rms_amp_envelopes.append(np.sqrt(np.mean(amplitude_envelope ** 2)))
        rms_inst_freqs.append(np.sqrt(np.mean(instantaneous_frequency ** 2)))
        plv = np.abs(np.mean(np.exp(1j * (first_phase - instantaneous_phase))))
        plv_inst_phases.append(plv)

    return (rms_amp_envelopes / rms_amp_envelopes[0]), (rms_inst_freqs / rms_inst_freqs[0]), plv_inst_phases
###########################################

def discrete_wavelet_tr(df):
    cA_vec = []
    cD_vec = []
    for col in df.columns:
        signal = df[col]
        wavelet = 'db4'
        coeffs = pywt.dwt(signal, wavelet)
        cA, cD = coeffs #approssimazione a bassa ed alta frequenza
        cA_vec.append(cA)
        cD_vec.append(cD)
    cA_mean = (np.sum(cA_vec, axis=1))/len(cA_vec)
    cA_mean = cA_mean/cA_mean[0]
    cD_mean = (np.sum(cD_vec, axis=1))/len(cD_vec)
    cD_mean = cD_mean / cD_mean[0]
    return cA_mean, cD_mean

def short_time_ft(df):
    fs = 200 #f_c
    mav_mag = []
    mav_pha = []
    max_mag=[]
    for col in df.columns:
        signal = df[col]
        nseg = np.round(len(signal)/35) #numero campioni per finestra
        f, t, Zxx = stft(signal, fs, nperseg=nseg)
        magnitude = np.abs(Zxx)  # Modulo della STFT
        phase = np.angle(Zxx)  # Fase della STFT
        mav_magn = ((abs(magnitude)).sum()) / len(magnitude)
        mav_phas = ((abs(phase)).sum()) / len(phase)
        max_magn = np.max(magnitude)
        mav_mag.append(mav_magn)
        mav_pha.append(mav_phas)
        max_mag.append(max_magn)
    return (mav_mag/mav_mag[0]), (mav_pha/mav_pha[0]), (max_mag/max_mag[0])

def ramunjan_ft(df):
    rft = []
    for col in df.columns:
        signal = df[col].to_numpy()
        N= len(signal)
        RFT = np.zeros(N, dtype=complex)
        for k in range(1, N + 1):
            sum_k = 0
            for n in range(1, N + 1):
                sum_k += signal[n - 1] * np.exp(-2j * np.pi * (k - 1) * n / N) * divisor_sigma(n)
            RFT[k - 1] = sum_k / N

        rft.append(np.sum(np.abs(RFT)) / N)
    return rft

def intrinsic_tcd (df):
    componenti = []
    residui = []

    for col in df.columns:
        signal = df[col]
        components = []
        residuo = []
        residual = np.array(signal.copy())

        for _ in range(10): # 10 numero massimo di iterazioni
            peaks = np.where((np.roll(residual, 1) < residual) & (np.roll(residual, -1) < residual))[0] # trova gli estremi locali, [0] per farlo diventare una lista
            valleys = np.where((np.roll(residual, 1) > residual) & (np.roll(residual, -1) > residual))[0]
            if len(peaks) < 2 or len(valleys) < 2:
                break  # interrompe se non ci sono abbastanza punti

            upper_envelope = CubicSpline(peaks, residual[peaks])(np.arange(len(signal))) # interpolazione degli estremi
            lower_envelope = CubicSpline(valleys, residual[valleys])(np.arange(len(signal)))
            baseline = (upper_envelope + lower_envelope) / 2 #calcola la componente
            component = residual - baseline
            components.append(component)
            residual = baseline  # aggiorna il residuo

        residuo.append(residual)  # aggiunge il residuo finale

        componenti.append(np.sum(np.abs(components)) / len(components))
        residui.append((np.sum(np.abs(residuo)) / len(residuo)))
    return (componenti / componenti[0]), (residui / residui[0])

def svd (df):
    signals = df.iloc[:, 1:]  # Tutte le righe, dalla seconda colonna in poi
    baseline = df.iloc[:, 0]
    U, S, Vt = np.linalg.svd(signals, full_matrices=False)
    reference_projection = np.dot(U.T, baseline)  #proietta il segnale di riferimento sulle componenti principali
    signals_projection = np.dot(U.T, signals)  #proiettia tutti i segnali
    cosine_similarities = [1]
    euclidean_distances = [1]
    for i in range(signals.shape[1]):
        cos_sim = np.dot(reference_projection, signals_projection[:, i]) / (
                np.linalg.norm(reference_projection) * np.linalg.norm(signals_projection[:, i])
        )
        cosine_similarities.append(cos_sim)
        eucl_dist = np.linalg.norm(reference_projection - signals_projection[:, i])
        euclidean_distances.append(eucl_dist)

    return cosine_similarities, np.array(euclidean_distances) * (0.001) # divido per mille in modo da avere valori vicino a 1



